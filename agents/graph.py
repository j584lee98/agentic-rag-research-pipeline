import logging
import statistics
from dataclasses import dataclass, field
from typing import Callable, Literal, NotRequired, TypedDict

import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Deterministic statistics computed from raw retrieval distances."""

    chunk_count: int
    # OpenAI embeddings are L2-normalised; similarity = 1 - l2_distance / 2 maps to [0, 1]
    similarity_scores: list[float] = field(default_factory=list)
    min_score: float = 0.0
    max_score: float = 0.0
    mean_score: float = 0.0
    median_score: float = 0.0
    above_threshold_count: int = 0  # scores >= SIMILARITY_THRESHOLD
    coverage_verdict: Literal["sufficient", "partial", "insufficient"] = "insufficient"


# Minimum similarity score (0-1) to consider a chunk meaningfully relevant
SIMILARITY_THRESHOLD = 0.50


class AgentState(TypedDict):
    prompt: str
    response: str
    route: NotRequired[Literal["direct", "reason"]]
    context: NotRequired[str]
    retrieval_distances: NotRequired[list[float]]  # raw L2 distances from ChromaDB
    retrieval_diagnostics: NotRequired[RetrievalDiagnostics]


class RouteDecision(BaseModel):
    route: Literal["direct", "reason"]


@dataclass(frozen=True)
class AgentRuntime:
    model_name: str
    embedding_model: str
    llm_factory: Callable[[str], ChatOpenAI]


def _default_llm_factory(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(model=model_name)


def build_default_runtime() -> AgentRuntime:
    settings = get_settings()
    return AgentRuntime(
        model_name=settings.openai_model,
        embedding_model=settings.openai_embedding_model,
        llm_factory=_default_llm_factory,
    )


def _get_collection():
    settings = get_settings()
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    return client.get_or_create_collection(name=settings.chroma_collection_name)


def _format_retrieved_context(
    documents: list[str], metadatas: list[dict[str, object] | None]
) -> str:
    formatted_chunks: list[str] = []

    for idx, document in enumerate(documents):
        metadata = metadatas[idx] if idx < len(metadatas) else None
        source = "unknown"

        if metadata:
            filename = metadata.get("filename")
            chunk_index = metadata.get("chunk_index")
            chunk_count = metadata.get("chunk_count")
            if filename is not None:
                source = str(filename)
            if chunk_index is not None and chunk_count is not None:
                source = f"{source} (chunk {chunk_index}/{chunk_count})"

        formatted_chunks.append(
            f"[Context {idx + 1} | source: {source}]\n{document.strip()}"
        )

    return "\n\n".join(formatted_chunks)


def _retrieve_raw(
    prompt: str, embedding_model: str, top_k: int = 4
) -> tuple[list[str], list[dict[str, object] | None], list[float]]:
    """Return (documents, metadatas, l2_distances) for the top-k results."""
    collection = _get_collection()
    embedding_client = OpenAIEmbeddings(model=embedding_model)
    query_embedding = embedding_client.embed_query(prompt)

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    document_rows = result.get("documents") or []
    metadata_rows = result.get("metadatas") or []
    distance_rows = result.get("distances") or []

    documents: list[str] = document_rows[0] if document_rows else []
    metadatas: list[dict[str, object] | None] = (
        metadata_rows[0] if metadata_rows else []
    )
    distances: list[float] = distance_rows[0] if distance_rows else []

    return documents, metadatas, distances


def make_retrieval_node(runtime: AgentRuntime):
    def retrieval_node(state: AgentState) -> AgentState:
        prompt = state["prompt"].strip()
        if not prompt:
            return {
                "prompt": "",
                "response": "Please provide a prompt.",
                "route": state.get("route", "reason"),
                "context": "",
                "retrieval_distances": [],
            }

        try:
            documents, metadatas, distances = _retrieve_raw(
                prompt, runtime.embedding_model
            )
        except Exception:
            logger.exception(
                "Context retrieval failed. Continuing with an empty context."
            )
            return {
                "prompt": prompt,
                "response": state.get("response", ""),
                "route": state.get("route", "reason"),
                "context": "No relevant context was retrieved from the knowledge base.",
                "retrieval_distances": [],
            }

        context = (
            _format_retrieved_context(documents, metadatas)
            if documents
            else "No relevant context was retrieved from the knowledge base."
        )

        return {
            "prompt": prompt,
            "response": state.get("response", ""),
            "route": state.get("route", "reason"),
            "context": context,
            "retrieval_distances": distances,
        }

    return retrieval_node


def _compute_diagnostics(distances: list[float]) -> RetrievalDiagnostics:
    """Pure, deterministic analysis of retrieval quality from raw L2 distances."""
    if not distances:
        return RetrievalDiagnostics(
            chunk_count=0,
            similarity_scores=[],
            min_score=0.0,
            max_score=0.0,
            mean_score=0.0,
            median_score=0.0,
            above_threshold_count=0,
            coverage_verdict="insufficient",
        )

    # Convert L2 distance [0, 2] → similarity [0, 1]  (valid for normalised embeddings)
    scores = [max(0.0, min(1.0, 1.0 - d / 2.0)) for d in distances]

    min_score = min(scores)
    max_score = max(scores)
    mean_score = statistics.mean(scores)
    median_score = statistics.median(scores)
    above_threshold_count = sum(1 for s in scores if s >= SIMILARITY_THRESHOLD)

    if mean_score >= 0.60 or above_threshold_count >= 2:
        verdict: Literal["sufficient", "partial", "insufficient"] = "sufficient"
    elif mean_score >= 0.40 or above_threshold_count >= 1:
        verdict = "partial"
    else:
        verdict = "insufficient"

    return RetrievalDiagnostics(
        chunk_count=len(distances),
        similarity_scores=scores,
        min_score=round(min_score, 4),
        max_score=round(max_score, 4),
        mean_score=round(mean_score, 4),
        median_score=round(median_score, 4),
        above_threshold_count=above_threshold_count,
        coverage_verdict=verdict,
    )


def make_analysis_node():
    """Deterministic node — no LLM. Computes retrieval quality diagnostics."""

    def analysis_node(state: AgentState) -> AgentState:
        distances: list[float] = state.get("retrieval_distances") or []
        diagnostics = _compute_diagnostics(distances)
        return {
            **state,  # type: ignore[misc]
            "retrieval_diagnostics": diagnostics,
        }

    return analysis_node


def make_reason_node(runtime: AgentRuntime):
    def reason_node(state: AgentState) -> AgentState:
        prompt = state["prompt"].strip()
        if not prompt:
            return {
                "prompt": "",
                "response": "Please provide a prompt.",
                "route": state.get("route", "reason"),
                "context": state.get("context", ""),
            }

        llm = runtime.llm_factory(runtime.model_name)
        context = state.get(
            "context", "No relevant context was retrieved from the knowledge base."
        )

        diagnostics: RetrievalDiagnostics | None = state.get("retrieval_diagnostics")
        diagnostics_summary = (
            f"Retrieval diagnostics:\n"
            f"  chunks retrieved: {diagnostics.chunk_count}\n"
            f"  similarity scores (min/mean/max): "
            f"{diagnostics.min_score:.3f} / {diagnostics.mean_score:.3f} / {diagnostics.max_score:.3f}\n"
            f"  chunks above relevance threshold ({SIMILARITY_THRESHOLD}): "
            f"{diagnostics.above_threshold_count}/{diagnostics.chunk_count}\n"
            f"  coverage verdict: {diagnostics.coverage_verdict}"
            if diagnostics
            else "Retrieval diagnostics: unavailable"
        )

        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful research assistant. You are given retrieved context "
                    "and diagnostics that describe how well the retrieved chunks match the "
                    "user question. Use the diagnostics to judge whether the retrieved "
                    "context provides enough information to answer confidently. "
                    "If coverage is 'insufficient' or 'partial', say so explicitly and "
                    "supplement with your own knowledge where appropriate.",
                ),
                (
                    "human",
                    "Question:\n{question}\n\n"
                    "{diagnostics_summary}\n\n"
                    "Retrieved context:\n{context}\n\nAnswer:",
                ),
            ]
        )

        chain = (
            RunnablePassthrough.assign(
                question=lambda payload: payload["question"],
                context=lambda payload: payload["context"],
                diagnostics_summary=lambda payload: payload["diagnostics_summary"],
            )
            | rag_prompt
            | llm
        )

        try:
            model_response = chain.invoke(
                {
                    "question": prompt,
                    "context": context,
                    "diagnostics_summary": diagnostics_summary,
                }
            )
        except Exception:
            logger.exception("Reasoning model invocation failed.")
            return {
                "prompt": prompt,
                "response": "Model invocation failed. Please try again.",
                "route": state.get("route", "reason"),
                "context": context,
            }

        response_text = str(model_response.content)

        return {
            "prompt": prompt,
            "response": response_text,
            "route": state.get("route", "reason"),
            "context": context,
        }

    return reason_node


def make_router_node(runtime: AgentRuntime):
    def router_node(state: AgentState) -> AgentState:
        prompt = state["prompt"].strip()
        if not prompt:
            return {
                "prompt": "",
                "response": "Please provide a prompt.",
                "route": "direct",
            }

        llm = runtime.llm_factory(runtime.model_name)
        router_llm = llm.with_structured_output(RouteDecision)

        routing_prompt = (
            "You are a query complexity router. Classify the user prompt as either "
            "'direct' or 'reason'. Return 'direct' for direct Q&A requests that can "
            "be answered in one straightforward response. Return 'reason' if the "
            "request requires multi-step reasoning, planning, tool use, or iterative "
            "processing.\n\n"
            f"User prompt:\n{prompt}"
        )

        try:
            decision = router_llm.invoke(routing_prompt)
            route = decision.route
        except Exception:
            logger.exception("Routing model invocation failed. Falling back to reason.")
            route = "reason"

        return {
            "prompt": prompt,
            "response": "",
            "route": route,
        }

    return router_node


def make_direct_node(runtime: AgentRuntime):
    def direct_node(state: AgentState) -> AgentState:
        prompt = state["prompt"].strip()
        if not prompt:
            return {
                "prompt": "",
                "response": "Please provide a prompt.",
                "route": "direct",
            }

        llm = runtime.llm_factory(runtime.model_name)

        try:
            model_response = llm.invoke(prompt)
        except Exception:
            logger.exception("Direct model invocation failed.")
            return {
                "prompt": prompt,
                "response": "Model invocation failed. Please try again.",
                "route": "direct",
            }

        response_text = str(model_response.content)

        return {
            "prompt": prompt,
            "response": response_text,
            "route": "direct",
        }

    return direct_node


def _select_route(state: AgentState) -> Literal["direct", "reason"]:
    return state.get("route", "reason")


def build_graph(runtime: AgentRuntime):
    graph = StateGraph(AgentState)
    graph.add_node("router", make_router_node(runtime))
    graph.add_node("direct", make_direct_node(runtime))
    graph.add_node("retrieval", make_retrieval_node(runtime))
    graph.add_node("analysis", make_analysis_node())
    graph.add_node("reason", make_reason_node(runtime))

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _select_route,
        {
            "direct": "direct",
            "reason": "retrieval",
        },
    )

    graph.add_edge("direct", END)
    graph.add_edge("retrieval", "analysis")
    graph.add_edge("analysis", "reason")
    graph.add_edge("reason", END)

    return graph.compile()


agent_graph = build_graph(build_default_runtime())


def invoke_agent(prompt: str) -> str:
    result: AgentState = agent_graph.invoke({"prompt": prompt, "response": ""})
    return result["response"]
