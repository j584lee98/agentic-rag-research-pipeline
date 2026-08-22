import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from agents.runtime import AgentRuntime
from agents.state import (
    AgentState,
    AgentStateUpdate,
    EMPTY_PROMPT_RESPONSE,
    MAX_WEB_SEARCH_ITERATIONS,
    MODEL_FAILURE_RESPONSE,
    NO_RETRIEVED_CONTEXT,
    RetrievalDiagnostics,
    SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful research assistant. You are given retrieved context "
            "and diagnostics that describe how well the retrieved chunks match the "
            "user question. Use the diagnostics to judge whether the retrieved "
            "context provides enough information to answer confidently. "
            "If coverage is 'insufficient' or 'partial', say so explicitly and "
            "supplement with your own knowledge where appropriate. You may call "
            "web_search when current or missing information requires verification. "
            "When web-search results are provided, use them to produce a final "
            "answer with source URLs.",
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "{diagnostics_summary}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Web-search results:\n{web_search_results}\n\nAnswer:",
        ),
    ]
)


def format_web_search_results(results: list[str]) -> str:
    return "\n\n".join(results) if results else "No web-search results yet."


def format_diagnostics(diagnostics: RetrievalDiagnostics | None) -> str:
    if diagnostics is None:
        return "Retrieval diagnostics: unavailable"

    return (
        "Retrieval diagnostics:\n"
        f"  chunks retrieved: {diagnostics.chunk_count}\n"
        "  similarity scores (min/mean/max): "
        f"{diagnostics.min_score:.3f} / {diagnostics.mean_score:.3f} / "
        f"{diagnostics.max_score:.3f}\n"
        f"  chunks above relevance threshold ({SIMILARITY_THRESHOLD}): "
        f"{diagnostics.above_threshold_count}/{diagnostics.chunk_count}\n"
        f"  coverage verdict: {diagnostics.coverage_verdict}"
    )


def format_document_chunks(document_chunks: list[str]) -> str:
    if not document_chunks:
        return NO_RETRIEVED_CONTEXT

    return "\n\n".join(
        f"[Context {index}]\n{document_chunk.strip()}"
        for index, document_chunk in enumerate(document_chunks, start=1)
    )


def make_reason_node(runtime: AgentRuntime):
    @tool
    def web_search(query: str) -> str:
        """Search the public web for current or missing information."""
        return "Search request submitted."

    def reason_node(state: AgentState) -> AgentStateUpdate:
        prompt = state["prompt"].strip()
        if not prompt:
            return {"prompt": "", "response": EMPTY_PROMPT_RESPONSE}

        try:
            llm = runtime.llm_factory(runtime.model_name)
            if state.get("web_search_iterations", 0) < MAX_WEB_SEARCH_ITERATIONS:
                llm = llm.bind_tools([web_search])
            response = (PROMPT_TEMPLATE | llm).invoke(
                {
                    "question": prompt,
                    "context": format_document_chunks(
                        state.get("final_document_chunks", [])
                    ),
                    "diagnostics_summary": format_diagnostics(
                        state.get("retrieval_diagnostics")
                    ),
                    "web_search_results": format_web_search_results(
                        state.get("web_search_results", [])
                    ),
                }
            )
        except Exception:
            logger.exception("Reasoning model invocation failed.")
            return {"response": MODEL_FAILURE_RESPONSE}

        tool_calls = getattr(response, "tool_calls", [])
        for tool_call in tool_calls:
            if tool_call.get("name") == "web_search":
                query = tool_call.get("args", {}).get("query", "").strip()
                if query:
                    return {"response": "", "web_search_query": query}

        return {"response": str(response.content), "web_search_query": ""}

    return reason_node
