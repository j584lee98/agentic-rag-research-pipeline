import logging

from langchain_core.prompts import ChatPromptTemplate

from agents.runtime import AgentRuntime
from agents.state import (
    AgentState,
    AgentStateUpdate,
    EMPTY_PROMPT_RESPONSE,
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


def make_reason_node(runtime: AgentRuntime):
    def reason_node(state: AgentState) -> AgentStateUpdate:
        prompt = state["prompt"].strip()
        if not prompt:
            return {"prompt": "", "response": EMPTY_PROMPT_RESPONSE}

        try:
            response = (
                PROMPT_TEMPLATE | runtime.llm_factory(runtime.model_name)
            ).invoke(
                {
                    "question": prompt,
                    "context": state.get("context", NO_RETRIEVED_CONTEXT),
                    "diagnostics_summary": format_diagnostics(
                        state.get("retrieval_diagnostics")
                    ),
                }
            )
        except Exception:
            logger.exception("Reasoning model invocation failed.")
            return {"response": MODEL_FAILURE_RESPONSE}

        return {"response": str(response.content)}

    return reason_node
