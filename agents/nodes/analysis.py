import logging

from pydantic import BaseModel

from agents.analysis import compute_retrieval_diagnostics
from agents.runtime import AgentRuntime
from agents.state import (
    AgentState,
    AgentStateUpdate,
    AnalysisVerdict,
    NO_RETRIEVED_CONTEXT,
    RetrievalDiagnostics,
)


logger = logging.getLogger(__name__)


class AnalysisDecision(BaseModel):
    verdict: AnalysisVerdict


def format_diagnostics(diagnostics: RetrievalDiagnostics) -> str:
    return (
        f"chunks retrieved: {diagnostics.chunk_count}\n"
        "similarity scores (min/mean/max): "
        f"{diagnostics.min_score:.3f} / {diagnostics.mean_score:.3f} / "
        f"{diagnostics.max_score:.3f}\n"
        f"median similarity score: {diagnostics.median_score:.3f}\n"
        f"chunks above relevance threshold: "
        f"{diagnostics.above_threshold_count}/{diagnostics.chunk_count}\n"
        f"deterministic coverage verdict: {diagnostics.coverage_verdict}"
    )


def make_analysis_node(runtime: AgentRuntime):
    def analysis_node(state: AgentState) -> AgentStateUpdate:
        diagnostics = compute_retrieval_diagnostics(
            state.get("retrieval_distances", [])
        )
        assessment_llm = runtime.llm_factory(runtime.model_name).with_structured_output(
            AnalysisDecision
        )
        assessment_prompt = (
            "Assess whether the retrieved context is sufficient to answer the user "
            "prompt accurately. Return 'pass' only when the context is relevant and "
            "adequately covers the request; otherwise return 'fail'.\n\n"
            f"User prompt:\n{state.get('prompt', '')}\n\n"
            f"Retrieved context:\n{state.get('context', NO_RETRIEVED_CONTEXT)}\n\n"
            f"Retrieval diagnostics:\n{format_diagnostics(diagnostics)}"
        )

        try:
            verdict = assessment_llm.invoke(assessment_prompt).verdict
        except Exception:
            logger.exception("Retrieval assessment failed. Treating context as failed.")
            verdict = "fail"

        return {
            "retrieval_diagnostics": diagnostics,
            "analysis_verdict": verdict,
        }

    return analysis_node


def select_analysis_verdict(state: AgentState) -> AnalysisVerdict:
    return state.get("analysis_verdict", "fail")
