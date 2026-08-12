from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict


Route = Literal["direct", "reason"]
AnalysisVerdict = Literal["pass", "fail"]
CoverageVerdict = Literal["sufficient", "partial", "insufficient"]

NO_RETRIEVED_CONTEXT = "No relevant context was retrieved from the knowledge base."
EMPTY_PROMPT_RESPONSE = "Please provide a prompt."
MODEL_FAILURE_RESPONSE = "Model invocation failed. Please try again."
SIMILARITY_THRESHOLD = 0.50


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Deterministic statistics computed from raw retrieval distances."""

    chunk_count: int
    similarity_scores: list[float] = field(default_factory=list)
    min_score: float = 0.0
    max_score: float = 0.0
    mean_score: float = 0.0
    median_score: float = 0.0
    above_threshold_count: int = 0
    coverage_verdict: CoverageVerdict = "insufficient"


class AgentState(TypedDict):
    prompt: str
    response: str
    route: NotRequired[Route]
    context: NotRequired[str]
    retrieval_distances: NotRequired[list[float]]
    retrieval_diagnostics: NotRequired[RetrievalDiagnostics]
    analysis_verdict: NotRequired[AnalysisVerdict]


class AgentStateUpdate(TypedDict, total=False):
    prompt: str
    response: str
    route: Route
    context: str
    retrieval_distances: list[float]
    retrieval_diagnostics: RetrievalDiagnostics
    analysis_verdict: AnalysisVerdict
