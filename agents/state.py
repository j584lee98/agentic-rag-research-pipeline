from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict


Route = Literal["direct", "reason"]
AnalysisVerdict = Literal["pass", "fail"]
CoverageVerdict = Literal["sufficient", "partial", "insufficient"]
QueryType = Literal["original", "generated"]

NO_RETRIEVED_CONTEXT = "No relevant context was retrieved from the knowledge base."
EMPTY_PROMPT_RESPONSE = "Please provide a prompt."
MODEL_FAILURE_RESPONSE = "Model invocation failed. Please try again."
SIMILARITY_THRESHOLD = 0.50
MAX_WEB_SEARCH_ITERATIONS = 3


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


class QueryRetrieval(TypedDict):
    """Retrieved chunks and supporting data for one search query."""

    query: str
    query_type: QueryType
    document_chunks: list[str]
    metadatas: list[dict[str, object] | None]
    distances: list[float]


class AgentState(TypedDict):
    prompt: str
    response: str
    route: NotRequired[Route]
    context: NotRequired[str]
    retrieval_distances: NotRequired[list[float]]
    query_retrievals: NotRequired[list[QueryRetrieval]]
    final_document_chunks: NotRequired[list[str]]
    retrieval_diagnostics: NotRequired[RetrievalDiagnostics]
    analysis_verdict: NotRequired[AnalysisVerdict]
    web_search_query: NotRequired[str]
    web_search_results: NotRequired[list[str]]
    web_search_iterations: NotRequired[int]


class AgentStateUpdate(TypedDict, total=False):
    prompt: str
    response: str
    route: Route
    context: str
    retrieval_distances: list[float]
    query_retrievals: list[QueryRetrieval]
    final_document_chunks: list[str]
    retrieval_diagnostics: RetrievalDiagnostics
    analysis_verdict: AnalysisVerdict
    web_search_query: str
    web_search_results: list[str]
    web_search_iterations: int
