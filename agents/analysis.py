import statistics

from agents.state import (
    CoverageVerdict,
    RetrievalDiagnostics,
    SIMILARITY_THRESHOLD,
)


def compute_retrieval_diagnostics(distances: list[float]) -> RetrievalDiagnostics:
    """Compute deterministic retrieval-quality statistics from L2 distances."""
    if not distances:
        return RetrievalDiagnostics(chunk_count=0)

    similarity_scores = [
        max(0.0, min(1.0, 1.0 - distance / 2.0)) for distance in distances
    ]
    mean_score = statistics.mean(similarity_scores)
    above_threshold_count = sum(
        score >= SIMILARITY_THRESHOLD for score in similarity_scores
    )

    if mean_score >= 0.60 or above_threshold_count >= 2:
        coverage_verdict: CoverageVerdict = "sufficient"
    elif mean_score >= 0.40 or above_threshold_count >= 1:
        coverage_verdict = "partial"
    else:
        coverage_verdict = "insufficient"

    return RetrievalDiagnostics(
        chunk_count=len(distances),
        similarity_scores=similarity_scores,
        min_score=round(min(similarity_scores), 4),
        max_score=round(max(similarity_scores), 4),
        mean_score=round(mean_score, 4),
        median_score=round(statistics.median(similarity_scores), 4),
        above_threshold_count=above_threshold_count,
        coverage_verdict=coverage_verdict,
    )
