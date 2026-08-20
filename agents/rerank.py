import json
from urllib.request import Request, urlopen


def rerank(
    query: str,
    documents: list[str],
    *,
    model_name: str,
    base_url: str,
) -> list[int]:
    """Return document indices ordered by relevance from vLLM's rerank API."""
    request = Request(
        f"{base_url.rstrip('/')}/rerank",
        data=json.dumps(
            {
                "model": model_name,
                "query": query,
                "documents": documents,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        payload = json.load(response)

    results = payload.get("data", [])
    ranked_results = sorted(
        results,
        key=lambda result: float(result.get("relevance_score", 0.0)),
        reverse=True,
    )
    ranked_indices: list[int] = []
    for result in ranked_results:
        index = result.get("index")
        if (
            isinstance(index, int)
            and 0 <= index < len(documents)
            and index not in ranked_indices
        ):
            ranked_indices.append(index)

    return ranked_indices
