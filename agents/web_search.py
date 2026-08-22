import json
import logging
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def search_web(query: str) -> str:
    """Search the web with Tavily and return concise, citable text results."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Web search is unavailable because TAVILY_API_KEY is not configured."

    request = Request(
        TAVILY_SEARCH_URL,
        data=json.dumps(
            {
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=15) as response:  # noqa: S310
            results = json.load(response).get("results", [])
    except URLError, TimeoutError, ValueError:
        logger.exception("Web search failed.")
        return "Web search failed; continue using the available context."

    if not results:
        return "Web search returned no results."

    return "\n\n".join(
        f"[{result.get('title', 'Untitled')}]({result.get('url', '')})\n"
        f"{result.get('content', '')}"
        for result in results
    )
