import logging

from agents.runtime import AgentRuntime
from agents.state import AgentState, AgentStateUpdate

logger = logging.getLogger(__name__)


def make_web_search_node(runtime: AgentRuntime):
    def web_search_node(state: AgentState) -> AgentStateUpdate:
        query = state.get("web_search_query", "").strip()
        results = list(state.get("web_search_results", []))
        iterations = state.get("web_search_iterations", 0)
        if not query:
            return {"web_search_results": results}

        try:
            result = runtime.web_search(query)
        except Exception:
            logger.exception("Web search tool invocation failed.")
            result = "Web search failed; continue using the available context."

        return {
            "web_search_query": "",
            "web_search_results": results + [f"Search query: {query}\n{result}"],
            "web_search_iterations": iterations + 1,
        }

    return web_search_node
