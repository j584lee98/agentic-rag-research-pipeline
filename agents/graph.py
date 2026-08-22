from langgraph.graph import END, START, StateGraph

from agents.nodes.analysis import make_analysis_node, select_analysis_verdict
from agents.nodes.expand_query import make_expand_query_node
from agents.nodes.input import make_normalize_input_node
from agents.nodes.merge_deduplicate import make_merge_deduplicate_node
from agents.nodes.reasoning import make_reason_node
from agents.nodes.rerank import make_rerank_node
from agents.nodes.retrieval import make_retrieval_node
from agents.nodes.routing import make_direct_node, make_router_node, select_route
from agents.nodes.web_search import make_web_search_node
from agents.runtime import AgentRuntime, build_default_runtime
from agents.state import AgentState, MAX_WEB_SEARCH_ITERATIONS


def select_reasoning_next_step(state: AgentState) -> str:
    if (
        state.get("web_search_query")
        and state.get("web_search_iterations", 0) < MAX_WEB_SEARCH_ITERATIONS
    ):
        return "web_search"
    return "end"


def build_graph(runtime: AgentRuntime):
    """Build the routed RAG workflow.

    Direct queries end after a single model response. Research queries always
    progress through retrieval, deterministic analysis, and reasoning.
    """
    graph = StateGraph(AgentState)
    graph.add_node("normalize_input", make_normalize_input_node())
    graph.add_node("router", make_router_node(runtime))
    graph.add_node("direct", make_direct_node(runtime))
    graph.add_node("retrieval", make_retrieval_node(runtime))
    graph.add_node("analysis", make_analysis_node(runtime))
    graph.add_node("expand_query", make_expand_query_node(runtime))
    graph.add_node("merge_deduplicate", make_merge_deduplicate_node(runtime))
    graph.add_node("rerank", make_rerank_node(runtime))
    graph.add_node("reason", make_reason_node(runtime))
    graph.add_node("web_search", make_web_search_node(runtime))

    graph.add_edge(START, "normalize_input")
    graph.add_edge("normalize_input", "router")
    graph.add_conditional_edges(
        "router",
        select_route,
        {"direct": "direct", "reason": "retrieval"},
    )
    graph.add_edge("direct", END)
    graph.add_edge("retrieval", "analysis")
    graph.add_conditional_edges(
        "analysis",
        select_analysis_verdict,
        {"pass": "rerank", "fail": "expand_query"},
    )
    graph.add_edge("expand_query", "merge_deduplicate")
    graph.add_edge("merge_deduplicate", "rerank")
    graph.add_edge("rerank", "reason")
    graph.add_conditional_edges(
        "reason",
        select_reasoning_next_step,
        {"web_search": "web_search", "end": END},
    )
    graph.add_edge("web_search", "reason")

    return graph.compile()


agent_graph = build_graph(build_default_runtime())


def invoke_agent(prompt: str) -> str:
    result = agent_graph.invoke({"prompt": prompt, "response": ""})
    return result["response"]
