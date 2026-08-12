from langgraph.graph import END, START, StateGraph

from agents.nodes.analysis import make_analysis_node
from agents.nodes.reasoning import make_reason_node
from agents.nodes.retrieval import make_retrieval_node
from agents.nodes.routing import make_direct_node, make_router_node, select_route
from agents.runtime import AgentRuntime, build_default_runtime
from agents.state import AgentState


def build_graph(runtime: AgentRuntime):
    """Build the routed RAG workflow.

    Direct queries end after a single model response. Research queries always
    progress through retrieval, deterministic analysis, and reasoning.
    """
    graph = StateGraph(AgentState)
    graph.add_node("router", make_router_node(runtime))
    graph.add_node("direct", make_direct_node(runtime))
    graph.add_node("retrieval", make_retrieval_node(runtime))
    graph.add_node("analysis", make_analysis_node())
    graph.add_node("reason", make_reason_node(runtime))

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        select_route,
        {"direct": "direct", "reason": "retrieval"},
    )
    graph.add_edge("direct", END)
    graph.add_edge("retrieval", "analysis")
    graph.add_edge("analysis", "reason")
    graph.add_edge("reason", END)

    return graph.compile()


agent_graph = build_graph(build_default_runtime())


def invoke_agent(prompt: str) -> str:
    result = agent_graph.invoke({"prompt": prompt, "response": ""})
    return result["response"]
