import logging
from dataclasses import dataclass
from typing import Callable, Literal, NotRequired, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    prompt: str
    response: str
    route: NotRequired[Literal["simple", "complex"]]


class RouteDecision(BaseModel):
    route: Literal["simple", "complex"]


@dataclass(frozen=True)
class AgentRuntime:
    model_name: str
    llm_factory: Callable[[str], ChatOpenAI]


def _default_llm_factory(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(model=model_name)


def build_default_runtime() -> AgentRuntime:
    settings = get_settings()
    return AgentRuntime(
        model_name=settings.openai_model,
        llm_factory=_default_llm_factory,
    )


def make_agent_node(runtime: AgentRuntime):
    def agent_node(state: AgentState) -> AgentState:
        prompt = state["prompt"].strip()
        if not prompt:
            return {
                "prompt": "",
                "response": "Please provide a prompt.",
                "route": state.get("route", "complex"),
            }

        llm = runtime.llm_factory(runtime.model_name)

        try:
            model_response = llm.invoke(prompt)
        except Exception:
            logger.exception("Model invocation failed.")
            return {
                "prompt": prompt,
                "response": "Model invocation failed. Please try again.",
                "route": state.get("route", "complex"),
            }

        response_text = str(model_response.content)

        return {
            "prompt": prompt,
            "response": response_text,
            "route": state.get("route", "complex"),
        }

    return agent_node


def make_router_node(runtime: AgentRuntime):
    def router_node(state: AgentState) -> AgentState:
        prompt = state["prompt"].strip()
        if not prompt:
            return {
                "prompt": "",
                "response": "Please provide a prompt.",
                "route": "simple",
            }

        llm = runtime.llm_factory(runtime.model_name)
        router_llm = llm.with_structured_output(RouteDecision)

        routing_prompt = (
            "You are a query complexity router. Classify the user prompt as either "
            "'simple' or 'complex'. Return 'simple' for direct Q&A requests that can "
            "be answered in one straightforward response. Return 'complex' if the "
            "request requires multi-step reasoning, planning, tool use, or iterative "
            "processing.\n\n"
            f"User prompt:\n{prompt}"
        )

        try:
            decision = router_llm.invoke(routing_prompt)
            route = decision.route
        except Exception:
            logger.exception(
                "Routing model invocation failed. Falling back to complex."
            )
            route = "complex"

        return {
            "prompt": prompt,
            "response": "",
            "route": route,
        }

    return router_node


def make_simple_llm_node(runtime: AgentRuntime):
    def simple_llm_node(state: AgentState) -> AgentState:
        prompt = state["prompt"].strip()
        if not prompt:
            return {
                "prompt": "",
                "response": "Please provide a prompt.",
                "route": "simple",
            }

        llm = runtime.llm_factory(runtime.model_name)

        try:
            model_response = llm.invoke(prompt)
        except Exception:
            logger.exception("Simple model invocation failed.")
            return {
                "prompt": prompt,
                "response": "Model invocation failed. Please try again.",
                "route": "simple",
            }

        response_text = str(model_response.content)

        return {
            "prompt": prompt,
            "response": response_text,
            "route": "simple",
        }

    return simple_llm_node


def _select_route(state: AgentState) -> Literal["simple", "complex"]:
    return state.get("route", "complex")


def build_graph(runtime: AgentRuntime):
    graph = StateGraph(AgentState)
    graph.add_node("router", make_router_node(runtime))
    graph.add_node("simple_llm", make_simple_llm_node(runtime))
    graph.add_node("agent", make_agent_node(runtime))

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _select_route,
        {
            "simple": "simple_llm",
            "complex": "agent",
        },
    )

    graph.add_edge("simple_llm", END)
    graph.add_edge("agent", END)

    return graph.compile()


agent_graph = build_graph(build_default_runtime())


def invoke_agent(prompt: str) -> str:
    result: AgentState = agent_graph.invoke({"prompt": prompt, "response": ""})
    return result["response"]
