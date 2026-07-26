import logging
from dataclasses import dataclass
from typing import Callable, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.config import get_settings

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    prompt: str
    response: str


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
            return {"prompt": "", "response": "Please provide a prompt."}

        llm = runtime.llm_factory(runtime.model_name)

        try:
            model_response = llm.invoke(prompt)
        except Exception:
            logger.exception("Model invocation failed.")
            return {
                "prompt": prompt,
                "response": "Model invocation failed. Please try again.",
            }

        response_text = str(model_response.content)

        return {
            "prompt": prompt,
            "response": response_text,
        }

    return agent_node


def build_graph(runtime: AgentRuntime):
    graph = StateGraph(AgentState)
    graph.add_node("agent", make_agent_node(runtime))
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()


agent_graph = build_graph(build_default_runtime())


def invoke_agent(prompt: str) -> str:
    result: AgentState = agent_graph.invoke({"prompt": prompt, "response": ""})
    return result["response"]
