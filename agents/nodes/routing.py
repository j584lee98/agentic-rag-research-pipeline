import logging

from pydantic import BaseModel

from agents.runtime import AgentRuntime
from agents.state import (
    AgentState,
    AgentStateUpdate,
    EMPTY_PROMPT_RESPONSE,
    MODEL_FAILURE_RESPONSE,
    Route,
)

logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    route: Route


def make_router_node(runtime: AgentRuntime):
    def router_node(state: AgentState) -> AgentStateUpdate:
        prompt = state["prompt"].strip()
        if not prompt:
            return {"prompt": "", "response": EMPTY_PROMPT_RESPONSE, "route": "direct"}

        router_llm = runtime.llm_factory(runtime.model_name).with_structured_output(
            RouteDecision
        )
        routing_prompt = (
            "You are a query complexity router. Classify the user prompt as either "
            "'direct' or 'reason'. Return 'direct' for direct Q&A requests that can "
            "be answered in one straightforward response. Return 'reason' if the "
            "request requires multi-step reasoning, planning, tool use, or iterative "
            "processing.\n\n"
            f"User prompt:\n{prompt}"
        )

        try:
            route = router_llm.invoke(routing_prompt).route
        except Exception:
            logger.exception("Routing model invocation failed. Falling back to reason.")
            route = "reason"

        return {"prompt": prompt, "response": "", "route": route}

    return router_node


def make_direct_node(runtime: AgentRuntime):
    def direct_node(state: AgentState) -> AgentStateUpdate:
        prompt = state["prompt"].strip()
        if not prompt:
            return {"prompt": "", "response": EMPTY_PROMPT_RESPONSE, "route": "direct"}

        try:
            response = runtime.llm_factory(runtime.model_name).invoke(prompt)
        except Exception:
            logger.exception("Direct model invocation failed.")
            return {"response": MODEL_FAILURE_RESPONSE, "route": "direct"}

        return {"response": str(response.content), "route": "direct"}

    return direct_node


def select_route(state: AgentState) -> Route:
    return state.get("route", "reason")
