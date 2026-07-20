import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()


class AgentState(TypedDict):
    prompt: str
    response: str


def agent_node(state: AgentState) -> AgentState:
    prompt = state["prompt"].strip()
    if not prompt:
        return {"prompt": "", "response": "Please provide a prompt."}

    model_name = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    llm = ChatOpenAI(model=model_name)

    try:
        model_response = llm.invoke(prompt)
    except Exception as exc:
        return {
            "prompt": prompt,
            "response": f"Model invocation failed: {exc}",
        }

    response_text = model_response.content

    return {
        "prompt": prompt,
        "response": response_text,
    }


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()


agent_graph = build_graph()


def invoke_agent(prompt: str) -> str:
    result: AgentState = agent_graph.invoke({"prompt": prompt, "response": ""})
    return result["response"]
