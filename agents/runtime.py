from collections.abc import Callable
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from app.config import get_settings


@dataclass(frozen=True)
class AgentRuntime:
    model_name: str
    embedding_model: str
    llm_factory: Callable[[str], ChatOpenAI]


def build_default_runtime() -> AgentRuntime:
    settings = get_settings()
    return AgentRuntime(
        model_name=settings.openai_model,
        embedding_model=settings.openai_embedding_model,
        llm_factory=ChatOpenAI,
    )
