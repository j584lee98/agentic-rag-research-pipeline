import logging

from agents.retrieval import format_context, retrieve
from agents.runtime import AgentRuntime
from agents.state import (
    AgentState,
    AgentStateUpdate,
    EMPTY_PROMPT_RESPONSE,
    NO_RETRIEVED_CONTEXT,
)

logger = logging.getLogger(__name__)


def make_retrieval_node(runtime: AgentRuntime):
    def retrieval_node(state: AgentState) -> AgentStateUpdate:
        prompt = state["prompt"].strip()
        if not prompt:
            return {
                "prompt": "",
                "response": EMPTY_PROMPT_RESPONSE,
                "context": "",
                "retrieval_distances": [],
                "query_retrievals": [],
                "final_document_chunks": [],
            }

        try:
            documents, metadatas, distances = retrieve(prompt, runtime.embedding_model)
        except Exception:
            logger.exception("Context retrieval failed. Continuing without context.")
            return {
                "context": NO_RETRIEVED_CONTEXT,
                "retrieval_distances": [],
                "final_document_chunks": [],
                "query_retrievals": [
                    {
                        "query": prompt,
                        "query_type": "original",
                        "document_chunks": [],
                        "metadatas": [],
                        "distances": [],
                    }
                ],
            }

        return {
            "context": format_context(documents, metadatas)
            if documents
            else NO_RETRIEVED_CONTEXT,
            "retrieval_distances": distances,
            "final_document_chunks": documents,
            "query_retrievals": [
                {
                    "query": prompt,
                    "query_type": "original",
                    "document_chunks": documents,
                    "metadatas": metadatas,
                    "distances": distances,
                }
            ],
        }

    return retrieval_node
