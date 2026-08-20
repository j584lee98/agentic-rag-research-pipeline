import logging

from agents.rerank import rerank
from agents.runtime import AgentRuntime
from agents.state import AgentState, AgentStateUpdate

logger = logging.getLogger(__name__)

RERANK_TOP_K = 5


def make_rerank_node(runtime: AgentRuntime):
    def rerank_node(state: AgentState) -> AgentStateUpdate:
        document_chunks = state.get("final_document_chunks", [])
        if not document_chunks:
            return {"final_document_chunks": []}

        try:
            ranked_indices = rerank(
                state.get("prompt", ""),
                document_chunks,
                model_name=runtime.rerank_model,
                base_url=runtime.vllm_base_url,
            )
            selected_chunks = [
                document_chunks[index] for index in ranked_indices[:RERANK_TOP_K]
            ]
        except Exception:
            logger.exception("vLLM reranking failed. Using retrieval order.")
            selected_chunks = document_chunks[:RERANK_TOP_K]

        return {"final_document_chunks": selected_chunks}

    return rerank_node
