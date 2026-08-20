from agents.runtime import AgentRuntime
from agents.state import AgentState, AgentStateUpdate


def _deduplication_key(document_chunk: str) -> str:
    return " ".join(document_chunk.split()).casefold()


def make_merge_deduplicate_node(runtime: AgentRuntime):
    _ = runtime

    def merge_deduplicate_node(state: AgentState) -> AgentStateUpdate:
        final_document_chunks: list[str] = []
        seen_chunks: set[str] = set()

        for retrieval in state.get("query_retrievals", []):
            for document_chunk in retrieval["document_chunks"]:
                deduplication_key = _deduplication_key(document_chunk)
                if deduplication_key and deduplication_key not in seen_chunks:
                    seen_chunks.add(deduplication_key)
                    final_document_chunks.append(document_chunk)

        return {"final_document_chunks": final_document_chunks}

    return merge_deduplicate_node
