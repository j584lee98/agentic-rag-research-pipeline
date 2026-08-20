import logging

from pydantic import BaseModel, Field

from agents.retrieval import retrieve
from agents.runtime import AgentRuntime
from agents.state import (
    AgentState,
    AgentStateUpdate,
    QueryRetrieval,
)

logger = logging.getLogger(__name__)


class QueryExpansion(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=3)


def make_expand_query_node(runtime: AgentRuntime):
    def expand_query_node(state: AgentState) -> AgentStateUpdate:
        prompt = state["prompt"].strip()
        existing_retrievals = list(state.get("query_retrievals", []))
        if not prompt:
            return {"query_retrievals": existing_retrievals}

        expansion_llm = runtime.llm_factory(runtime.model_name).with_structured_output(
            QueryExpansion
        )
        expansion_prompt = (
            "Generate three distinct search queries that could retrieve complementary "
            "context for the user's question. Preserve the user's intent, vary wording "
            "and perspective, and return only the queries.\n\n"
            f"User question:\n{prompt}"
        )

        try:
            generated_queries = expansion_llm.invoke(expansion_prompt).queries
        except Exception:
            logger.exception("Query expansion model invocation failed.")
            return {"query_retrievals": existing_retrievals}

        generated_retrievals: list[QueryRetrieval] = []
        for query in dict.fromkeys(
            query.strip() for query in generated_queries if query.strip()
        ):
            try:
                documents, metadatas, distances = retrieve(
                    query, runtime.embedding_model
                )
            except Exception:
                logger.exception("Expanded-query retrieval failed.")
                documents, metadatas, distances = [], [], []

            generated_retrievals.append(
                {
                    "query": query,
                    "query_type": "generated",
                    "document_chunks": documents,
                    "metadatas": metadatas,
                    "distances": distances,
                }
            )

        return {
            "query_retrievals": existing_retrievals + generated_retrievals,
        }

    return expand_query_node
