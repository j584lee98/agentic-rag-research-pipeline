import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agents.analysis import compute_retrieval_diagnostics
from agents.graph import agent_graph
from agents.nodes.analysis import make_analysis_node
from agents.nodes.expand_query import make_expand_query_node
from agents.nodes.input import make_normalize_input_node
from agents.nodes.merge_deduplicate import make_merge_deduplicate_node
from agents.nodes.reasoning import make_reason_node
from agents.nodes.rerank import make_rerank_node
from agents.nodes.retrieval import make_retrieval_node
from agents.nodes.web_search import make_web_search_node
from agents.runtime import AgentRuntime
from agents.state import MAX_WEB_SEARCH_ITERATIONS


class RetrievalDiagnosticsTests(unittest.TestCase):
    def test_empty_distances_are_insufficient(self) -> None:
        diagnostics = compute_retrieval_diagnostics([])

        self.assertEqual(diagnostics.chunk_count, 0)
        self.assertEqual(diagnostics.coverage_verdict, "insufficient")

    def test_analysis_node_returns_diagnostics_and_llm_pass_verdict(self) -> None:
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value.verdict = "pass"
        runtime = AgentRuntime(
            model_name="test-model",
            embedding_model="test-embedding",
            vllm_base_url="http://vllm.test/v1",
            rerank_model="test-reranker",
            llm_factory=lambda _: llm,
        )

        result = make_analysis_node(runtime)(
            {
                "prompt": "question",
                "response": "",
                "context": "[Context 1 | source: test]\nRelevant answer",
                "retrieval_distances": [0.2, 0.4],
            }
        )

        self.assertEqual(set(result), {"retrieval_diagnostics", "analysis_verdict"})
        self.assertEqual(result["retrieval_diagnostics"].coverage_verdict, "sufficient")
        self.assertEqual(result["analysis_verdict"], "pass")
        assessment_prompt = (
            llm.with_structured_output.return_value.invoke.call_args.args[0]
        )
        self.assertIn("question", assessment_prompt)
        self.assertIn("Relevant answer", assessment_prompt)
        self.assertIn("similarity scores", assessment_prompt)


class GraphTopologyTests(unittest.TestCase):
    def test_normalize_input_sanitizes_prompt_before_routing(self) -> None:
        result = make_normalize_input_node()(
            {
                "prompt": (
                    "<b>Plan</b> my trip!!! Email me@example.com or call "
                    "+1 (555) 123-4567. SSN 123-45-6789. \x00"
                ),
                "response": "",
            }
        )

        self.assertEqual(
            result,
            {
                "prompt": (
                    "Plan my trip!!! Email [email removed] or call [phone removed]. "
                    "SSN [ssn removed]."
                )
            },
        )

    def test_research_route_flows_through_analysis(self) -> None:
        graph = agent_graph.get_graph()
        nodes = set(graph.nodes)
        edges = {(edge.source, edge.target) for edge in graph.edges}

        self.assertTrue(
            {
                "__start__",
                "__end__",
                "normalize_input",
                "router",
                "direct",
                "retrieval",
                "analysis",
                "expand_query",
                "merge_deduplicate",
                "rerank",
                "reason",
                "web_search",
            }
            <= nodes
        )
        self.assertTrue(
            {
                ("__start__", "normalize_input"),
                ("normalize_input", "router"),
                ("retrieval", "analysis"),
                ("analysis", "rerank"),
                ("analysis", "expand_query"),
                ("expand_query", "merge_deduplicate"),
                ("merge_deduplicate", "rerank"),
                ("rerank", "reason"),
                ("reason", "web_search"),
                ("reason", "__end__"),
                ("web_search", "reason"),
            }
            <= edges
        )

    def test_web_search_node_records_results_and_increments_iteration(self) -> None:
        runtime = AgentRuntime(
            model_name="test-model",
            embedding_model="test-embedding",
            vllm_base_url="http://vllm.test/v1",
            rerank_model="test-reranker",
            llm_factory=MagicMock(),
            web_search=lambda query: f"result for {query}",
        )

        result = make_web_search_node(runtime)(
            {
                "prompt": "question",
                "response": "",
                "web_search_query": "latest research",
                "web_search_iterations": 2,
                "web_search_results": ["previous result"],
            }
        )

        self.assertEqual(result["web_search_iterations"], MAX_WEB_SEARCH_ITERATIONS)
        self.assertEqual(result["web_search_query"], "")
        self.assertEqual(len(result["web_search_results"]), 2)
        self.assertIn("latest research", result["web_search_results"][-1])


class ReasoningWebSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = MagicMock()
        self.bound_llm = MagicMock()
        self.llm.bind_tools.return_value = self.bound_llm
        self.runtime = AgentRuntime(
            model_name="test-model",
            embedding_model="test-embedding",
            vllm_base_url="http://vllm.test/v1",
            rerank_model="test-reranker",
            llm_factory=lambda _: self.llm,
        )

    def test_reasoning_returns_web_search_request_from_tool_call(self) -> None:
        self.bound_llm.return_value = SimpleNamespace(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "current findings"}}],
        )

        result = make_reason_node(self.runtime)({"prompt": "question", "response": ""})

        self.assertEqual(
            result, {"response": "", "web_search_query": "current findings"}
        )
        self.llm.bind_tools.assert_called_once()

    def test_reasoning_disables_tool_calls_after_three_searches(self) -> None:
        self.llm.return_value = SimpleNamespace(content="final answer", tool_calls=[])

        result = make_reason_node(self.runtime)(
            {
                "prompt": "question",
                "response": "",
                "web_search_iterations": MAX_WEB_SEARCH_ITERATIONS,
            }
        )

        self.assertEqual(result, {"response": "final answer", "web_search_query": ""})
        self.llm.bind_tools.assert_not_called()


class QueryExpansionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = MagicMock()
        self.runtime = AgentRuntime(
            model_name="test-model",
            embedding_model="test-embedding",
            vllm_base_url="http://vllm.test/v1",
            rerank_model="test-reranker",
            llm_factory=lambda _: self.llm,
        )

    @patch("agents.nodes.retrieval.retrieve")
    def test_initial_retrieval_records_original_query(
        self, retrieve_mock: MagicMock
    ) -> None:
        retrieve_mock.return_value = (
            ["original chunk"],
            [{"filename": "test"}],
            [0.2],
        )

        result = make_retrieval_node(self.runtime)(
            {"prompt": "question", "response": ""}
        )

        self.assertEqual(
            result["query_retrievals"],
            [
                {
                    "query": "question",
                    "query_type": "original",
                    "document_chunks": ["original chunk"],
                    "metadatas": [{"filename": "test"}],
                    "distances": [0.2],
                }
            ],
        )

    @patch("agents.nodes.expand_query.retrieve")
    def test_expanded_retrievals_append_to_original_record(
        self, retrieve_mock: MagicMock
    ) -> None:
        self.llm.with_structured_output.return_value.invoke.return_value.queries = [
            "alternative one",
            "alternative two",
        ]
        retrieve_mock.side_effect = [
            (["first chunk"], [{"filename": "first"}], [0.2]),
            (["second chunk"], [{"filename": "second"}], [0.4]),
        ]
        original_record = {
            "query": "question",
            "query_type": "original",
            "document_chunks": ["original chunk"],
            "metadatas": [{"filename": "original"}],
            "distances": [0.1],
        }

        result = make_expand_query_node(self.runtime)(
            {
                "prompt": "question",
                "response": "",
                "query_retrievals": [original_record],
            }
        )

        self.assertEqual(
            [record["query"] for record in result["query_retrievals"]],
            ["question", "alternative one", "alternative two"],
        )
        self.assertEqual(
            [record["query_type"] for record in result["query_retrievals"]],
            ["original", "generated", "generated"],
        )
        self.assertEqual(retrieve_mock.call_count, 2)

    def test_merge_deduplicate_combines_all_query_retrievals(self) -> None:
        result = make_merge_deduplicate_node(self.runtime)(
            {
                "prompt": "question",
                "response": "",
                "query_retrievals": [
                    {
                        "query": "question",
                        "query_type": "original",
                        "document_chunks": ["Original chunk", "Shared chunk"],
                        "metadatas": [{}, {}],
                        "distances": [0.1, 0.2],
                    },
                    {
                        "query": "alternative",
                        "query_type": "generated",
                        "document_chunks": ["  shared   CHUNK ", "Generated chunk"],
                        "metadatas": [{}, {}],
                        "distances": [0.3, 0.4],
                    },
                ],
            }
        )

        self.assertEqual(
            result["final_document_chunks"],
            ["Original chunk", "Shared chunk", "Generated chunk"],
        )

    @patch("agents.nodes.rerank.rerank")
    def test_rerank_keeps_top_five_chunks_in_reranker_order(
        self, rerank_mock: MagicMock
    ) -> None:
        rerank_mock.return_value = [5, 2, 4, 0, 3, 1]

        result = make_rerank_node(self.runtime)(
            {
                "prompt": "question",
                "response": "",
                "final_document_chunks": [f"chunk {index}" for index in range(6)],
            }
        )

        self.assertEqual(
            result["final_document_chunks"],
            ["chunk 5", "chunk 2", "chunk 4", "chunk 0", "chunk 3"],
        )
        rerank_mock.assert_called_once_with(
            "question",
            [f"chunk {index}" for index in range(6)],
            model_name="test-reranker",
            base_url="http://vllm.test/v1",
        )


if __name__ == "__main__":
    unittest.main()
