import unittest
from unittest.mock import MagicMock, patch

from agents.analysis import compute_retrieval_diagnostics
from agents.graph import agent_graph
from agents.nodes.analysis import make_analysis_node
from agents.nodes.expand_query import make_expand_query_node
from agents.nodes.input import make_normalize_input_node
from agents.nodes.retrieval import make_retrieval_node
from agents.runtime import AgentRuntime


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
                "reason",
            }
            <= nodes
        )
        self.assertTrue(
            {
                ("__start__", "normalize_input"),
                ("normalize_input", "router"),
                ("retrieval", "analysis"),
                ("analysis", "reason"),
                ("analysis", "expand_query"),
                ("expand_query", "reason"),
                ("reason", "__end__"),
            }
            <= edges
        )


class QueryExpansionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = MagicMock()
        self.runtime = AgentRuntime(
            model_name="test-model",
            embedding_model="test-embedding",
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
        self.assertIn("original chunk", result["context"])
        self.assertIn("first chunk", result["context"])
        self.assertIn("second chunk", result["context"])
        self.assertEqual(retrieve_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
