import unittest
from unittest.mock import MagicMock

from agents.analysis import compute_retrieval_diagnostics
from agents.graph import agent_graph
from agents.nodes.analysis import make_analysis_node
from agents.nodes.input import make_normalize_input_node
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
                ("analysis", "retrieval"),
                ("reason", "__end__"),
            }
            <= edges
        )


if __name__ == "__main__":
    unittest.main()
