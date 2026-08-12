from agents.analysis import compute_retrieval_diagnostics
from agents.state import AgentState, AgentStateUpdate


def make_analysis_node():
    def analysis_node(state: AgentState) -> AgentStateUpdate:
        return {
            "retrieval_diagnostics": compute_retrieval_diagnostics(
                state.get("retrieval_distances", [])
            )
        }

    return analysis_node
