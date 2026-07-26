from agents.graph import invoke_agent


class AgentService:
    def invoke(self, prompt: str) -> str:
        return invoke_agent(prompt)
