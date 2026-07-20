# agentic-rag-research-pipeline

## Basic LangGraph Agent (single node)

This project now includes a minimal LangGraph setup that is easy to expand into a multi-agent system.

- Graph entry point: `agent_graph` in `agents/graph.py`
- Node: `agent_node` in `agents/graph.py`
- Invocation helper: `invoke_agent(prompt: str) -> str` in `agents/graph.py`
- API schemas: `InvokeRequest` and `InvokeResponse` in `app/schemas.py`

### Environment variables

Set your OpenAI API key before invoking the agent:

```bash
set OPENAI_API_KEY=your_api_key_here
```

Optional model override:

```bash
set OPENAI_MODEL=gpt-5-nano
```

### Run the API

```bash
uv run uvicorn main:app --reload
```

### Invoke the agent

```bash
curl -X POST "http://127.0.0.1:8000/agent/invoke" \
	-H "Content-Type: application/json" \
	-d "{\"prompt\":\"hello agent\"}"
```

Expected response shape:

```json
{"response":"<model output text>"}
```

## Pre-commit hook (Ruff)

This repo uses `pre-commit` to run Ruff automatically before each commit.

### One-time setup

```bash
uv sync --group dev
uv run pre-commit install
```

### Run manually (optional)

```bash
uv run pre-commit run --all-files
```