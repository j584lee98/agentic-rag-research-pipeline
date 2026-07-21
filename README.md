# agentic-rag-research-pipeline

## Overview

This project contains:

- a minimal single-node LangGraph agent
- a document ingestion endpoint that chunks and embeds uploads into ChromaDB

## LangGraph agent

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

## Document ingestion service

The ingestion endpoint:

- accepts only `.txt`, `.md`, `.pdf`
- stores original files in `data/documents`
- chunks text with `chunk_size=1000`, `chunk_overlap=200`
- embeds chunks with `text-embedding-3-small`
- upserts vectors into ChromaDB at `data/chroma`, collection `research_documents`
- blocks duplicate uploads using SHA-256 checksum (returns HTTP 409)

## Run the API

```bash
uv run uvicorn main:app --reload
```

## API endpoints

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

### Ingest a document

```bash
curl -X POST "http://127.0.0.1:8000/documents/ingest" ^
	-H "accept: application/json" ^
	-F "file=@C:\\path\\to\\document.pdf"
```

Expected response shape:

```json
{
	"document_id": "<uuid>",
	"filename": "document.pdf",
	"stored_path": "<absolute path to saved file>",
	"chunks_ingested": 8,
	"collection_name": "research_documents"
}
```

Duplicate response example:

```json
{"detail":"Duplicate document already ingested."}
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