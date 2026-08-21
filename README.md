# agentic-rag-research-pipeline

## Overview

This project contains:

- a routed LangGraph research agent with deterministic retrieval diagnostics
- a document ingestion endpoint that chunks and embeds uploads into ChromaDB

## LangGraph agent

- Graph entry point: `agent_graph` in `agents/graph.py`
- Reasoning route: `retrieval` -> `analysis` -> `rerank` -> `reason`, with
  query expansion and deduplication before reranking when analysis fails
- Direct route: `direct`
- Invocation helper: `invoke_agent(prompt: str) -> str` in `agents/graph.py`
- API schemas: `InvokeRequest` and `InvokeResponse` in `app/schemas.py`

### Graph structure

- `agents/graph.py`: graph assembly and public invocation API
- `agents/state.py`: shared graph state, diagnostics types, and constants
- `agents/runtime.py`: model runtime configuration
- `agents/retrieval.py`: ChromaDB retrieval and context formatting
- `agents/analysis.py`: deterministic retrieval diagnostics
- `agents/nodes/`: individual routing, retrieval, analysis, and reasoning nodes

The router chooses either `direct` -> `END` or `retrieval` -> `analysis`.
Analysis computes deterministic score statistics, then asks an LLM for a
structured `pass` or `fail` assessment using the prompt, retrieved context, and
statistics. `pass` continues to `rerank` -> `reason` -> `END`; `fail` expands
the query, retrieves additional chunks, merges and deduplicates them, then
reranks the candidates. The vLLM reranker selects the five highest-ranked chunks
for reasoning. Nodes return only the state fields they update, allowing future
branches to add state without overwriting unrelated values.

### Environment variables

Set your OpenAI API key before invoking the agent:

```bash
set OPENAI_API_KEY=your_api_key_here
```

Optional model override:

```bash
set OPENAI_MODEL=gpt-5-nano
```

Run a vLLM server with a reranker model, then configure its OpenAI-compatible
base URL and model name (these are the defaults):

```bash
set VLLM_BASE_URL=http://localhost:8000/v1
set VLLM_RERANK_MODEL=BAAI/bge-reranker-v2-m3
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

## Run with Docker

Docker Compose starts the FastAPI application, a persistent Chroma server, and
a GPU-backed vLLM reranker. Copy the environment template, supply the required
OpenAI API key, then start the stack:

```bash
copy .env.example .env
docker compose up --build
```

The API is available at `http://localhost:8000`, Chroma is exposed on port
`8001`, and vLLM is exposed on port `8002`. The vLLM container requires a
working NVIDIA Container Toolkit installation and a compatible NVIDIA GPU. Its
first startup downloads the reranker model into the `huggingface_cache` volume.

The application automatically uses Chroma's HTTP API when `CHROMA_HOST` is
set; leaving it unset preserves the local persistent Chroma behavior used for
non-container development.

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

### Delete a document

Deletes all chunk embeddings in ChromaDB for the document and removes the stored source file.

```bash
curl -X DELETE "http://127.0.0.1:8000/documents/<document_id>"
```

Expected response shape:

```json
{
	"document_id": "<document_id>",
	"file_deleted": true,
	"embeddings_deleted": 8,
	"collection_name": "research_documents"
}
```

## Pre-commit hook (Ruff, single-commit flow)

This repo uses a Git pre-commit hook that:

- runs `ruff format` on staged Python files
- runs `ruff check --fix` on staged Python files
- re-stages changed files automatically
- runs a final `ruff check`

This allows a single `git commit` command to succeed after formatting/fixes, without re-running commit manually.

### One-time setup

```bash
uv sync --group dev
git config core.hooksPath .githooks
```

### Run manually (optional)

```bash
uv run ruff format .
uv run ruff check --fix .
uv run ruff check .
```