import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from langchain_core.runnables.graph_mermaid import draw_mermaid_png

from agents.graph import agent_graph

from app.schemas import (
    DeleteDocumentResponse,
    IngestDocumentResponse,
    InvokeRequest,
    InvokeResponse,
)
from app.services import AgentService, DocumentService


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    output_path = Path(__file__).resolve().parent / "flow.png"

    try:
        mermaid_syntax = agent_graph.get_graph().draw_mermaid()
        mermaid_syntax = mermaid_syntax.replace(
            "router -.-> direct;",
            "router -. &nbsp;direct&nbsp; .-> direct;",
            1,
        )
        mermaid_syntax = mermaid_syntax.replace(
            "router -.-> retrieval;",
            "router -. &nbsp;reason&nbsp; .-> retrieval;",
            1,
        )
        lr_mermaid_syntax = mermaid_syntax.replace("graph TD;", "graph LR;", 1)
        graph_png = draw_mermaid_png(lr_mermaid_syntax)
        output_path.write_bytes(graph_png)
        logger.info("Saved LangGraph flow diagram to %s", output_path)
    except Exception:
        logger.exception("Failed to generate LangGraph flow diagram.")

    yield


app = FastAPI(title="Agentic RAG Research Pipeline", lifespan=lifespan)

agent_service = AgentService()
document_service = DocumentService()


@app.get("/")
async def hello_world() -> dict[str, str]:
    return {"message": "hello world"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/agent/invoke", response_model=InvokeResponse)
async def invoke(request: InvokeRequest) -> InvokeResponse:
    return InvokeResponse(response=agent_service.invoke(request.prompt))


@app.post("/documents/ingest", response_model=IngestDocumentResponse)
async def ingest_document(
    file: Annotated[UploadFile, File()],
) -> IngestDocumentResponse:
    result = await document_service.ingest(file)
    return IngestDocumentResponse(**result)


@app.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
async def remove_document(document_id: str) -> DeleteDocumentResponse:
    result = document_service.delete(document_id)
    return DeleteDocumentResponse(**result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
