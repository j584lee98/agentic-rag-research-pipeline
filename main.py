from fastapi import FastAPI, File, UploadFile

from agents.graph import invoke_agent
from app.ingestion import ingest_upload
from app.schemas import IngestDocumentResponse, InvokeRequest, InvokeResponse


app = FastAPI(title="Agentic RAG Research Pipeline")


@app.get("/")
async def hello_world() -> dict[str, str]:
    return {"message": "hello world"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/agent/invoke", response_model=InvokeResponse)
async def invoke(request: InvokeRequest) -> InvokeResponse:
    return InvokeResponse(response=invoke_agent(request.prompt))


@app.post("/documents/ingest", response_model=IngestDocumentResponse)
async def ingest_document(file: UploadFile = File(...)) -> IngestDocumentResponse:
    result = ingest_upload(file)
    return IngestDocumentResponse(**result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
