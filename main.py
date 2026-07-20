from fastapi import FastAPI

from agents.graph import invoke_agent
from app.schemas import InvokeRequest, InvokeResponse


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
