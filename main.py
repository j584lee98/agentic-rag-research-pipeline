from fastapi import FastAPI


app = FastAPI(title="Agentic RAG Research Pipeline")


@app.get("/")
async def hello_world() -> dict[str, str]:
    return {"message": "hello world"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
