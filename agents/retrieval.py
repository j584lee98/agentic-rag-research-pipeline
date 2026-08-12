import chromadb
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings


def get_collection() -> chromadb.Collection:
    settings = get_settings()
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    return client.get_or_create_collection(name=settings.chroma_collection_name)


def retrieve(
    prompt: str, embedding_model: str, top_k: int = 4
) -> tuple[list[str], list[dict[str, object] | None], list[float]]:
    """Return documents, metadata, and L2 distances for the top results."""
    query_embedding = OpenAIEmbeddings(model=embedding_model).embed_query(prompt)
    result = get_collection().query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    document_rows = result.get("documents") or []
    metadata_rows = result.get("metadatas") or []
    distance_rows = result.get("distances") or []

    return (
        document_rows[0] if document_rows else [],
        metadata_rows[0] if metadata_rows else [],
        distance_rows[0] if distance_rows else [],
    )


def format_context(
    documents: list[str], metadatas: list[dict[str, object] | None]
) -> str:
    formatted_chunks: list[str] = []

    for index, document in enumerate(documents, start=1):
        metadata = metadatas[index - 1] if index <= len(metadatas) else None
        source = "unknown"

        if metadata:
            filename = metadata.get("filename")
            chunk_index = metadata.get("chunk_index")
            chunk_count = metadata.get("chunk_count")
            if filename is not None:
                source = str(filename)
            if chunk_index is not None and chunk_count is not None:
                source = f"{source} (chunk {chunk_index}/{chunk_count})"

        formatted_chunks.append(
            f"[Context {index} | source: {source}]\n{document.strip()}"
        )

    return "\n\n".join(formatted_chunks)
