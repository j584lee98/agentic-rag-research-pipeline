import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_model: str
    openai_embedding_model: str
    vllm_base_url: str
    vllm_rerank_model: str
    chroma_collection_name: str
    chroma_persist_dir: Path
    documents_dir: Path
    chunk_size: int
    chunk_overlap: int


def _validate_chunking(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("CHUNK_OVERLAP must be greater than or equal to 0.")
    if chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings(
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
        openai_embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        vllm_rerank_model=os.getenv("VLLM_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        chroma_collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME", "research_documents"
        ),
        chroma_persist_dir=Path(
            os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
        ).resolve(),
        documents_dir=Path(os.getenv("DOCUMENTS_DIR", "data/documents")).resolve(),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
    )

    _validate_chunking(settings.chunk_size, settings.chunk_overlap)
    return settings
