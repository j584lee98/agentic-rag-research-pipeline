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
    chroma_collection_name: str
    chroma_persist_dir: Path
    documents_dir: Path
    chunk_size: int
    chunk_overlap: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
        openai_embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
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
