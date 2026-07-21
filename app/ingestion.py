import hashlib
import uuid
from io import BytesIO
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from fastapi import HTTPException, UploadFile
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
}

DOCUMENTS_DIR = Path("data/documents").resolve()
CHROMA_PERSIST_DIR = Path("data/chroma").resolve()
CHROMA_COLLECTION_NAME = "research_documents"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def _decode_text_bytes(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise HTTPException(status_code=400, detail="Unable to decode text file.")


def _extract_pdf_text(raw: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(raw))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to read PDF: {exc}"
        ) from exc

    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n\n".join(pages).strip()


def _extract_document_text(filename: str, raw: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        return _extract_pdf_text(raw)
    return _decode_text_bytes(raw)


def _ensure_supported_file(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Allowed: {allowed}",
        )
    return extension


def _save_uploaded_file(filename: str, extension: str, raw: bytes) -> tuple[str, Path]:
    document_id = uuid.uuid4().hex
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = Path(filename).stem.replace(" ", "_")
    stored_filename = f"{safe_name}_{document_id}{extension}"
    destination = DOCUMENTS_DIR / stored_filename
    destination.write_bytes(raw)

    return document_id, destination


def _build_checksum(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def ingest_upload(upload: UploadFile) -> dict[str, str | int]:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="A filename is required.")

    extension = _ensure_supported_file(upload.filename)
    raw = upload.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    checksum = _build_checksum(raw)

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    existing = collection.get(
        where={"checksum": checksum}, include=["metadatas"], limit=1
    )
    if existing.get("ids"):
        raise HTTPException(
            status_code=409, detail="Duplicate document already ingested."
        )

    text = _extract_document_text(upload.filename, raw)
    if not text.strip():
        raise HTTPException(
            status_code=400, detail="No extractable text found in document."
        )

    document_id, stored_path = _save_uploaded_file(upload.filename, extension, raw)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = [chunk for chunk in splitter.split_text(text) if chunk.strip()]
    if not chunks:
        raise HTTPException(
            status_code=400, detail="No text chunks generated from document."
        )

    embeddings_client = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

    try:
        embeddings = embeddings_client.embed_documents(chunks)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Embedding generation failed: {exc}"
        ) from exc

    ids = [f"{document_id}-{idx}" for idx, _ in enumerate(chunks)]
    metadatas = [
        {
            "document_id": document_id,
            "filename": upload.filename,
            "stored_path": str(stored_path),
            "chunk_index": idx,
            "chunk_count": len(chunks),
            "checksum": checksum,
        }
        for idx, _ in enumerate(chunks)
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return {
        "document_id": document_id,
        "filename": upload.filename,
        "stored_path": str(stored_path),
        "chunks_ingested": len(chunks),
        "collection_name": CHROMA_COLLECTION_NAME,
    }


def delete_document(document_id: str) -> dict[str, str | int | bool]:
    if not document_id.strip():
        raise HTTPException(status_code=400, detail="document_id is required.")

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    existing = collection.get(where={"document_id": document_id}, include=["metadatas"])
    ids = existing.get("ids", [])
    if not ids:
        raise HTTPException(status_code=404, detail="Document not found.")

    metadatas = existing.get("metadatas", [])
    stored_path_value = ""
    if metadatas and metadatas[0]:
        stored_path_value = str(metadatas[0].get("stored_path", ""))

    collection.delete(where={"document_id": document_id})

    file_deleted = False
    if stored_path_value:
        stored_path = Path(stored_path_value)
        if stored_path.exists() and stored_path.is_file():
            stored_path.unlink()
            file_deleted = True

    return {
        "document_id": document_id,
        "file_deleted": file_deleted,
        "embeddings_deleted": len(ids),
        "collection_name": CHROMA_COLLECTION_NAME,
    }
