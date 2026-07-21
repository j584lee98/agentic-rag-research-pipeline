from pydantic import BaseModel


class InvokeRequest(BaseModel):
    prompt: str


class InvokeResponse(BaseModel):
    response: str


class IngestDocumentResponse(BaseModel):
    document_id: str
    filename: str
    stored_path: str
    chunks_ingested: int
    collection_name: str


class DeleteDocumentResponse(BaseModel):
    document_id: str
    file_deleted: bool
    embeddings_deleted: int
    collection_name: str
