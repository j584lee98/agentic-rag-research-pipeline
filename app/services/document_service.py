from fastapi import UploadFile

from app.ingestion import delete_document, ingest_upload


class DocumentService:
    def ingest(self, file: UploadFile) -> dict[str, str | int]:
        return ingest_upload(file)

    def delete(self, document_id: str) -> dict[str, str | int | bool]:
        return delete_document(document_id)
