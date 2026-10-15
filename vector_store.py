import hashlib
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from models import DocumentChunk, RetrievedChunk


COLLECTION_NAME = "college_documents"


class VectorStoreError(RuntimeError):
    """Raised when the local Chroma database cannot complete an operation."""


def document_id_for(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


class ChromaVectorStore:
    def __init__(self, path: str | Path = "chroma_db", client: Any | None = None) -> None:
        try:
            self.client = client or chromadb.PersistentClient(
                path=str(path),
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as error:
            raise VectorStoreError(
                "Could not open the local ChromaDB database. Check that the "
                "chroma_db folder is writable."
            ) from error

    def has_document(self, document_id: str) -> bool:
        try:
            result = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
                limit=1,
            )
            return bool(result["ids"])
        except Exception as error:
            raise VectorStoreError("Could not check the indexed document.") from error

    def is_empty(self) -> bool:
        try:
            return self.collection.count() == 0
        except Exception as error:
            raise VectorStoreError("Could not check the ChromaDB collection.") from error

    def add_chunks(
        self,
        document_id: str,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Every document chunk must have exactly one embedding.")
        if not chunks:
            raise ValueError("Cannot store a document without text chunks.")

        ids = [
            f"{document_id}:{chunk.page_number}:{chunk.chunk_index}:"
            f"{hashlib.sha256(chunk.text.encode()).hexdigest()[:12]}"
            for chunk in chunks
        ]
        try:
            self.collection.upsert(
                ids=ids,
                documents=[chunk.text for chunk in chunks],
                metadatas=[
                    {
                        "source": chunk.source,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "document_id": document_id,
                    }
                    for chunk in chunks
                ],
                embeddings=embeddings,
            )
        except Exception as error:
            raise VectorStoreError("Could not save document chunks in ChromaDB.") from error

    def search(
        self, query_embedding: list[float], limit: int = 3
    ) -> list[RetrievedChunk]:
        if limit <= 0:
            return []
        try:
            collection_size = self.collection.count()
            if collection_size == 0:
                return []
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(limit, collection_size),
                include=["documents", "metadatas", "distances"],
            )
            documents = result["documents"][0]
            metadatas = result["metadatas"][0]
            distances = result["distances"][0]
            return [
                RetrievedChunk(
                    source=metadata["source"],
                    page_number=int(metadata["page_number"]),
                    text=text,
                    distance=float(distance),
                )
                for text, metadata, distance in zip(
                    documents, metadatas, distances, strict=True
                )
            ]
        except Exception as error:
            raise VectorStoreError("Could not search the ChromaDB collection.") from error

    def list_documents(self) -> list[dict[str, str]]:
        try:
            result = self.collection.get(include=["metadatas"])
            documents: dict[str, dict[str, str]] = {}
            for metadata in result["metadatas"]:
                identifier = metadata["document_id"]
                documents[identifier] = {
                    "document_id": identifier,
                    "source": metadata["source"],
                }
            return sorted(documents.values(), key=lambda item: item["source"].lower())
        except Exception as error:
            raise VectorStoreError("Could not list indexed documents.") from error

    def delete_document(self, document_id: str) -> None:
        try:
            if self.has_document(document_id):
                self.collection.delete(where={"document_id": document_id})
        except Exception as error:
            raise VectorStoreError("Could not remove the indexed document.") from error
