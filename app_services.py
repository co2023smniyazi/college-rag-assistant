from pathlib import Path

from embeddings import OllamaEmbedder
from llm_service import OllamaAnswerGenerator
from rag_service import RAGService
from retriever import Retriever
from vector_store import ChromaVectorStore


def build_rag_service(
    database_path: str | Path,
) -> tuple[ChromaVectorStore, OllamaEmbedder, RAGService]:
    store = ChromaVectorStore(path=database_path)
    embedder = OllamaEmbedder()
    service = RAGService(
        retriever=Retriever(embedder=embedder, store=store),
        answer_generator=OllamaAnswerGenerator(),
    )
    return store, embedder, service
