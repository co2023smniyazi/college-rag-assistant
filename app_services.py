import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import ollama

from embeddings import OllamaEmbedder
from llm_service import OllamaAnswerGenerator
from rag_service import RAGService
from retriever import Retriever
from vector_store import ChromaVectorStore


logger = logging.getLogger(__name__)


def get_ollama_status() -> tuple[bool, set[str], str]:
    try:
        response: Any = ollama.Client().list()
        models = (
            response.get("models", [])
            if isinstance(response, Mapping)
            else getattr(response, "models", [])
        )
        names: set[str] = set()
        for model in models:
            name = (
                model.get("model") or model.get("name")
                if isinstance(model, Mapping)
                else getattr(model, "model", None) or getattr(model, "name", None)
            )
            if isinstance(name, str):
                names.add(name)
        return True, names, ""
    except Exception:
        logger.exception("Could not check the local Ollama service.")
        return (
            False,
            set(),
            "Ollama is not reachable. Start the Ollama application or run "
            "`ollama serve`.",
        )


def model_is_available(model_name: str, installed_models: set[str]) -> bool:
    return model_name in installed_models or f"{model_name}:latest" in installed_models


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
