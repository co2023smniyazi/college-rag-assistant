import logging
import os
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


def resolve_ollama_host(
    secrets: Mapping[str, Any] | None = None,
) -> str | None:
    environment_host = os.environ.get("OLLAMA_HOST")
    if environment_host:
        return environment_host
    if secrets is None:
        return None

    secret_host = secrets.get("OLLAMA_HOST")
    if secret_host is not None and not isinstance(secret_host, str):
        raise ValueError("OLLAMA_HOST must be configured as a string.")
    return secret_host or None


def get_ollama_status(
    ollama_host: str | None = None,
) -> tuple[bool, set[str], str]:
    try:
        response: Any = create_ollama_client(ollama_host).list()
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
        logger.exception("Could not check the configured Ollama service.")
        return (
            False,
            set(),
            "Ollama is not reachable. Check the local service or configure a "
            "reachable OLLAMA_HOST endpoint.",
        )


def model_is_available(model_name: str, installed_models: set[str]) -> bool:
    return model_name in installed_models or f"{model_name}:latest" in installed_models


def create_ollama_client(host: str | None = None) -> ollama.Client:
    if host:
        return ollama.Client(host=host)
    return ollama.Client()


def build_rag_service(
    database_path: str | Path,
    ollama_host: str | None = None,
) -> tuple[ChromaVectorStore, OllamaEmbedder, RAGService]:
    store = ChromaVectorStore(path=database_path)
    client = create_ollama_client(ollama_host)
    embedder = OllamaEmbedder(client=client)
    service = RAGService(
        retriever=Retriever(embedder=embedder, store=store),
        answer_generator=OllamaAnswerGenerator(client=client),
    )
    return store, embedder, service
