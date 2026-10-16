import logging
import os
from collections.abc import Mapping
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import ollama

from embeddings import OllamaEmbedder
from llm_service import OllamaAnswerGenerator
from rag_service import RAGService
from retriever import Retriever
from vector_store import ChromaVectorStore


logger = logging.getLogger(__name__)
CF_ACCESS_CLIENT_ID = "CF_ACCESS_CLIENT_ID"
CF_ACCESS_CLIENT_SECRET = "CF_ACCESS_CLIENT_SECRET"
CF_ACCESS_CLIENT_ID_HEADER = "CF-Access-Client-Id"
CF_ACCESS_CLIENT_SECRET_HEADER = "CF-Access-Client-Secret"


class OllamaAccessConfigurationError(ValueError):
    """Raised when Cloudflare Access credentials are incomplete or invalid."""


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
    access_headers: Mapping[str, str] | None = None,
) -> tuple[bool, set[str], str]:
    try:
        response: Any = create_ollama_client(
            ollama_host, access_headers=access_headers
        ).list()
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
        logger.error("Could not check the configured Ollama service.")
        return (
            False,
            set(),
            "Ollama is not reachable. Check the local service or configure a "
            "reachable OLLAMA_HOST endpoint.",
        )


def model_is_available(model_name: str, installed_models: set[str]) -> bool:
    return model_name in installed_models or f"{model_name}:latest" in installed_models


def create_ollama_client(
    host: str | None = None,
    access_headers: Mapping[str, str] | None = None,
) -> ollama.Client:
    if host:
        client_options: dict[str, Any] = {"host": host}
        if access_headers and not _is_local_ollama_host(host):
            client_options["headers"] = dict(access_headers)
        return ollama.Client(**client_options)
    return ollama.Client()


def resolve_ollama_access_headers(
    host: str | None,
    secrets: Mapping[str, Any] | None = None,
) -> dict[str, str] | None:
    if not host or _is_local_ollama_host(host):
        return None

    environment_id = os.environ.get(CF_ACCESS_CLIENT_ID)
    environment_secret = os.environ.get(CF_ACCESS_CLIENT_SECRET)
    if environment_id is not None or environment_secret is not None:
        client_id, client_secret = environment_id, environment_secret
    elif secrets is not None:
        client_id = secrets.get(CF_ACCESS_CLIENT_ID)
        client_secret = secrets.get(CF_ACCESS_CLIENT_SECRET)
    else:
        return None

    if client_id is None and client_secret is None:
        return None
    if not isinstance(client_id, str) or not client_id.strip():
        raise OllamaAccessConfigurationError(
            "Both Cloudflare Access client credentials must be configured together."
        )
    if not isinstance(client_secret, str) or not client_secret.strip():
        raise OllamaAccessConfigurationError(
            "Both Cloudflare Access client credentials must be configured together."
        )
    if urlsplit(host).scheme.casefold() != "https":
        raise OllamaAccessConfigurationError(
            "Cloudflare Access credentials require an HTTPS Ollama endpoint."
        )

    return {
        CF_ACCESS_CLIENT_ID_HEADER: client_id,
        CF_ACCESS_CLIENT_SECRET_HEADER: client_secret,
    }


def _is_local_ollama_host(host: str) -> bool:
    parsed_host = urlsplit(host).hostname
    if parsed_host is None:
        parsed_host = urlsplit(f"//{host}").hostname
    if parsed_host is None:
        return False
    if parsed_host.casefold() == "localhost":
        return True
    try:
        return ip_address(parsed_host).is_loopback
    except ValueError:
        return False


def build_rag_service(
    database_path: str | Path,
    ollama_host: str | None = None,
    access_headers: Mapping[str, str] | None = None,
) -> tuple[ChromaVectorStore, OllamaEmbedder, RAGService]:
    store = ChromaVectorStore(path=database_path)
    client = create_ollama_client(ollama_host, access_headers=access_headers)
    embedder = OllamaEmbedder(client=client)
    service = RAGService(
        retriever=Retriever(embedder=embedder, store=store),
        answer_generator=OllamaAnswerGenerator(client=client),
    )
    return store, embedder, service
