import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import ollama

from chunker import chunk_pages
from document_loader import DocumentProcessingError, extract_pdf_pages
from embeddings import OllamaEmbedder
from llm_service import OllamaAnswerGenerator
from rag_service import RAGService
from retriever import Retriever
from vector_store import ChromaVectorStore, document_id_for


logger = logging.getLogger(__name__)
CF_ACCESS_CLIENT_ID = "CF_ACCESS_CLIENT_ID"
CF_ACCESS_CLIENT_SECRET = "CF_ACCESS_CLIENT_SECRET"
CF_ACCESS_CLIENT_ID_HEADER = "CF-Access-Client-Id"
CF_ACCESS_CLIENT_SECRET_HEADER = "CF-Access-Client-Secret"
OLLAMA_API_KEY = "OLLAMA_API_KEY"


class OllamaAccessConfigurationError(ValueError):
    """Raised when Ollama endpoint or authentication settings are invalid."""


class ChromaConfigurationError(ValueError):
    """Raised when remote ChromaDB configuration is incomplete or invalid."""


@dataclass(frozen=True)
class PdfIndexResult:
    already_indexed: bool
    page_count: int
    chunk_count: int


def _configured_value(
    name: str, secrets: Mapping[str, Any] | None
) -> Any:
    environment_value = os.environ.get(name)
    if environment_value is not None:
        return environment_value
    return secrets.get(name) if secrets is not None else None


def resolve_ollama_host(
    secrets: Mapping[str, Any] | None = None,
) -> str | None:
    configured_mode = _configured_value("OLLAMA_MODE", secrets)
    if configured_mode is not None and not isinstance(configured_mode, str):
        raise OllamaAccessConfigurationError("OLLAMA_MODE must be local or remote.")
    mode = configured_mode.casefold().strip() if configured_mode else None
    remote_host = _configured_value("OLLAMA_REMOTE_HOST", secrets)
    legacy_host = _configured_value("OLLAMA_HOST", secrets)
    if remote_host is None:
        remote_host = legacy_host
    if mode is None:
        mode = (
            "remote"
            if remote_host
            and (
                _configured_value("OLLAMA_REMOTE_HOST", secrets) is not None
                or not isinstance(remote_host, str)
                or not _is_loopback_host(remote_host)
            )
            else "local"
        )
    if mode not in {"local", "remote"}:
        raise OllamaAccessConfigurationError("OLLAMA_MODE must be local or remote.")

    if mode == "remote":
        if not isinstance(remote_host, str) or not remote_host.strip():
            raise OllamaAccessConfigurationError(
                "Set OLLAMA_REMOTE_HOST to a network-reachable Ollama endpoint."
            )
        remote_host = remote_host.strip()
        if _is_loopback_host(remote_host):
            raise OllamaAccessConfigurationError(
                "Remote Ollama mode cannot use localhost or a loopback address."
            )
        return remote_host

    local_host = _configured_value("OLLAMA_LOCAL_HOST", secrets)
    if (
        local_host is None
        and _configured_value("OLLAMA_REMOTE_HOST", secrets) is None
        and legacy_host
        and isinstance(legacy_host, str)
        and _is_loopback_host(legacy_host)
    ):
        local_host = legacy_host
    if local_host is not None and not isinstance(local_host, str):
        raise OllamaAccessConfigurationError("OLLAMA_LOCAL_HOST must be a string.")
    if not local_host:
        return None
    return local_host.strip() or None


def resolve_chroma_settings(
    secrets: Mapping[str, Any] | None = None,
) -> tuple[str | None, int, bool, str | None]:
    host_value = _configured_value("CHROMA_HOST", secrets)
    if host_value is not None and not isinstance(host_value, str):
        raise ChromaConfigurationError("CHROMA_HOST must be a hostname.")
    host = host_value.strip() if host_value else None
    if host:
        try:
            parsed = urlsplit(f"//{host}")
            configured_port = parsed.port
        except ValueError as error:
            raise ChromaConfigurationError(
                "CHROMA_HOST must be a valid hostname without a port."
            ) from error
        if (
            parsed.hostname is None
            or configured_port is not None
            or parsed.path
            or parsed.username
            or parsed.password
        ):
            raise ChromaConfigurationError(
                "CHROMA_HOST must be a hostname only; configure port and TLS separately."
            )
        host = parsed.hostname
        if _is_loopback_host(host):
            raise ChromaConfigurationError(
                "Remote CHROMA_HOST cannot use localhost or a loopback address."
            )

    port_value = _configured_value("CHROMA_PORT", secrets)
    if isinstance(port_value, bool):
        raise ChromaConfigurationError("CHROMA_PORT must be an integer.")
    try:
        port = int(port_value) if port_value is not None else 8000
    except (TypeError, ValueError) as error:
        raise ChromaConfigurationError("CHROMA_PORT must be an integer.") from error
    if not 1 <= port <= 65535:
        raise ChromaConfigurationError("CHROMA_PORT must be between 1 and 65535.")

    ssl_value = _configured_value("CHROMA_SSL", secrets)
    if isinstance(ssl_value, bool):
        use_ssl = ssl_value
    elif ssl_value is None:
        use_ssl = False
    elif isinstance(ssl_value, str) and ssl_value.casefold().strip() in {
        "true",
        "false",
    }:
        use_ssl = ssl_value.casefold().strip() == "true"
    else:
        raise ChromaConfigurationError("CHROMA_SSL must be true or false.")

    token_value = _configured_value("CHROMA_AUTH_TOKEN", secrets)
    if token_value is not None and (
        not isinstance(token_value, str) or not token_value.strip()
    ):
        raise ChromaConfigurationError("CHROMA_AUTH_TOKEN must be a non-empty string.")
    token = token_value.strip() if isinstance(token_value, str) else None
    if token and not host:
        raise ChromaConfigurationError(
            "CHROMA_AUTH_TOKEN requires a configured remote CHROMA_HOST."
        )
    if token and not use_ssl:
        raise ChromaConfigurationError(
            "CHROMA_AUTH_TOKEN requires CHROMA_SSL=true to protect credentials."
        )
    return host, port, use_ssl, token


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
            "network-reachable OLLAMA_REMOTE_HOST endpoint.",
        )


def model_is_available(model_name: str, installed_models: set[str]) -> bool:
    return model_name in installed_models or f"{model_name}:latest" in installed_models


def create_ollama_client(
    host: str | None = None,
    access_headers: Mapping[str, str] | None = None,
) -> ollama.Client:
    if host:
        client_options: dict[str, Any] = {"host": host}
        if access_headers and not _is_loopback_host(host):
            client_options["headers"] = dict(access_headers)
        return ollama.Client(**client_options)
    return ollama.Client()


def resolve_ollama_access_headers(
    host: str | None,
    secrets: Mapping[str, Any] | None = None,
) -> dict[str, str] | None:
    if not host or _is_loopback_host(host):
        return None

    client_id = _configured_value(CF_ACCESS_CLIENT_ID, secrets)
    client_secret = _configured_value(CF_ACCESS_CLIENT_SECRET, secrets)

    if client_id is None and client_secret is None:
        headers: dict[str, str] = {}
    else:
        if not isinstance(client_id, str) or not client_id.strip():
            raise OllamaAccessConfigurationError(
                "Both Cloudflare Access client credentials must be configured together."
            )
        if not isinstance(client_secret, str) or not client_secret.strip():
            raise OllamaAccessConfigurationError(
                "Both Cloudflare Access client credentials must be configured together."
            )
        headers = {
            CF_ACCESS_CLIENT_ID_HEADER: client_id,
            CF_ACCESS_CLIENT_SECRET_HEADER: client_secret,
        }

    api_key = _configured_value(OLLAMA_API_KEY, secrets)
    if api_key is not None and (
        not isinstance(api_key, str) or not api_key.strip()
    ):
        raise OllamaAccessConfigurationError(
            "OLLAMA_API_KEY must be a non-empty string."
        )
    if api_key:
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    if headers and urlsplit(host).scheme.casefold() != "https":
        raise OllamaAccessConfigurationError(
            "Remote Ollama credentials require an HTTPS endpoint."
        )

    return headers or None


def _is_loopback_host(host: str) -> bool:
    try:
        parsed_host = urlsplit(host).hostname
    except ValueError:
        parsed_host = None
    if parsed_host is None:
        try:
            parsed_host = urlsplit(f"//{host}").hostname
        except ValueError:
            return False
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
    chroma_host: str | None = None,
    chroma_port: int = 8000,
    chroma_ssl: bool = False,
    chroma_auth_token: str | None = None,
) -> tuple[ChromaVectorStore, OllamaEmbedder, RAGService]:
    store = ChromaVectorStore(
        path=database_path,
        host=chroma_host,
        port=chroma_port,
        ssl=chroma_ssl,
        auth_token=chroma_auth_token,
    )
    client = create_ollama_client(ollama_host, access_headers=access_headers)
    embedder = OllamaEmbedder(client=client)
    service = RAGService(
        retriever=Retriever(embedder=embedder, store=store),
        answer_generator=OllamaAnswerGenerator(client=client),
    )
    return store, embedder, service


def index_uploaded_pdf(
    filename: str,
    pdf_bytes: bytes,
    store: ChromaVectorStore,
    embedder: OllamaEmbedder,
) -> PdfIndexResult:
    identifier = document_id_for(pdf_bytes)
    if store.has_document(identifier):
        return PdfIndexResult(already_indexed=True, page_count=0, chunk_count=0)

    pages = extract_pdf_pages(pdf_bytes, filename)
    chunks = chunk_pages(pages)
    if not chunks:
        raise DocumentProcessingError(
            "No text chunks could be created from the uploaded PDF."
        )

    store.ensure_embedding_model(embedder.embed_texts)
    vectors = embedder.embed_texts([chunk.text for chunk in chunks])
    store.add_chunks(identifier, chunks, vectors)
    return PdfIndexResult(
        already_indexed=False,
        page_count=len(pages),
        chunk_count=len(chunks),
    )
