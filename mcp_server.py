from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Callable

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from app_services import (
    build_rag_service,
    resolve_ollama_access_headers,
    resolve_ollama_host,
)
from embeddings import EmbeddingServiceError, OllamaEmbedder
from llm_service import LLMServiceError
from models import RetrievedChunk
from rag_service import RAGService
from retriever import DEFAULT_RESULT_COUNT
from vector_store import ChromaVectorStore, VectorStoreError


PROJECT_DIR = Path(__file__).resolve().parent
CHROMA_PATH = PROJECT_DIR / "chroma_db"
MAX_QUERY_LENGTH = 4000
MAX_RESULT_COUNT = 10

ServiceComponents = tuple[ChromaVectorStore, OllamaEmbedder, RAGService]
ServiceBuilder = Callable[[str | Path], ServiceComponents]


def _build_configured_services(database_path: str | Path) -> ServiceComponents:
    ollama_host = resolve_ollama_host()
    access_headers = resolve_ollama_access_headers(ollama_host)
    return build_rag_service(
        database_path,
        ollama_host=ollama_host,
        access_headers=access_headers,
    )


def _validate_query(query: str) -> str:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("The search query or question cannot be empty.")
    if len(normalized_query) > MAX_QUERY_LENGTH:
        raise ValueError(
            f"The search query or question cannot exceed {MAX_QUERY_LENGTH} characters."
        )
    return normalized_query


def _chunk_result(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "source": chunk.source,
        "page_number": chunk.page_number,
        "text": chunk.text,
        "distance": chunk.distance,
    }


def _error_result(error: Exception) -> dict[str, Any]:
    return {"status": "error", "message": str(error)}


def create_mcp_server(
    database_path: str | Path = CHROMA_PATH,
    service_builder: ServiceBuilder | None = None,
) -> FastMCP:
    """Create a read-only MCP server backed by the application's existing RAG services."""
    server = FastMCP("College RAG Assistant")
    read_only = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        openWorldHint=False,
    )
    build_services = service_builder or _build_configured_services
    components: ServiceComponents | None = None

    def get_components() -> ServiceComponents:
        nonlocal components
        if components is None:
            components = build_services(database_path)
        return components

    def retrieve(query: str, limit: int) -> list[RetrievedChunk]:
        _, _, rag_service = get_components()
        return rag_service.retriever.retrieve(query, limit=limit)

    @server.tool(annotations=read_only)
    def list_indexed_documents() -> dict[str, Any]:
        """List document names currently indexed in the existing ChromaDB collection."""
        try:
            store, _, _ = get_components()
            documents = store.list_documents()
        except VectorStoreError as error:
            return _error_result(error)
        return {
            "status": "ok" if documents else "empty",
            "count": len(documents),
            "documents": documents,
            "message": (
                "Indexed documents are available."
                if documents
                else "No documents are currently indexed."
            ),
        }

    @server.tool(annotations=read_only)
    def search_college_documents(
        query: Annotated[
            str,
            Field(
                description="A question or search query about the indexed college documents.",
                min_length=1,
                max_length=MAX_QUERY_LENGTH,
            ),
        ],
        limit: Annotated[
            int,
            Field(
                description="Maximum relevant passages to return, from 1 through 10.",
                ge=1,
                le=MAX_RESULT_COUNT,
            ),
        ] = DEFAULT_RESULT_COUNT,
    ) -> dict[str, Any]:
        """Search the existing RAG retriever and return relevant passages with sources."""
        try:
            normalized_query = _validate_query(query)
            chunks = retrieve(normalized_query, limit)
        except (EmbeddingServiceError, VectorStoreError) as error:
            return _error_result(error)
        return {
            "status": "ok" if chunks else "no_context",
            "query": normalized_query,
            "count": len(chunks),
            "results": [_chunk_result(chunk) for chunk in chunks],
            "message": (
                "Relevant document context was found."
                if chunks
                else "No relevant context was found in the indexed documents."
            ),
        }

    @server.tool(annotations=read_only)
    def ask_college_assistant(
        question: Annotated[
            str,
            Field(
                description="A question to answer using only indexed college documents.",
                min_length=1,
                max_length=MAX_QUERY_LENGTH,
            ),
        ],
        limit: Annotated[
            int,
            Field(
                description="Maximum relevant passages to use, from 1 through 10.",
                ge=1,
                le=MAX_RESULT_COUNT,
            ),
        ] = DEFAULT_RESULT_COUNT,
    ) -> dict[str, Any]:
        """Answer from retrieved document context using the existing Ollama RAG pipeline."""
        try:
            normalized_question = _validate_query(question)
            _, _, rag_service = get_components()
            answer, chunks = rag_service.ask(normalized_question, limit=limit)
        except (
            EmbeddingServiceError,
            LLMServiceError,
            VectorStoreError,
        ) as error:
            return _error_result(error)
        return {
            "status": "ok" if chunks else "no_context",
            "question": normalized_question,
            "answer": answer,
            "sources": [_chunk_result(chunk) for chunk in chunks],
            "message": (
                "Answer generated from retrieved document context."
                if chunks
                else "No relevant context was found; the assistant did not generate an answer."
            ),
        }

    @server.tool(annotations=read_only)
    def get_document_context(
        query: Annotated[
            str,
            Field(
                description="A question or topic to find relevant passages for.",
                min_length=1,
                max_length=MAX_QUERY_LENGTH,
            ),
        ],
        limit: Annotated[
            int,
            Field(
                description="Maximum relevant passages to return, from 1 through 10.",
                ge=1,
                le=MAX_RESULT_COUNT,
            ),
        ] = DEFAULT_RESULT_COUNT,
    ) -> dict[str, Any]:
        """Retrieve source-attributed context only; this tool does not generate an answer."""
        try:
            normalized_query = _validate_query(query)
            chunks = retrieve(normalized_query, limit)
        except (EmbeddingServiceError, VectorStoreError) as error:
            return _error_result(error)
        return {
            "status": "ok" if chunks else "no_context",
            "query": normalized_query,
            "count": len(chunks),
            "context": [_chunk_result(chunk) for chunk in chunks],
            "message": (
                "Relevant document context was found."
                if chunks
                else "No relevant context was found in the indexed documents."
            ),
        }

    return server


mcp = create_mcp_server()


if __name__ == "__main__":
    mcp.run(transport="stdio")
