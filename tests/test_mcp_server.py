import asyncio

import pytest

import mcp_server
from embeddings import EmbeddingServiceError
from llm_service import LLMServiceError, UNKNOWN_ANSWER
from mcp_server import create_mcp_server
from models import RetrievedChunk
from mcp.server.fastmcp.exceptions import ToolError
from mcp.shared.memory import create_connected_server_and_client_session
from rag_service import RAGService
from vector_store import VectorStoreError


class FakeRetriever:
    def __init__(self, chunks=None, error=None):
        self.chunks = chunks or []
        self.error = error
        self.calls = []

    def retrieve(self, query, limit):
        self.calls.append((query, limit))
        if self.error:
            raise self.error
        return self.chunks


class FakeAnswerGenerator:
    def __init__(self, answer="Attendance is 75%.", error=None):
        self.answer = answer
        self.error = error
        self.calls = []

    def generate(self, question, chunks):
        self.calls.append((question, chunks))
        if self.error:
            raise self.error
        return self.answer


class FakeStore:
    def __init__(self, documents=None, error=None):
        self.documents = documents or []
        self.error = error
        self.calls = 0

    def list_documents(self):
        self.calls += 1
        if self.error:
            raise self.error
        return self.documents


def make_server(store, retriever, answer_generator):
    service = RAGService(retriever, answer_generator)
    build_calls = []

    def build_services(database_path):
        build_calls.append(database_path)
        return store, object(), service

    return create_mcp_server(service_builder=build_services), build_calls


def invoke_tool(server, name, arguments=None):
    async def invoke():
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool(name, arguments or {})
            if result.isError:
                message = result.content[0].text if result.content else "MCP tool failed."
                raise ToolError(message)
            return result.structuredContent

    return asyncio.run(invoke())


def invoke_registered_tool(server, name, arguments=None):
    async def invoke():
        return await server._tool_manager.call_tool(name, arguments or {})

    return asyncio.run(invoke())


def list_client_tools(server):
    async def list_tools():
        async with create_connected_server_and_client_session(server) as client:
            return await client.list_tools()

    return asyncio.run(list_tools())


def test_registers_read_only_rag_tools_with_useful_schemas() -> None:
    server, _ = make_server(FakeStore(), FakeRetriever(), FakeAnswerGenerator())
    tools = {tool.name: tool for tool in list_client_tools(server).tools}

    assert set(tools) == {
        "list_indexed_documents",
        "search_college_documents",
        "ask_college_assistant",
        "get_document_context",
    }
    assert "query" in tools["search_college_documents"].inputSchema["properties"]
    assert "question" in tools["ask_college_assistant"].inputSchema["properties"]
    assert (
        tools["search_college_documents"].inputSchema["properties"]["limit"]["maximum"]
        == 10
    )
    assert tools["search_college_documents"].annotations.readOnlyHint is True


def test_mcp_uses_remote_ollama_and_access_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "https://ollama.example")
    monkeypatch.setenv("CF_ACCESS_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("CF_ACCESS_CLIENT_SECRET", "test-client-secret")
    calls = []
    store = FakeStore()
    service = RAGService(FakeRetriever(), FakeAnswerGenerator())

    def build_services(database_path, **kwargs):
        calls.append((database_path, kwargs))
        return store, object(), service

    monkeypatch.setattr(mcp_server, "build_rag_service", build_services)
    server = create_mcp_server(database_path=tmp_path / "chroma")

    result = invoke_tool(server, "list_indexed_documents")

    assert result["status"] == "empty"
    assert calls == [
        (
            tmp_path / "chroma",
            {
                "ollama_host": "https://ollama.example",
                "access_headers": {
                    "CF-Access-Client-Id": "test-client-id",
                    "CF-Access-Client-Secret": "test-client-secret",
                },
            },
        )
    ]


def test_lists_indexed_documents_and_handles_empty_library() -> None:
    chunk = RetrievedChunk("rules.pdf", 1, "Attendance is 75%.", 0.1)
    store, retriever, generator = (
        FakeStore([{"source": "rules.pdf", "document_id": "abc"}]),
        FakeRetriever([chunk]),
        FakeAnswerGenerator(),
    )
    server, builds = make_server(store, retriever, generator)

    result = invoke_tool(server, "list_indexed_documents")
    empty_server, _ = make_server(FakeStore(), FakeRetriever(), FakeAnswerGenerator())
    empty_result = invoke_tool(empty_server, "list_indexed_documents")

    assert result["status"] == "ok"
    assert result["documents"][0]["source"] == "rules.pdf"
    assert empty_result["status"] == "empty"
    assert empty_result["count"] == 0
    assert len(builds) == 1


def test_search_uses_existing_retriever_and_returns_attributed_chunks() -> None:
    chunk = RetrievedChunk("rules.pdf", 2, "Attendance is 75%.", 0.2)
    retriever = FakeRetriever([chunk])
    server, _ = make_server(FakeStore(), retriever, FakeAnswerGenerator())

    result = invoke_tool(
        server,
        "search_college_documents",
        {"query": "  What is attendance?  ", "limit": 5},
    )

    assert result["status"] == "ok"
    assert result["results"] == [
        {
            "source": "rules.pdf",
            "page_number": 2,
            "text": "Attendance is 75%.",
            "distance": 0.2,
        }
    ]
    assert retriever.calls == [("What is attendance?", 5)]


def test_question_answering_uses_existing_rag_pipeline() -> None:
    chunk = RetrievedChunk("rules.pdf", 2, "Attendance is 75%.", 0.2)
    generator = FakeAnswerGenerator("The minimum attendance is 75%.")
    server, _ = make_server(FakeStore(), FakeRetriever([chunk]), generator)

    result = invoke_tool(
        server,
        "ask_college_assistant",
        {"question": "What is the minimum attendance?"},
    )

    assert result["status"] == "ok"
    assert result["answer"] == "The minimum attendance is 75%."
    assert result["sources"][0]["source"] == "rules.pdf"


def test_empty_search_and_context_do_not_invent_document_content() -> None:
    server, _ = make_server(FakeStore(), FakeRetriever(), FakeAnswerGenerator())

    search_result = invoke_tool(
        server, "search_college_documents", {"query": "Unknown policy"}
    )
    context_result = invoke_tool(
        server, "get_document_context", {"query": "Unknown policy"}
    )
    answer_result = invoke_tool(
        server, "ask_college_assistant", {"question": "Unknown policy"}
    )

    assert search_result["status"] == "no_context"
    assert context_result["context"] == []
    assert answer_result["answer"] == UNKNOWN_ANSWER
    assert answer_result["sources"] == []


@pytest.mark.parametrize(
    "query",
    ["", "   ", "x" * 4001],
)
def test_search_rejects_invalid_query(query) -> None:
    server, _ = make_server(FakeStore(), FakeRetriever(), FakeAnswerGenerator())

    with pytest.raises(ToolError):
        invoke_registered_tool(server, "search_college_documents", {"query": query})


def test_search_rejects_out_of_range_result_limit() -> None:
    server, _ = make_server(FakeStore(), FakeRetriever(), FakeAnswerGenerator())

    with pytest.raises(ToolError):
        invoke_registered_tool(
            server,
            "search_college_documents",
            {"query": "policy", "limit": 11},
        )


def test_search_surfaces_embedding_and_storage_errors() -> None:
    embedding_server, _ = make_server(
        FakeStore(),
        FakeRetriever(error=EmbeddingServiceError("Ollama is offline.")),
        FakeAnswerGenerator(),
    )
    store_server, _ = make_server(
        FakeStore(),
        FakeRetriever(error=VectorStoreError("Chroma is unavailable.")),
        FakeAnswerGenerator(),
    )

    embedding_result = invoke_tool(
        embedding_server, "search_college_documents", {"query": "policy"}
    )
    store_result = invoke_tool(
        store_server, "search_college_documents", {"query": "policy"}
    )

    assert embedding_result == {"status": "error", "message": "Ollama is offline."}
    assert store_result == {"status": "error", "message": "Chroma is unavailable."}


def test_answer_tool_surfaces_ollama_connection_failure() -> None:
    chunk = RetrievedChunk("rules.pdf", 1, "Attendance is 75%.", 0.1)
    server, _ = make_server(
        FakeStore(),
        FakeRetriever([chunk]),
        FakeAnswerGenerator(error=LLMServiceError("Ollama is offline.")),
    )

    result = invoke_tool(
        server, "ask_college_assistant", {"question": "What is attendance?"}
    )

    assert result == {"status": "error", "message": "Ollama is offline."}


def test_document_list_surfaces_chroma_errors() -> None:
    server, _ = make_server(
        FakeStore(error=VectorStoreError("Chroma is unavailable.")),
        FakeRetriever(),
        FakeAnswerGenerator(),
    )

    assert invoke_tool(server, "list_indexed_documents") == {
        "status": "error",
        "message": "Chroma is unavailable.",
    }
