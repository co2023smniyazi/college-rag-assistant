import pytest

from llm_service import LLMServiceError, OllamaAnswerGenerator, UNKNOWN_ANSWER
from models import RetrievedChunk
from rag_service import RAGService
from retriever import Retriever


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.question = None
        self.limit = None

    def retrieve(self, question, limit=3):
        self.question = question
        self.limit = limit
        return self.chunks


class FakeAnswerGenerator:
    def __init__(self, answer="The requirement is 75%."):
        self.answer = answer
        self.received_chunks = None

    def generate(self, question, chunks):
        self.received_chunks = chunks
        return self.answer


def test_rag_service_passes_retrieved_context_and_returns_sources() -> None:
    chunk = RetrievedChunk("rules.pdf", 2, "Attendance is 75%.", 0.1)
    retriever = FakeRetriever([chunk])
    generator = FakeAnswerGenerator()
    service = RAGService(retriever, generator)

    answer, sources = service.ask("  What attendance?  ")

    assert answer == "The requirement is 75%."
    assert sources == [chunk]
    assert retriever.question == "What attendance?"
    assert generator.received_chunks == [chunk]


def test_rag_service_passes_configured_retrieval_limit() -> None:
    class LimitedRetriever(FakeRetriever):
        def retrieve(self, question, limit):
            self.limit = limit
            return self.chunks

    retriever = LimitedRetriever([])
    service = RAGService(retriever, FakeAnswerGenerator())

    service.ask("question", limit=7)

    assert retriever.limit == 7


def test_rag_service_refuses_without_relevant_context() -> None:
    retriever = FakeRetriever([])
    generator = FakeAnswerGenerator()
    service = RAGService(retriever, generator)

    answer, sources = service.ask("What is the hostel fee?")

    assert answer == UNKNOWN_ANSWER
    assert sources == []
    assert generator.received_chunks is None


def test_rag_service_rejects_empty_question() -> None:
    service = RAGService(FakeRetriever([]), FakeAnswerGenerator())

    with pytest.raises(ValueError, match="Enter a question"):
        service.ask(" ")


def test_retriever_filters_low_relevance_context() -> None:
    class FakeEmbedder:
        def embed_texts(self, texts):
            return [[1.0, 0.0]]

    class FakeStore:
        def is_empty(self):
            return False

        def search(self, embedding, limit):
            return [
                RetrievedChunk("relevant.pdf", 1, "supported", 0.2),
                RetrievedChunk("unrelated.pdf", 7, "unrelated", 0.95),
            ]

    chunks = Retriever(FakeEmbedder(), FakeStore()).retrieve("question")

    assert [chunk.source for chunk in chunks] == ["relevant.pdf"]


def test_retriever_returns_empty_without_calling_embedder_for_empty_store() -> None:
    class FakeEmbedder:
        def embed_texts(self, texts):
            raise AssertionError("There is nothing to embed when no documents exist.")

    class EmptyStore:
        def is_empty(self):
            return True

    assert Retriever(FakeEmbedder(), EmptyStore()).retrieve("question") == []


class FakeChatClient:
    def __init__(self, response=None, error=None):
        self.response = response or {
            "message": {"content": "Attendance must be at least 75%."}
        }
        self.error = error
        self.call = None

    def chat(self, **kwargs):
        self.call = kwargs
        if self.error:
            raise self.error
        return self.response


def test_answer_generator_uses_context_and_required_local_model() -> None:
    from llm_service import CHAT_MODEL, SYSTEM_INSTRUCTION

    chunk = RetrievedChunk("rules.pdf", 3, "Attendance is 75%.", 0.1)
    client = FakeChatClient()

    answer = OllamaAnswerGenerator(client=client).generate("What attendance?", [chunk])

    assert answer == "Attendance must be at least 75%."
    assert client.call["model"] == CHAT_MODEL
    assert client.call["messages"][0]["content"] == SYSTEM_INSTRUCTION
    assert "[Source: rules.pdf, page 3]" in client.call["messages"][1]["content"]
    assert "Attendance is 75%." in client.call["messages"][1]["content"]


def test_answer_generator_surfaces_ollama_errors() -> None:
    with pytest.raises(LLMServiceError, match="Check that Ollama is running"):
        OllamaAnswerGenerator(client=FakeChatClient(error=ConnectionError())).generate(
            "question", [RetrievedChunk("rules.pdf", 1, "text", 0.1)]
        )
