import pytest

from embeddings import EMBEDDING_MODEL, EmbeddingServiceError, OllamaEmbedder


class FakeEmbeddingClient:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response or {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        self.error = error
        self.call = None

    def embed(self, **kwargs):
        self.call = kwargs
        if self.error:
            raise self.error
        return self.response


def test_embedder_uses_required_model_and_returns_vectors() -> None:
    client = FakeEmbeddingClient()
    embedder = OllamaEmbedder(client=client)

    vectors = embedder.embed_texts(["one", "two"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert client.call == {"model": EMBEDDING_MODEL, "input": ["one", "two"]}


def test_embedder_handles_empty_input_without_calling_ollama() -> None:
    client = FakeEmbeddingClient()

    assert OllamaEmbedder(client=client).embed_texts([]) == []
    assert client.call is None


def test_embedder_surfaces_service_errors_without_logging_credentials(caplog) -> None:
    secret = "test-access-client-secret"
    client = FakeEmbeddingClient(error=ConnectionError(f"offline: {secret}"))

    with pytest.raises(
        EmbeddingServiceError, match="Check that Ollama is running"
    ) as error:
        OllamaEmbedder(client=client).embed_texts(["question"])

    assert secret not in str(error.value)
    assert secret not in caplog.text


def test_embedder_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="empty text"):
        OllamaEmbedder(client=FakeEmbeddingClient()).embed_texts([""])
