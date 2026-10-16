from types import SimpleNamespace

import app_services


def test_ollama_status_reports_connection_and_available_models(monkeypatch) -> None:
    monkeypatch.setattr(
        app_services.ollama,
        "Client",
        lambda: SimpleNamespace(
            list=lambda: SimpleNamespace(
                models=[SimpleNamespace(model="llama3.2:3b")]
            )
        ),
    )

    connected, models, message = app_services.get_ollama_status()

    assert connected is True
    assert models == {"llama3.2:3b"}
    assert message == ""


def test_ollama_host_prefers_environment_over_streamlit_secret(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://env-ollama:11434")

    assert app_services.resolve_ollama_host(
        {"OLLAMA_HOST": "https://secret-ollama.example"}
    ) == "http://env-ollama:11434"


def test_ollama_host_uses_streamlit_secret_when_environment_is_unset(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    assert app_services.resolve_ollama_host(
        {"OLLAMA_HOST": "https://secret-ollama.example"}
    ) == "https://secret-ollama.example"


def test_ollama_host_defaults_to_local_client_when_unconfigured(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    assert app_services.resolve_ollama_host() is None


def test_ollama_status_uses_configured_endpoint(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        app_services.ollama,
        "Client",
        lambda **kwargs: (
            calls.append(kwargs)
            or SimpleNamespace(list=lambda: {"models": []})
        ),
    )

    connected, models, message = app_services.get_ollama_status(
        ollama_host="https://ollama.example"
    )

    assert connected is True
    assert models == set()
    assert message == ""
    assert calls == [{"host": "https://ollama.example"}]


def test_ollama_status_reports_unavailable_service(monkeypatch) -> None:
    def fail_to_connect():
        raise ConnectionError("offline")

    monkeypatch.setattr(app_services.ollama, "Client", fail_to_connect)

    connected, models, message = app_services.get_ollama_status()

    assert connected is False
    assert models == set()
    assert "OLLAMA_HOST" in message


def test_model_availability_accepts_latest_tag() -> None:
    assert app_services.model_is_available("nomic-embed-text", {"nomic-embed-text:latest"})
    assert not app_services.model_is_available("llama3.2:3b", {"llama3.1:8b"})


def test_build_rag_service_uses_same_configured_ollama_client(monkeypatch, tmp_path) -> None:
    client = object()
    received_clients = []
    monkeypatch.setattr(app_services.ollama, "Client", lambda **kwargs: client)
    monkeypatch.setattr(app_services, "ChromaVectorStore", lambda **kwargs: object())

    class Embedder:
        def __init__(self, client):
            received_clients.append(client)

    class AnswerGenerator:
        def __init__(self, client):
            received_clients.append(client)

    monkeypatch.setattr(app_services, "OllamaEmbedder", Embedder)
    monkeypatch.setattr(app_services, "OllamaAnswerGenerator", AnswerGenerator)

    _, _, service = app_services.build_rag_service(
        tmp_path / "chroma",
        ollama_host="https://ollama.example",
    )

    assert received_clients == [client, client]
    assert service.answer_generator is not None
