from types import SimpleNamespace

import pytest

import app_services


CF_CLIENT_ID = "test-access-client-id"
CF_CLIENT_SECRET = "test-access-client-secret"
REMOTE_OLLAMA_HOST = "https://ollama.example"


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


def test_remote_ollama_client_sends_both_cloudflare_access_headers(monkeypatch) -> None:
    monkeypatch.delenv("CF_ACCESS_CLIENT_ID", raising=False)
    monkeypatch.delenv("CF_ACCESS_CLIENT_SECRET", raising=False)
    calls = []
    monkeypatch.setattr(
        app_services.ollama,
        "Client",
        lambda **kwargs: calls.append(kwargs) or object(),
    )
    access_headers = app_services.resolve_ollama_access_headers(
        REMOTE_OLLAMA_HOST,
        {
            "CF_ACCESS_CLIENT_ID": CF_CLIENT_ID,
            "CF_ACCESS_CLIENT_SECRET": CF_CLIENT_SECRET,
        },
    )

    app_services.create_ollama_client(
        REMOTE_OLLAMA_HOST,
        access_headers=access_headers,
    )

    assert calls == [
        {
            "host": REMOTE_OLLAMA_HOST,
            "headers": {
                "CF-Access-Client-Id": CF_CLIENT_ID,
                "CF-Access-Client-Secret": CF_CLIENT_SECRET,
            },
        }
    ]


def test_remote_ollama_client_without_access_secrets_keeps_existing_behavior(
    monkeypatch,
) -> None:
    monkeypatch.delenv("CF_ACCESS_CLIENT_ID", raising=False)
    monkeypatch.delenv("CF_ACCESS_CLIENT_SECRET", raising=False)
    calls = []
    monkeypatch.setattr(
        app_services.ollama,
        "Client",
        lambda **kwargs: calls.append(kwargs) or object(),
    )

    access_headers = app_services.resolve_ollama_access_headers(
        REMOTE_OLLAMA_HOST,
        {},
    )
    app_services.create_ollama_client(
        REMOTE_OLLAMA_HOST,
        access_headers=access_headers,
    )
    app_services.create_ollama_client()

    assert access_headers is None
    assert calls == [{"host": REMOTE_OLLAMA_HOST}, {}]


@pytest.mark.parametrize(
    "secrets",
    [
        {"CF_ACCESS_CLIENT_ID": CF_CLIENT_ID},
        {"CF_ACCESS_CLIENT_SECRET": CF_CLIENT_SECRET},
    ],
)
def test_remote_ollama_rejects_incomplete_access_credentials(secrets) -> None:
    with pytest.raises(
        app_services.OllamaAccessConfigurationError,
        match="Both Cloudflare Access client credentials",
    ) as error:
        app_services.resolve_ollama_access_headers(REMOTE_OLLAMA_HOST, secrets)

    assert CF_CLIENT_ID not in str(error.value)
    assert CF_CLIENT_SECRET not in str(error.value)


def test_cloudflare_access_credentials_require_https() -> None:
    with pytest.raises(
        app_services.OllamaAccessConfigurationError,
        match="require an HTTPS Ollama endpoint",
    ):
        app_services.resolve_ollama_access_headers(
            "http://ollama.example",
            {
                "CF_ACCESS_CLIENT_ID": CF_CLIENT_ID,
                "CF_ACCESS_CLIENT_SECRET": CF_CLIENT_SECRET,
            },
        )


def test_access_credentials_can_be_read_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("CF_ACCESS_CLIENT_ID", CF_CLIENT_ID)
    monkeypatch.setenv("CF_ACCESS_CLIENT_SECRET", CF_CLIENT_SECRET)

    assert app_services.resolve_ollama_access_headers(
        REMOTE_OLLAMA_HOST, {}
    ) == {
        "CF-Access-Client-Id": CF_CLIENT_ID,
        "CF-Access-Client-Secret": CF_CLIENT_SECRET,
    }


def test_loopback_ollama_does_not_receive_cloudflare_access_headers(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        app_services.ollama,
        "Client",
        lambda **kwargs: calls.append(kwargs) or object(),
    )
    access_headers = {
        "CF-Access-Client-Id": CF_CLIENT_ID,
        "CF-Access-Client-Secret": CF_CLIENT_SECRET,
    }

    app_services.create_ollama_client(
        "http://127.0.0.1:11434",
        access_headers=access_headers,
    )

    assert calls == [{"host": "http://127.0.0.1:11434"}]


def test_ollama_status_does_not_log_or_return_access_credentials(
    monkeypatch, caplog
) -> None:
    def fail_to_connect(**kwargs):
        raise ConnectionError(
            f"request failed with {CF_CLIENT_ID} and {CF_CLIENT_SECRET}"
        )

    monkeypatch.setattr(app_services.ollama, "Client", fail_to_connect)
    headers = {
        "CF-Access-Client-Id": CF_CLIENT_ID,
        "CF-Access-Client-Secret": CF_CLIENT_SECRET,
    }

    connected, _, message = app_services.get_ollama_status(
        REMOTE_OLLAMA_HOST,
        access_headers=headers,
    )

    assert connected is False
    assert CF_CLIENT_ID not in message
    assert CF_CLIENT_SECRET not in message
    assert CF_CLIENT_ID not in caplog.text
    assert CF_CLIENT_SECRET not in caplog.text


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
    client_options = []

    def create_client(**kwargs):
        client_options.append(kwargs)
        return client

    monkeypatch.setattr(app_services.ollama, "Client", create_client)
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
        access_headers={
            "CF-Access-Client-Id": CF_CLIENT_ID,
            "CF-Access-Client-Secret": CF_CLIENT_SECRET,
        },
    )

    assert received_clients == [client, client]
    assert client_options == [
        {
            "host": "https://ollama.example",
            "headers": {
                "CF-Access-Client-Id": CF_CLIENT_ID,
                "CF-Access-Client-Secret": CF_CLIENT_SECRET,
            },
        }
    ]
    assert service.answer_generator is not None
