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


def test_ollama_status_reports_unavailable_service(monkeypatch) -> None:
    def fail_to_connect():
        raise ConnectionError("offline")

    monkeypatch.setattr(app_services.ollama, "Client", fail_to_connect)

    connected, models, message = app_services.get_ollama_status()

    assert connected is False
    assert models == set()
    assert "ollama serve" in message


def test_model_availability_accepts_latest_tag() -> None:
    assert app_services.model_is_available("nomic-embed-text", {"nomic-embed-text:latest"})
    assert not app_services.model_is_available("llama3.2:3b", {"llama3.1:8b"})
