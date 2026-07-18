"""Client construction and transport: URLs, headers, health, models, errors."""

import pytest
from niquests.exceptions import HTTPError

from sthai.client import Client
from sthai.const import SESSION_PIN_HEADER
from sthai.structs.models import ModelCard

from conftest import MockBackend


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="api_key"):
        Client(api_key="")


def test_https_url_by_default(backend: MockBackend, client: Client) -> None:
    backend.register("health")
    client.healthy()
    assert backend.last_call.url == "https://ai.sitehost.nz/health"


def test_insecure_url(backend: MockBackend) -> None:
    backend.register("health")
    insecure = Client(api_key="test-key", secure=False)
    insecure.healthy()
    assert backend.last_call.url == "http://ai.sitehost.nz/health"


def test_endpoint_path_normalization(client: Client) -> None:
    assert client._endpoint_url("health") == client._endpoint_url("/health")


def test_authorization_header_sent(backend: MockBackend, client: Client) -> None:
    backend.register("health")
    client.healthy()
    assert backend.last_call.headers["Authorization"] == "Bearer test-key"


def test_no_session_pin_by_default(backend: MockBackend, client: Client) -> None:
    backend.register("health")
    client.healthy()
    assert SESSION_PIN_HEADER not in backend.last_call.headers


def test_explicit_session_pin_header(backend: MockBackend) -> None:
    backend.register("health")
    pinned = Client(api_key="test-key", session_pin="my-pin")
    pinned.healthy()
    assert backend.last_call.headers[SESSION_PIN_HEADER] == "my-pin"


def test_auto_session_generates_pin(backend: MockBackend) -> None:
    backend.register("health")
    pinned = Client(api_key="test-key", auto_session=True)
    pinned.healthy()
    pin = backend.last_call.headers[SESSION_PIN_HEADER]
    assert len(pin) == 48
    int(pin, 16)  # hex-encoded token


def test_healthy_true_on_200(backend: MockBackend, client: Client) -> None:
    backend.register("health")
    assert client.healthy() is True


def test_healthy_false_on_500(backend: MockBackend, client: Client) -> None:
    backend.respond("GET", "/health", {"status": "down"}, status_code=500)
    assert client.healthy() is False


def test_models_returns_model_cards(backend: MockBackend, client: Client) -> None:
    fixture = backend.register("models")
    models = client.models()
    assert all(isinstance(card, ModelCard) for card in models)
    assert [card.id for card in models] == [
        entry["id"] for entry in fixture["response"]["data"]
    ]


def test_http_error_raised_on_bad_model(backend: MockBackend, client: Client) -> None:
    backend.register("error_bad_model")
    with pytest.raises(HTTPError):
        client.chat("hello", model="does-not-exist", use_history=False)
