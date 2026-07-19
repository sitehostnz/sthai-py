"""The exception surface: status mapping, envelope parsing, and wrapping."""

from typing import Any, NoReturn

import niquests.exceptions
import pytest

from sthai.async_client import AsyncClient
from sthai.client import Client
from sthai.const import INFERENCE_ENDPOINT, MODELS_ENDPOINT
from sthai.exceptions import APIError, ClientError, ResponseError, TransportError

from conftest import MockBackend

ERROR_BODY = {"error": {"message": "model not found", "type": "invalid_request_error"}}


def test_4xx_parses_error_envelope(backend: MockBackend, client: Client) -> None:
    fixture = backend.register("error_bad_model")
    with pytest.raises(ClientError) as exc_info:
        client.chat("hello", model="does-not-exist", use_history=False)
    error = exc_info.value
    assert error.status_code == fixture["status_code"] == 404
    assert error.server_message == "model not found"
    assert error.error_type == "invalid_request_error"
    assert str(error) == "404: model not found (invalid_request_error)"
    assert error.response.status_code == 404


def test_5xx_raises_api_error(backend: MockBackend, client: Client) -> None:
    backend.respond("POST", INFERENCE_ENDPOINT, ERROR_BODY, status_code=503)
    with pytest.raises(APIError) as exc_info:
        client.chat("hello", use_history=False)
    assert exc_info.value.status_code == 503
    assert exc_info.value.server_message == "model not found"


def test_unparseable_error_body_falls_back(
    backend: MockBackend, client: Client
) -> None:
    backend.respond(
        "POST", INFERENCE_ENDPOINT, "<html>bad gateway</html>", status_code=502
    )
    with pytest.raises(APIError) as exc_info:
        client.chat("hello", use_history=False)
    error = exc_info.value
    assert error.status_code == 502
    assert error.server_message is None
    assert error.error_type is None
    assert str(error) == "502: HTTP error"


def test_malformed_success_body_raises_response_error(
    backend: MockBackend, client: Client
) -> None:
    # Valid JSON of the wrong shape must not leak a raw msgspec exception
    backend.respond("GET", MODELS_ENDPOINT, {"data": ["not-a-model-card"]})
    with pytest.raises(ResponseError, match="ModelList"):
        client.models()


def test_sync_transport_error_wrapped(
    monkeypatch: pytest.MonkeyPatch, client: Client
) -> None:
    cause = niquests.exceptions.ConnectionError("boom")

    def raise_connection_error(*args: Any, **kwargs: Any) -> NoReturn:
        raise cause

    monkeypatch.setattr("sthai.client.request", raise_connection_error)
    with pytest.raises(TransportError, match="request failed") as exc_info:
        client.models()
    assert exc_info.value.__cause__ is cause


async def test_async_transport_error_wrapped(
    monkeypatch: pytest.MonkeyPatch, async_client: AsyncClient
) -> None:
    cause = niquests.exceptions.ConnectionError("boom")

    async def raise_connection_error(*args: Any, **kwargs: Any) -> NoReturn:
        raise cause

    monkeypatch.setattr("sthai.async_client.arequest", raise_connection_error)
    with pytest.raises(TransportError, match="request failed") as exc_info:
        await async_client.models()
    assert exc_info.value.__cause__ is cause
