"""response(): one-off inference and structured (schema-enforced) outputs."""

import json

import msgspec
import pytest
from msgspec import Struct

from sthai.client import Client
from sthai.const import INFERENCE_ENDPOINT
from sthai.structs.completions import InferenceResponse

from conftest import MockBackend, load_fixture


class CityInfo(Struct):
    """Mirrors the schema captured in the response_struct fixture."""

    name: str
    country: str
    population: int


class StrictCity(Struct):
    """Superset schema the captured payload cannot satisfy."""

    name: str
    country: str
    population: int
    mayor: str


def respond_with_content(backend: MockBackend, content: str | None) -> None:
    """Serve the response_struct fixture with its message content replaced."""
    fixture = load_fixture("response_struct")
    fixture["response"]["choices"][0]["message"]["content"] = content
    backend.respond("POST", INFERENCE_ENDPOINT, fixture["response"])


def test_plain_response_returns_inference_response(
    backend: MockBackend, client: Client
) -> None:
    backend.register("chat_simple")
    result = client.response("hello")
    assert isinstance(result, InferenceResponse)
    assert "response_format" not in backend.last_call.body
    assert client.last_response() is result


def test_response_does_not_touch_history(backend: MockBackend, client: Client) -> None:
    backend.register("chat_simple")
    client.chat("first")
    client.response("one-off")
    assert [m["role"] for m in backend.last_call.body["messages"]] == ["user"]
    client.chat("second")
    messages = backend.last_call.body["messages"]
    # Only the chat() turns are in history; the one-off never joined it
    assert [m["content"] for m in messages if m["role"] == "user"] == [
        "first",
        "second",
    ]


def test_struct_response_parsed_and_schema_sent(
    backend: MockBackend, client: Client
) -> None:
    fixture = backend.register("response_struct")
    result = client.response(
        "Give me basic facts about Wellington, New Zealand.",
        response_type=CityInfo,
    )
    assert isinstance(result, CityInfo)
    content = fixture["response"]["choices"][0]["message"]["content"]
    assert result == msgspec.json.decode(content.encode(), type=CityInfo)

    response_format = backend.last_call.body["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "CityInfo"
    # The generated schema itself goes over the wire as "schema"
    assert "schema" in response_format["json_schema"]


def test_dict_response_uses_json_object_mode(
    backend: MockBackend, client: Client
) -> None:
    fixture = backend.register("response_json_object")
    result = client.response("Return a JSON object.", response_type=dict)
    content = fixture["response"]["choices"][0]["message"]["content"]
    assert result == json.loads(content)
    response_format = backend.last_call.body["response_format"]
    assert response_format == {"type": "json_object"}


def test_truncated_response_raises_helpful_error(
    backend: MockBackend, client: Client
) -> None:
    backend.register("response_truncated")
    with pytest.raises(ValueError, match="token limit"):
        client.response("facts please", response_type=CityInfo, max_tokens=10)


def test_prose_response_raises_not_valid_json(
    backend: MockBackend, client: Client
) -> None:
    # The server occasionally skips the schema with thinking enabled and
    # returns prose; parse() must name that failure rather than crash
    respond_with_content(backend, "Sure! Wellington is the capital of New Zealand.")
    with pytest.raises(ValueError, match="not valid JSON"):
        client.response("facts please", response_type=CityInfo)


def test_wrong_shape_raises_validation_error(
    backend: MockBackend, client: Client
) -> None:
    with pytest.raises(msgspec.ValidationError, match="mayor"):
        backend.register("response_struct")
        client.response("facts please", response_type=StrictCity)


def test_missing_content_raises(backend: MockBackend, client: Client) -> None:
    respond_with_content(backend, None)
    with pytest.raises(ValueError, match="no text content"):
        client.response("facts please", response_type=CityInfo)
