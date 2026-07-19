"""
AsyncClient: the async dispatch path for every API method.

The request-building and validation logic is shared with the sync Client
through _BaseClient and exhaustively covered by the sync tests; these tests
cover what is genuinely async-path code - the awaited transport, decoding,
and the stateful history/last-response plumbing around it.
"""

import msgspec
import pytest
from msgspec import Struct

from sthai.async_client import AsyncClient
from sthai.const import SESSION_PIN_HEADER
from sthai.exceptions import ClientError, InputError
from sthai.structs.completions import InferenceResponse

from conftest import MockBackend


class CityInfo(Struct):
    """Mirrors the schema captured in the response_struct fixture."""

    name: str
    country: str
    population: int


async def test_healthy(backend: MockBackend, async_client: AsyncClient) -> None:
    backend.register("health")
    assert await async_client.healthy() is True


async def test_models(backend: MockBackend, async_client: AsyncClient) -> None:
    fixture = backend.register("models")
    models = await async_client.models()
    assert [card.id for card in models] == [
        entry["id"] for entry in fixture["response"]["data"]
    ]


async def test_chat_returns_inference_response(
    backend: MockBackend, async_client: AsyncClient
) -> None:
    fixture = backend.register("chat_simple")
    response = await async_client.chat("Reply with exactly: kia ora")
    assert isinstance(response, InferenceResponse)
    message = fixture["response"]["choices"][0]["message"]
    assert response.output().text == message["content"]


async def test_chat_request_wiring(
    backend: MockBackend, async_client: AsyncClient
) -> None:
    fixture = backend.register("chat_simple")
    await async_client.chat("hello")
    call = backend.last_call
    assert call.path == fixture["endpoint"]
    assert call.headers["Authorization"] == "Bearer test-key"
    assert call.body["messages"] == [{"role": "user", "content": "hello"}]


async def test_history_accumulates(
    backend: MockBackend, async_client: AsyncClient
) -> None:
    fixture = backend.register("chat_simple")
    await async_client.chat("first")
    await async_client.chat("second")
    messages = backend.last_call.body["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == "first"
    assert (
        messages[1]["content"]
        == fixture["response"]["choices"][0]["message"]["content"]
    )
    assert messages[2]["content"] == "second"


async def test_failed_call_raises_and_does_not_write_history(
    backend: MockBackend, async_client: AsyncClient
) -> None:
    backend.register("error_bad_model")
    with pytest.raises(ClientError):
        await async_client.chat("doomed")
    backend.register("chat_simple")
    await async_client.chat("second")
    assert [m["role"] for m in backend.last_call.body["messages"]] == ["user"]


async def test_last_response_and_reasoning(
    backend: MockBackend, async_client: AsyncClient
) -> None:
    backend.register("chat_thinking")
    assert async_client.last_response() is None
    response = await async_client.chat("What is 17 + 25?", use_thinking=True)
    assert async_client.last_response() is response
    assert async_client.last_reasoning() == response.output().reasoning


async def test_response_plain(backend: MockBackend, async_client: AsyncClient) -> None:
    backend.register("chat_simple")
    result = await async_client.response("hello")
    assert isinstance(result, InferenceResponse)
    assert "response_format" not in backend.last_call.body
    # A one-off response never joins the stored history
    await async_client.chat("second")
    assert [m["role"] for m in backend.last_call.body["messages"]] == ["user"]


async def test_response_struct_parsed_and_schema_sent(
    backend: MockBackend, async_client: AsyncClient
) -> None:
    fixture = backend.register("response_struct")
    result = await async_client.response(
        "Give me basic facts about Wellington, New Zealand.",
        response_type=CityInfo,
    )
    assert isinstance(result, CityInfo)
    content = fixture["response"]["choices"][0]["message"]["content"]
    assert result == msgspec.json.decode(content.encode(), type=CityInfo)
    assert backend.last_call.body["response_format"]["type"] == "json_schema"


async def test_embed(backend: MockBackend, async_client: AsyncClient) -> None:
    fixture = backend.register("embed_single")
    vector = await async_client.embed("kia ora")
    assert vector == fixture["response"]["data"][0]["embedding"]
    assert backend.last_call.path == fixture["endpoint"]


async def test_embed_validation_raises_before_any_request(
    backend: MockBackend, async_client: AsyncClient
) -> None:
    with pytest.raises(InputError, match="requires text and/or images"):
        await async_client.embed("")
    assert backend.calls == []


async def test_batch_embed(backend: MockBackend, async_client: AsyncClient) -> None:
    fixture = backend.register("batch_embed")
    vectors = await async_client.batch_embed(["first text", "second text"])
    assert vectors == [entry["embedding"] for entry in fixture["response"]["data"]]


async def test_rerank(backend: MockBackend, async_client: AsyncClient) -> None:
    fixture = backend.register("rerank")
    query = fixture["request"]["query"]
    documents = fixture["request"]["documents"]
    results = await async_client.rerank(query, documents)
    scores = [result.relevance_score for result in results]
    assert scores == sorted(scores, reverse=True)
    assert backend.last_call.body["query"] == query


async def test_auto_session_pin_header(backend: MockBackend) -> None:
    backend.register("health")
    pinned = AsyncClient(api_key="test-key", auto_session=True)
    await pinned.healthy()
    pin = backend.last_call.headers[SESSION_PIN_HEADER]
    assert pin
    # The generated pin is stable across calls on the same client
    await pinned.healthy()
    assert backend.last_call.headers[SESSION_PIN_HEADER] == pin
