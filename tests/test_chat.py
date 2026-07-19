"""chat(): responses, request wiring, and conversation history semantics."""

import pytest

from sthai.client import Client
from sthai.exceptions import ClientError
from sthai.structs.completions import InferenceResponse

from conftest import MockBackend


def test_returns_inference_response(backend: MockBackend, client: Client) -> None:
    fixture = backend.register("chat_simple")
    response = client.chat("Reply with exactly: kia ora")
    assert isinstance(response, InferenceResponse)
    message = fixture["response"]["choices"][0]["message"]
    assert response.output().text == message["content"]


def test_usage_summary_matches_fixture(backend: MockBackend, client: Client) -> None:
    fixture = backend.register("chat_simple")
    usage = client.chat("hello").usage()
    wire_usage = fixture["response"]["usage"]
    assert usage.input_tokens == wire_usage["prompt_tokens"]
    assert usage.output_tokens == wire_usage["completion_tokens"]


def test_thinking_response_has_reasoning(backend: MockBackend, client: Client) -> None:
    backend.register("chat_thinking")
    response = client.chat("What is 17 + 25?", use_thinking=True)
    assert response.output().reasoning
    assert client.last_reasoning() == response.output().reasoning


def test_last_response_returns_decoded(backend: MockBackend, client: Client) -> None:
    backend.register("chat_simple")
    assert client.last_response() is None
    response = client.chat("hello")
    assert client.last_response() is response


def test_minimal_request_body(backend: MockBackend, client: Client) -> None:
    backend.register("chat_simple")
    client.chat("hello")
    body = backend.last_call.body
    # UNSET optionals must be omitted entirely, not sent as null
    assert set(body) == {"messages", "model", "chat_template_kwargs"}
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_optional_params_on_the_wire(backend: MockBackend, client: Client) -> None:
    backend.register("chat_simple")
    client.chat("hello", max_tokens=50, temperature=0.2, use_thinking=True)
    body = backend.last_call.body
    # max_tokens maps to the non-deprecated max_completion_tokens field
    assert body["max_completion_tokens"] == 50
    assert "max_tokens" not in body
    assert body["temperature"] == 0.2
    assert body["chat_template_kwargs"] == {"enable_thinking": True}


def test_history_accumulates(backend: MockBackend, client: Client) -> None:
    fixture = backend.register("chat_simple")
    client.chat("first")
    client.chat("second")
    messages = backend.last_call.body["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == "first"
    assert (
        messages[1]["content"]
        == fixture["response"]["choices"][0]["message"]["content"]
    )
    assert messages[2]["content"] == "second"


def test_use_history_false_neither_sends_nor_records(
    backend: MockBackend, client: Client
) -> None:
    backend.register("chat_simple")
    client.chat("first")
    client.chat("standalone", use_history=False)
    assert [m["role"] for m in backend.last_call.body["messages"]] == ["user"]
    client.chat("third")
    # The standalone turn must not have leaked into the stored history
    messages = backend.last_call.body["messages"]
    assert [m["content"] for m in messages if m["role"] == "user"] == [
        "first",
        "third",
    ]


def test_write_history_false_never_records(backend: MockBackend) -> None:
    backend.register("chat_simple")
    ephemeral = Client(api_key="test-key", write_history=False)
    ephemeral.chat("first")
    ephemeral.chat("second")
    assert [m["role"] for m in backend.last_call.body["messages"]] == ["user"]


def test_system_prompt_prepended_not_stored(
    backend: MockBackend, client: Client
) -> None:
    backend.register("chat_simple")
    client.chat("first", system_prompt="Be terse.")
    messages = backend.last_call.body["messages"]
    assert messages[0] == {"role": "system", "content": "Be terse."}
    client.chat("second")
    # No system prompt this call, so none appears - it is per-call only
    assert [m["role"] for m in backend.last_call.body["messages"]] == [
        "user",
        "assistant",
        "user",
    ]


def test_clear_history(backend: MockBackend, client: Client) -> None:
    backend.register("chat_simple")
    client.chat("first")
    client.clear_history()
    client.chat("second")
    assert [m["role"] for m in backend.last_call.body["messages"]] == ["user"]


def test_failed_call_does_not_write_history(
    backend: MockBackend, client: Client
) -> None:
    backend.register("error_bad_model")
    with pytest.raises(ClientError):
        client.chat("doomed")
    backend.register("chat_simple")
    client.chat("second")
    assert [m["role"] for m in backend.last_call.body["messages"]] == ["user"]
