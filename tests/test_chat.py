"""chat(): responses, request wiring, and conversation history semantics."""

import pytest

from sthai.client import Client
from sthai.exceptions import ClientError, InputError
from sthai.structs.common import Usage
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


def test_history_getter_returns_turns(backend: MockBackend, client: Client) -> None:
    fixture = backend.register("chat_simple")
    client.chat("hello")
    assert client.history() == [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": fixture["response"]["choices"][0]["message"]["content"],
        },
    ]


def test_set_history_restores_conversation(
    backend: MockBackend, client: Client
) -> None:
    backend.register("chat_simple")
    restored = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "a reply"},
    ]
    client.set_history(restored)
    client.chat("second")
    messages = backend.last_call.body["messages"]
    assert messages == [*restored, {"role": "user", "content": "second"}]


def test_set_history_rejects_missing_content(client: Client) -> None:
    with pytest.raises(InputError, match="content"):
        client.set_history([{"role": "user"}])


def test_set_write_history_stops_recording(
    backend: MockBackend, client: Client
) -> None:
    backend.register("chat_simple")
    assert client.write_history() is True
    client.chat("first")
    client.set_write_history(False)
    assert client.write_history() is False
    client.chat("second")
    # The stored history was still sent, but the second exchange not recorded
    assert [m["role"] for m in backend.last_call.body["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert [m["content"] for m in client.history() if m["role"] == "user"] == ["first"]


def test_history_usage_accumulates(backend: MockBackend, client: Client) -> None:
    fixture = backend.register("chat_simple")
    client.chat("first")
    client.chat("second")
    usage = client.history_usage()
    wire_usage = fixture["response"]["usage"]
    assert usage.input_tokens == 2 * wire_usage["prompt_tokens"]
    assert usage.output_tokens == 2 * wire_usage["completion_tokens"]
    assert usage.cached_tokens == 0


def test_history_usage_excludes_unhistoried_calls(
    backend: MockBackend, client: Client
) -> None:
    fixture = backend.register("chat_simple")
    client.chat("first")
    client.chat("standalone", use_history=False)
    usage = client.history_usage()
    wire_usage = fixture["response"]["usage"]
    assert usage.input_tokens == wire_usage["prompt_tokens"]
    assert usage.output_tokens == wire_usage["completion_tokens"]


def test_history_usage_resets(backend: MockBackend, client: Client) -> None:
    backend.register("chat_simple")
    client.chat("first")
    assert client.history_usage().input_tokens > 0
    client.clear_history()
    assert client.history_usage() == Usage()
    client.chat("second")
    assert client.history_usage().input_tokens > 0
    client.set_history([{"role": "user", "content": "restored"}])
    assert client.history_usage() == Usage()
