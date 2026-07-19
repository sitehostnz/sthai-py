"""embed(): single-input embeddings, instruction steering, and guards."""

import pytest

from sthai.client import Client
from sthai.const import (
    EMBEDDING_DOCUMENT_INSTRUCTION,
    EMBEDDING_ENDPOINT,
    EMBEDDING_QUERY_INSTRUCTION,
)
from sthai.exceptions import InputError, ResponseError

from conftest import MockBackend, load_fixture

# image_content only sniffs magic bytes, so a stub PNG is enough client-side
FAKE_PNG = b"\x89PNG\r\n\x1a\nnot-a-real-image"


def test_returns_float_vector(backend: MockBackend, client: Client) -> None:
    fixture = backend.register("embed_single")
    vector = client.embed("The Beehive.", dimensions=32)
    assert vector == fixture["response"]["data"][0]["embedding"]
    assert len(vector) == 32
    assert all(isinstance(value, float) for value in vector)


def test_chat_form_request_shape(backend: MockBackend, client: Client) -> None:
    backend.register("embed_single")
    client.embed("The Beehive.", dimensions=32)
    body = backend.last_call.body
    assert body["messages"] == [
        {"role": "system", "content": EMBEDDING_DOCUMENT_INSTRUCTION},
        {"role": "user", "content": "The Beehive."},
        # The open assistant turn matches how the model was trained to embed
        {"role": "assistant", "content": ""},
    ]
    assert body["encoding_format"] == "float"
    assert body["dimensions"] == 32
    assert body["continue_final_message"] is True
    # Overrides the chat-form server default so tokenization matches
    # batch_embed's plain-input form
    assert body["add_special_tokens"] is True


def test_dimensions_omitted_when_not_given(
    backend: MockBackend, client: Client
) -> None:
    backend.register("embed_single")
    client.embed("The Beehive.")
    assert "dimensions" not in backend.last_call.body


def test_query_uses_query_instruction(backend: MockBackend, client: Client) -> None:
    backend.register("embed_single")
    client.embed("capital of NZ?", query=True)
    system = backend.last_call.body["messages"][0]
    assert system == {"role": "system", "content": EMBEDDING_QUERY_INSTRUCTION}


def test_explicit_instruction_overrides(backend: MockBackend, client: Client) -> None:
    backend.register("embed_single")
    client.embed("some text", query=True, instruction="Embed for clustering.")
    system = backend.last_call.body["messages"][0]
    assert system == {"role": "system", "content": "Embed for clustering."}


def test_multimodal_content_parts(backend: MockBackend, client: Client) -> None:
    backend.register("embed_multimodal")
    vector = client.embed("A small red square.", image_files=[FAKE_PNG])
    assert isinstance(vector, list)
    user_content = backend.last_call.body["messages"][1]["content"]
    assert user_content[0] == {"type": "text", "text": "A small red square."}
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_no_input_raises(client: Client) -> None:
    with pytest.raises(InputError, match="requires text and/or images"):
        client.embed()


def test_empty_string_treated_as_no_text(client: Client) -> None:
    with pytest.raises(InputError, match="requires text and/or images"):
        client.embed("")


def test_zero_dimensions_raises(client: Client) -> None:
    with pytest.raises(InputError, match="positive integer"):
        client.embed("text", dimensions=0)


def test_oversized_dimensions_warns(backend: MockBackend, client: Client) -> None:
    backend.register("embed_single")
    with pytest.warns(UserWarning, match="exceeds the native 4096"):
        client.embed("text", dimensions=8192)


def test_uneven_dimensions_warns(backend: MockBackend, client: Client) -> None:
    backend.register("embed_single")
    with pytest.warns(UserWarning, match="does not divide evenly"):
        client.embed("text", dimensions=3)


def test_empty_response_data_raises(backend: MockBackend, client: Client) -> None:
    fixture = load_fixture("embed_single")
    fixture["response"]["data"] = []
    backend.respond("POST", EMBEDDING_ENDPOINT, fixture["response"])
    with pytest.raises(ResponseError, match="no embedding data"):
        client.embed("text")
