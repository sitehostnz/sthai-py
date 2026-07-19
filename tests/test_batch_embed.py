"""batch_embed(): local templating, result ordering, and guards."""

import pytest

from sthai.client import Client
from sthai.const import (
    EMBEDDING_DOCUMENT_INSTRUCTION,
    EMBEDDING_ENDPOINT,
    QWEN_3_VL_EMBEDDING_TEMPLATE,
)
from sthai.exceptions import InputError

from conftest import MockBackend, load_fixture

TEXTS = [
    "Wellington is the capital of New Zealand.",
    "Auckland is the largest city in New Zealand.",
    "The kiwi is a flightless bird.",
]


def test_one_vector_per_text_in_order(backend: MockBackend, client: Client) -> None:
    fixture = backend.register("batch_embed")
    vectors = client.batch_embed(TEXTS, dimensions=32)
    expected = [entry["embedding"] for entry in fixture["response"]["data"]]
    assert vectors == expected
    assert all(len(vector) == 32 for vector in vectors)


def test_out_of_order_response_resorted_by_index(
    backend: MockBackend, client: Client
) -> None:
    fixture = load_fixture("batch_embed")
    by_index = {entry["index"]: entry for entry in fixture["response"]["data"]}
    fixture["response"]["data"] = [by_index[2], by_index[0], by_index[1]]
    backend.respond("POST", EMBEDDING_ENDPOINT, fixture["response"])
    vectors = client.batch_embed(TEXTS)
    assert vectors == [by_index[i]["embedding"] for i in range(3)]


def test_inputs_are_locally_templated(backend: MockBackend, client: Client) -> None:
    backend.register("batch_embed")
    client.batch_embed(TEXTS)
    body = backend.last_call.body
    assert body["input"] == [
        QWEN_3_VL_EMBEDDING_TEMPLATE.format(
            instruction=EMBEDDING_DOCUMENT_INSTRUCTION, text=text
        )
        for text in TEXTS
    ]
    assert body["encoding_format"] == "float"


def test_custom_raw_template(backend: MockBackend, client: Client) -> None:
    backend.register("batch_embed")
    client.batch_embed(TEXTS, model="unknown/model", template="{text}")
    assert backend.last_call.body["input"] == TEXTS


def test_empty_list_raises(client: Client) -> None:
    with pytest.raises(InputError, match="at least one text"):
        client.batch_embed([])


def test_empty_string_member_raises(client: Client) -> None:
    with pytest.raises(InputError, match="non-empty strings"):
        client.batch_embed(["fine", ""])


def test_unknown_model_without_template_raises(client: Client) -> None:
    with pytest.raises(InputError, match="no known embedding template"):
        client.batch_embed(["text"], model="unknown/model")


def test_template_without_instruction_placeholder_warns(
    backend: MockBackend, client: Client
) -> None:
    backend.register("batch_embed")
    with pytest.warns(UserWarning, match="no {instruction} placeholder"):
        client.batch_embed(TEXTS, template="{text}", query=True)


def test_stray_template_placeholder_raises(client: Client) -> None:
    with pytest.raises(InputError, match="escape literal braces"):
        client.batch_embed(["text"], template="{text} {oops}")
