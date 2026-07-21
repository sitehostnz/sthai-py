"""Struct-level behaviour: wire encoding, renames, and accessor methods."""

import msgspec
import pytest

from sthai.exceptions import ResponseError
from sthai.structs.common import InferenceOutput, Usage
from sthai.structs.completions import (
    InferenceRequest,
    InferenceResponse,
    JsonSchemaResponseFormat,
    PromptTokenUsageInfo,
    UsageInfo,
    UserMessage,
)
from sthai.structs.embeddings import EmbeddingResponse, EmbeddingResponseData
from sthai.structs.rerank import (
    RerankDocument,
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
)


def encoded_keys(struct: msgspec.Struct) -> set[str]:
    return set(msgspec.json.decode(msgspec.json.encode(struct)))


def test_unset_inference_request_fields_omitted() -> None:
    request = InferenceRequest(messages=[UserMessage(content="hi")])
    assert encoded_keys(request) == {"messages"}


def test_unset_rerank_request_fields_omitted() -> None:
    request = RerankRequest(query="q", documents=["d"])
    assert encoded_keys(request) == {"query", "documents"}


def test_usage_info_wire_rename_round_trips() -> None:
    wire = (
        b'{"id": "x", "model": "m", "choices": [],'
        b' "usage": {"prompt_tokens": 7, "total_tokens": 7}}'
    )
    decoded = msgspec.json.decode(wire, type=InferenceResponse)
    assert decoded.usage_info.prompt_tokens == 7
    # And it re-encodes under the wire name, not the attribute name
    keys = encoded_keys(decoded)
    assert "usage" in keys
    assert "usage_info" not in keys


def test_usage_rename_on_embedding_and_rerank_responses() -> None:
    embedding = EmbeddingResponse(
        id="x",
        data=[EmbeddingResponseData(index=0, embedding=[0.1])],
        usage_info=UsageInfo(prompt_tokens=3, total_tokens=3),
    )
    rerank = RerankResponse(
        id="x",
        model="m",
        usage_info=RerankUsage(prompt_tokens=5, total_tokens=5),
        results=[],
    )
    assert "usage" in encoded_keys(embedding)
    assert "usage" in encoded_keys(rerank)


def test_usage_summary_includes_cached_tokens() -> None:
    info = UsageInfo(
        prompt_tokens=100,
        total_tokens=140,
        completion_tokens=40,
        prompt_tokens_details=PromptTokenUsageInfo(cached_tokens=64),
    )
    assert info.summary() == Usage(input_tokens=100, output_tokens=40, cached_tokens=64)


def test_usage_summary_defaults_missing_details_to_zero() -> None:
    info = UsageInfo(prompt_tokens=10, total_tokens=10, completion_tokens=None)
    assert info.summary() == Usage(input_tokens=10)


def test_embedding_output_sorted_by_index() -> None:
    response = EmbeddingResponse(
        id="x",
        data=[
            EmbeddingResponseData(index=1, embedding=[0.2]),
            EmbeddingResponseData(index=0, embedding=[0.1]),
        ],
        usage_info=UsageInfo(),
    )
    assert response.output() == [[0.1], [0.2]]


def test_embedding_output_rejects_encoded_strings() -> None:
    response = EmbeddingResponse(
        id="x",
        data=[EmbeddingResponseData(index=0, embedding="bm90IGZsb2F0cw==")],
        usage_info=UsageInfo(),
    )
    with pytest.raises(ResponseError, match="encoded string"):
        response.output()


def test_embedding_output_on_empty_data_raises() -> None:
    response = EmbeddingResponse(id="x", data=[], usage_info=UsageInfo())
    with pytest.raises(ResponseError, match="no embedding data"):
        response.output()


def test_rerank_usage_is_input_only() -> None:
    response = RerankResponse(
        id="x",
        model="m",
        usage_info=RerankUsage(prompt_tokens=9, total_tokens=9),
        results=[],
    )
    assert response.usage() == Usage(input_tokens=9)


def test_rerank_output_returns_results() -> None:
    results = [
        RerankResult(index=1, document=RerankDocument(text="a"), relevance_score=0.9),
        RerankResult(index=0, document=RerankDocument(text="b"), relevance_score=0.1),
    ]
    response = RerankResponse(
        id="x", model="m", usage_info=RerankUsage(), results=results
    )
    assert response.output() == results


def test_output_on_empty_choices_is_empty() -> None:
    response = InferenceResponse(id="x", model="m", choices=[], usage_info=UsageInfo())
    assert response.output() == InferenceOutput(text=None, reasoning=None)


def test_json_schema_serialized_under_schema_key() -> None:
    response_format = JsonSchemaResponseFormat(
        name="thing", json_schema={"type": "object"}
    )
    encoded = msgspec.json.decode(msgspec.json.encode(response_format))
    assert encoded == {"name": "thing", "schema": {"type": "object"}}
