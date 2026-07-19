"""rerank(): result ordering, wire shape, and guards."""

import pytest

from sthai.client import Client
from sthai.exceptions import InputError
from sthai.structs.completions import TextContent
from sthai.structs.rerank import RerankResult, ScoreMultiModalParam

from conftest import MockBackend

QUERY = "What is the capital of New Zealand?"
DOCUMENTS = [
    "The capital of New Zealand is Wellington.",
    "Auckland has the largest population in New Zealand.",
    "Kiwi are nocturnal flightless birds native to New Zealand.",
    "The All Blacks are New Zealand's national rugby team.",
]


def test_results_sorted_by_relevance(backend: MockBackend, client: Client) -> None:
    backend.register("rerank")
    results = client.rerank(QUERY, DOCUMENTS)
    assert len(results) == len(DOCUMENTS)
    assert all(isinstance(result, RerankResult) for result in results)
    scores = [result.relevance_score for result in results]
    assert scores == sorted(scores, reverse=True)
    # Each index maps back to the position in the input documents list
    for result in results:
        assert result.document.text == DOCUMENTS[result.index]


def test_minimal_request_body(backend: MockBackend, client: Client) -> None:
    backend.register("rerank")
    client.rerank(QUERY, DOCUMENTS)
    body = backend.last_call.body
    assert body["query"] == QUERY
    assert body["documents"] == DOCUMENTS
    # None means "return all documents": the server default applies
    assert "top_n" not in body
    assert "instruction" not in body


def test_top_n_limits_results(backend: MockBackend, client: Client) -> None:
    backend.register("rerank_top_n")
    results = client.rerank(QUERY, DOCUMENTS, top_n=2)
    assert len(results) == 2
    assert backend.last_call.body["top_n"] == 2


def test_instruction_passthrough(backend: MockBackend, client: Client) -> None:
    backend.register("rerank")
    client.rerank(QUERY, DOCUMENTS, instruction="Rank by factual accuracy.")
    assert backend.last_call.body["instruction"] == "Rank by factual accuracy."


def test_multimodal_query_encoding(backend: MockBackend, client: Client) -> None:
    backend.register("rerank")
    query = ScoreMultiModalParam(content=[TextContent(text=QUERY)])
    client.rerank(query, DOCUMENTS)
    assert backend.last_call.body["query"] == {
        "content": [{"type": "text", "text": QUERY}]
    }


def test_empty_documents_raises(client: Client) -> None:
    with pytest.raises(InputError, match="at least one document"):
        client.rerank(QUERY, [])


def test_zero_top_n_raises(client: Client) -> None:
    with pytest.raises(InputError, match="positive integer"):
        client.rerank(QUERY, DOCUMENTS, top_n=0)
