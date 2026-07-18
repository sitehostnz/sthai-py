"""Request and response structs for the /v1/rerank endpoint.

These mirror the vLLM 0.22.1 rerank schema:
- vllm/entrypoints/pooling/scoring/protocol.py
- vllm/entrypoints/pooling/scoring/typing.py (ScoreInput)
- vllm/entrypoints/pooling/base/protocol.py (request field mixins)

Request-side optional fields default to UNSET so msgspec omits them from the
encoded payload and the server applies its own defaults. Response-side fields
use plain defaults because vLLM always serializes every field.

The query and each document are scored as pairs; the response is a
Cohere-compatible results array sorted by relevance score descending, with
each result's index referencing the document's position in the request.
"""

from typing import Any, Literal

from msgspec import UNSET, Struct, UnsetType, field

from sthai.structs.common import Usage
from sthai.structs.completions import ContentPart

# --- Requests ---


class ScoreMultiModalParam(Struct):
    # A multimodal query or document: content parts without a chat role
    content: list[ContentPart]


# A plain string, or content parts for multimodal input
ScoreInput = str | ScoreMultiModalParam


class RerankRequest(Struct):
    query: ScoreInput
    documents: ScoreInput | list[ScoreInput]
    # Server default is 0, meaning return all documents
    top_n: int | UnsetType = UNSET
    model: str | UnsetType = UNSET
    user: str | UnsetType = UNSET
    # Task instruction prepended to each scored pair via the chat template;
    # folded into chat_template_kwargs server-side (explicit keys there win).
    # The served model applies its own default instruction when omitted.
    instruction: str | UnsetType = UNSET
    chat_template_kwargs: dict[str, Any] | UnsetType = UNSET
    # Per-side truncation; server default is 0, meaning no truncation
    max_tokens_per_query: int | UnsetType = UNSET
    max_tokens_per_doc: int | UnsetType = UNSET
    # vLLM-specific fields
    use_activation: bool | UnsetType = UNSET
    truncate_prompt_tokens: int | UnsetType = UNSET
    truncation_side: Literal["left", "right"] | UnsetType = UNSET
    request_id: str | UnsetType = UNSET
    priority: int | UnsetType = UNSET
    mm_processor_kwargs: dict[str, Any] | UnsetType = UNSET
    cache_salt: str | UnsetType = UNSET


# --- Response ---


class RerankDocument(Struct):
    text: str | None = None
    multi_modal: list[ContentPart] | None = None


class RerankResult(Struct):
    index: int
    document: RerankDocument
    relevance_score: float


class RerankUsage(Struct):
    # Reranking costs input tokens only - no completion_tokens field,
    # matching vLLM's RerankUsage rather than the shared UsageInfo
    prompt_tokens: int = 0
    total_tokens: int = 0


class RerankResponse(Struct):
    id: str
    model: str
    # The raw wire "usage" object; the usage() method summarizes it
    usage_info: RerankUsage = field(name="usage")
    results: list[RerankResult]

    def usage(self) -> Usage:
        """Token usage summary; reranking only consumes input tokens."""
        return Usage(input_tokens=self.usage_info.prompt_tokens)

    def output(self) -> list[RerankResult]:
        """The rerank results, sorted by relevance score descending."""
        return self.results
