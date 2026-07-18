"""Request and response structs for the /v1/embeddings endpoint.

These mirror the vLLM 0.22.1 embedding schema:
- vllm/entrypoints/pooling/embed/protocol.py
- vllm/entrypoints/pooling/base/protocol.py (request field mixins)
- vllm/utils/serial_utils.py (encoding literals)

Request-side optional fields default to UNSET so msgspec omits them from the
encoded payload and the server applies its own defaults. Response-side fields
use plain defaults because vLLM always serializes every field.

The request structs are encode-only: msgspec cannot build a decoder for them
because the input union contains multiple array-like types (and the
EmbeddingRequest alias unions two untagged structs). The client only ever
encodes requests, so this is by design.

There are two request forms:
- EmbeddingCompletionRequest: plain text/token inputs. A list input is the
  batching approach - each entry produces its own embedding.
- EmbeddingChatRequest: chat messages rendered through the model's chat
  template, for instruction-trained and multimodal embedding models. All
  messages roll up into a SINGLE embedding; it does not batch.
"""

from typing import Any, Literal

from msgspec import UNSET, Struct, UnsetType, field

from sthai.structs.common import Usage
from sthai.structs.completions import ChatMessage, UsageInfo

# --- Requests ---


class EmbeddingCompletionRequest(Struct):
    # A string, a list of strings (the batching approach), or pre-tokenized
    # token IDs (a single prompt or a batch of prompts)
    input: str | list[str] | list[int] | list[list[int]]
    # OpenAI-standard fields
    model: str | UnsetType = UNSET
    encoding_format: Literal["float", "base64", "bytes", "bytes_only"] | UnsetType = (
        UNSET
    )
    # Matryoshka truncation of the output vector; powers of two work best,
    # up to the model's native dimension
    dimensions: int | UnsetType = UNSET
    user: str | UnsetType = UNSET
    # Server default is True for this form (BOS etc. added to raw prompts)
    add_special_tokens: bool | UnsetType = UNSET
    # vLLM-specific fields
    embed_dtype: (
        Literal["float32", "float16", "bfloat16", "fp8_e4m3", "fp8_e5m2"] | UnsetType
    ) = UNSET
    endianness: Literal["native", "big", "little"] | UnsetType = UNSET
    use_activation: bool | UnsetType = UNSET
    truncate_prompt_tokens: int | UnsetType = UNSET
    truncation_side: Literal["left", "right"] | UnsetType = UNSET
    request_id: str | UnsetType = UNSET
    priority: int | UnsetType = UNSET
    mm_processor_kwargs: dict[str, Any] | UnsetType = UNSET
    cache_salt: str | UnsetType = UNSET


class EmbeddingChatRequest(Struct):
    messages: list[ChatMessage]
    # OpenAI-standard fields
    model: str | UnsetType = UNSET
    encoding_format: Literal["float", "base64", "bytes", "bytes_only"] | UnsetType = (
        UNSET
    )
    # Matryoshka truncation of the output vector; powers of two work best,
    # up to the model's native dimension
    dimensions: int | UnsetType = UNSET
    user: str | UnsetType = UNSET
    # Chat template rendering; add_generation_prompt and continue_final_message
    # are mutually exclusive server-side
    add_generation_prompt: bool | UnsetType = UNSET
    continue_final_message: bool | UnsetType = UNSET
    # Server default is False for this form (the chat template already adds
    # its own special tokens)
    add_special_tokens: bool | UnsetType = UNSET
    chat_template: str | UnsetType = UNSET
    chat_template_kwargs: dict[str, Any] | UnsetType = UNSET
    media_io_kwargs: dict[str, dict[str, Any]] | UnsetType = UNSET
    # vLLM-specific fields
    embed_dtype: (
        Literal["float32", "float16", "bfloat16", "fp8_e4m3", "fp8_e5m2"] | UnsetType
    ) = UNSET
    endianness: Literal["native", "big", "little"] | UnsetType = UNSET
    use_activation: bool | UnsetType = UNSET
    truncate_prompt_tokens: int | UnsetType = UNSET
    truncation_side: Literal["left", "right"] | UnsetType = UNSET
    request_id: str | UnsetType = UNSET
    priority: int | UnsetType = UNSET
    mm_processor_kwargs: dict[str, Any] | UnsetType = UNSET
    cache_salt: str | UnsetType = UNSET


EmbeddingRequest = EmbeddingCompletionRequest | EmbeddingChatRequest


# --- Response ---


class EmbeddingResponseData(Struct):
    index: int
    # A list of floats for encoding_format "float", a base64 string otherwise
    embedding: list[float] | str
    object: str = "embedding"


class EmbeddingResponse(Struct):
    id: str
    data: list[EmbeddingResponseData]
    # The raw wire "usage" object; the usage() method summarizes it
    usage_info: UsageInfo = field(name="usage")
    object: str = "list"
    created: int = 0
    model: str | None = None

    def usage(self) -> Usage:
        """Token usage summary; embeddings only consume input tokens."""
        return self.usage_info.summary()

    def output(self) -> list[list[float] | str]:
        """The embeddings in input order (base64/binary formats are strings)."""
        return [d.embedding for d in sorted(self.data, key=lambda d: d.index)]
