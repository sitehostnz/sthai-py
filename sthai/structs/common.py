"""Client-friendly convenience structs shared by the response types.

Unlike the other structs modules these do not mirror vLLM schemas: they are
the small summaries returned by the usage() and output() helper methods on
the response structs.
"""

from msgspec import Struct


class Usage(Struct):
    """Generic token usage summary for any request type."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


class InferenceOutput(Struct):
    """The text (and any reasoning) produced by a chat completion."""

    text: str | None = None
    reasoning: str | None = None
