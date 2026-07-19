"""Request and response structs for the /v1/chat/completions endpoint.

These mirror the vLLM 0.22.1 chat completion schema:
- vllm/entrypoints/openai/chat_completion/protocol.py
- vllm/entrypoints/openai/engine/protocol.py

Request-side optional fields default to UNSET so msgspec omits them from the
encoded payload and the server applies its own defaults. Response-side fields
use plain defaults because vLLM always serializes every field.
"""

from typing import Any, Literal, TypeVar

import msgspec
from msgspec import UNSET, Struct, UnsetType, field

from sthai.exceptions import ResponseParseError
from sthai.structs.common import InferenceOutput, Usage

# TypeVar rather than PEP 695 syntax to stay compatible with Python 3.10
T = TypeVar("T")

# --- Shared tool-call structs (request assistant messages and responses) ---


class FunctionCall(Struct):
    name: str
    arguments: str


class ToolCall(Struct):
    id: str
    function: FunctionCall
    type: Literal["function"] = "function"


# --- Request message content parts (tagged union on "type") ---


class TextContent(Struct, tag_field="type", tag="text"):
    text: str


class ImageURL(Struct):
    url: str
    detail: str | UnsetType = UNSET


class ImageContent(Struct, tag_field="type", tag="image_url"):
    image_url: ImageURL


ContentPart = TextContent | ImageContent


# --- Request messages (tagged union on "role") ---


class SystemMessage(Struct, tag_field="role", tag="system"):
    content: str | list[TextContent]
    name: str | UnsetType = UNSET


class UserMessage(Struct, tag_field="role", tag="user"):
    content: str | list[ContentPart]
    name: str | UnsetType = UNSET


class AssistantMessage(Struct, tag_field="role", tag="assistant"):
    content: str | list[TextContent] | None = None
    name: str | UnsetType = UNSET
    tool_calls: list[ToolCall] | UnsetType = UNSET
    # vLLM-specific: reasoning output from a previous turn of a thinking model
    reasoning: str | UnsetType = UNSET


class ToolMessage(Struct, tag_field="role", tag="tool"):
    content: str | list[TextContent]
    tool_call_id: str


ChatMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage


# --- Request tool definitions ---


class FunctionDefinition(Struct):
    name: str
    description: str | UnsetType = UNSET
    parameters: dict[str, Any] | UnsetType = UNSET


class Tool(Struct):
    function: FunctionDefinition
    type: Literal["function"] = "function"


class NamedFunction(Struct):
    name: str


class NamedToolChoice(Struct):
    function: NamedFunction
    type: Literal["function"] = "function"


# --- Request response-format and stream options ---


class JsonSchemaResponseFormat(Struct):
    name: str
    description: str | UnsetType = UNSET
    # Serialized as "schema" on the wire, per the OpenAI spec
    json_schema: dict[str, Any] | UnsetType = field(default=UNSET, name="schema")
    strict: bool | UnsetType = UNSET


class ResponseFormat(Struct):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: JsonSchemaResponseFormat | UnsetType = UNSET


class StreamOptions(Struct):
    include_usage: bool | UnsetType = UNSET
    continuous_usage_stats: bool | UnsetType = UNSET


# --- Request ---


class InferenceRequest(Struct):
    messages: list[ChatMessage]
    # OpenAI-standard fields
    model: str | UnsetType = UNSET
    frequency_penalty: float | UnsetType = UNSET
    logit_bias: dict[str, float] | UnsetType = UNSET
    logprobs: bool | UnsetType = UNSET
    top_logprobs: int | UnsetType = UNSET
    # Deprecated upstream in favor of max_completion_tokens
    max_tokens: int | UnsetType = UNSET
    max_completion_tokens: int | UnsetType = UNSET
    n: int | UnsetType = UNSET
    presence_penalty: float | UnsetType = UNSET
    response_format: ResponseFormat | UnsetType = UNSET
    seed: int | UnsetType = UNSET
    stop: str | list[str] | UnsetType = UNSET
    stream: bool | UnsetType = UNSET
    stream_options: StreamOptions | UnsetType = UNSET
    temperature: float | UnsetType = UNSET
    top_p: float | UnsetType = UNSET
    tools: list[Tool] | UnsetType = UNSET
    tool_choice: Literal["none", "auto", "required"] | NamedToolChoice | UnsetType = (
        UNSET
    )
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | UnsetType
    ) = UNSET
    thinking_token_budget: int | UnsetType = UNSET
    include_reasoning: bool | UnsetType = UNSET
    parallel_tool_calls: bool | UnsetType = UNSET
    user: str | UnsetType = UNSET
    # vLLM-specific: extra kwargs for the chat template renderer,
    # e.g. {"enable_thinking": False} to toggle thinking on Qwen models
    chat_template_kwargs: dict[str, Any] | UnsetType = UNSET
    # vLLM-specific sampling params
    use_beam_search: bool | UnsetType = UNSET
    top_k: int | UnsetType = UNSET
    min_p: float | UnsetType = UNSET
    repetition_penalty: float | UnsetType = UNSET
    length_penalty: float | UnsetType = UNSET
    stop_token_ids: list[int] | UnsetType = UNSET
    include_stop_str_in_output: bool | UnsetType = UNSET
    ignore_eos: bool | UnsetType = UNSET
    min_tokens: int | UnsetType = UNSET
    skip_special_tokens: bool | UnsetType = UNSET
    spaces_between_special_tokens: bool | UnsetType = UNSET
    truncate_prompt_tokens: int | UnsetType = UNSET
    truncation_side: Literal["left", "right"] | UnsetType = UNSET
    prompt_logprobs: int | UnsetType = UNSET
    allowed_token_ids: list[int] | UnsetType = UNSET
    bad_words: list[str] | UnsetType = UNSET


# --- Response ---


class ResponseMessage(Struct):
    role: str
    content: str | None = None
    refusal: str | None = None
    annotations: dict[str, Any] | None = None
    audio: dict[str, Any] | None = None
    function_call: FunctionCall | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    # vLLM-specific: reasoning output from thinking models
    reasoning: str | None = None


class LogProb(Struct):
    token: str
    logprob: float = -9999.0
    bytes: list[int] | None = None


class LogProbsContent(Struct):
    token: str
    logprob: float = -9999.0
    bytes: list[int] | None = None
    top_logprobs: list[LogProb] = field(default_factory=list)


class LogProbs(Struct):
    content: list[LogProbsContent] | None = None


class ResponseChoice(Struct):
    index: int
    message: ResponseMessage
    logprobs: LogProbs | None = None
    finish_reason: str | None = "stop"
    # vLLM-specific fields
    stop_reason: int | str | None = None
    token_ids: list[int] | None = None
    routed_experts: str | None = None


class PromptTokenUsageInfo(Struct):
    cached_tokens: int | None = None


class UsageInfo(Struct):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0
    prompt_tokens_details: PromptTokenUsageInfo | None = None

    def summary(self) -> Usage:
        """Condense into the generic input/output/cached token summary."""
        details = self.prompt_tokens_details
        return Usage(
            input_tokens=self.prompt_tokens,
            output_tokens=self.completion_tokens or 0,
            cached_tokens=(details.cached_tokens or 0) if details is not None else 0,
        )


class InferenceResponse(Struct):
    id: str
    model: str
    choices: list[ResponseChoice]
    # The raw wire "usage" object; the usage() method summarizes it
    usage_info: UsageInfo = field(name="usage")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = 0
    service_tier: str | None = None
    system_fingerprint: str | None = None
    # vLLM-specific fields
    prompt_logprobs: list[dict[int, Any] | None] | None = None
    prompt_token_ids: list[int] | None = None
    prompt_text: str | None = None
    kv_transfer_params: dict[str, Any] | None = None

    def usage(self) -> Usage:
        """Token usage summary: input, output, and cached prompt tokens."""
        return self.usage_info.summary()

    def output(self) -> InferenceOutput:
        """The first choice's response text and reasoning, if any."""
        if not self.choices:
            return InferenceOutput()
        message = self.choices[0].message
        return InferenceOutput(text=message.content, reasoning=message.reasoning)

    def parse(self, response_type: type[T]) -> T:
        """
        Decode the response text into response_type, validating it.

        Guided decoding guarantees schema-valid syntax but not completeness
        (token-limit cutoffs) nor, with reasoning models, that the schema was
        applied at all - so parsing doubles as the check, raising a
        ResponseParseError naming the cause on failure.
        """
        text = self.output().text
        if text is None:
            raise ResponseParseError("response has no text content to parse")
        try:
            return msgspec.json.decode(text.encode(), type=response_type)
        except msgspec.ValidationError as exc:
            # Valid JSON of the wrong shape; msgspec's message already names
            # the offending field, so pass it through as-is
            raise ResponseParseError(str(exc)) from exc
        except msgspec.DecodeError as exc:
            if self.choices and self.choices[0].finish_reason == "length":
                raise ResponseParseError(
                    "structured response was cut off by the token limit before "
                    "the JSON completed; raise max_tokens"
                ) from exc
            # Guided decoding should make this impossible, but reasoning
            # models have been seen to intermittently emit non-JSON content
            raise ResponseParseError(
                f"structured response is not valid JSON ({exc}); "
                f"content began: {text[:120]!r}"
            ) from exc
