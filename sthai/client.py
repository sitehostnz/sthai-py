import warnings
from base64 import b64encode
from os import getenv
from pathlib import Path
from secrets import token_hex
from typing import Any

import msgspec
from msgspec import UNSET
from niquests import request
from niquests.models import Response

from sthai.const import (
    EMBEDDING_ENDPOINT,
    EMBEDDING_PARAMS,
    HEALTH_ENDPOINT,
    INFERENCE_ENDPOINT,
    MODELS_ENDPOINT,
    RERANKING_ENDPOINT,
    SESSION_PIN_HEADER,
)
from sthai.models import EmbeddingModel, InferenceModel, RerankingModel
from sthai.structs.completions import (
    AssistantMessage,
    ChatMessage,
    ContentPart,
    ImageContent,
    ImageURL,
    InferenceRequest,
    InferenceResponse,
    JsonSchemaResponseFormat,
    ResponseFormat,
    SystemMessage,
    TextContent,
    UserMessage,
)
from sthai.structs.embeddings import (
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
    EmbeddingResponse,
)
from sthai.structs.models import ModelCard, ModelList
from sthai.structs.rerank import (
    RerankRequest,
    RerankResponse,
    RerankResult,
    ScoreMultiModalParam,
)
from sthai.typing import HttpMethod

# Magic-byte prefixes for the image formats the API accepts
_IMAGE_MAGIC_BYTES = {
    b"\x89PNG": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF8": "image/gif",
}


def _image_file_to_data_uri(image: Path | bytes) -> str:
    """
    Base64-encode an image file (or raw image bytes) into a data URI
    suitable for the image_url content part of a chat message.
    """
    data = image.read_bytes() if isinstance(image, Path) else image
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        for magic, magic_mime in _IMAGE_MAGIC_BYTES.items():
            if data.startswith(magic):
                mime = magic_mime
                break
        else:
            raise ValueError(
                "unrecognized image format: expected PNG, JPEG, GIF, or WEBP"
            )
    return f"data:{mime};base64,{b64encode(data).decode('ascii')}"


def image_content(image: str | Path | bytes) -> ImageContent:
    """
    Build an image_url content part from a URL string, a local file path,
    or raw image bytes, for hand-constructing multimodal content.
    """
    if isinstance(image, str):
        return ImageContent(image_url=ImageURL(url=image))
    return ImageContent(image_url=ImageURL(url=_image_file_to_data_uri(image)))


def _build_image_parts(
    image_urls: list[str] | None,
    image_files: list[Path | bytes] | None,
) -> list[ImageContent]:
    """Image content parts for the given URLs and local files or bytes."""
    return [
        image_content(image) for image in [*(image_urls or []), *(image_files or [])]
    ]


def _prompt_content(
    prompt: str,
    image_urls: list[str] | None,
    image_files: list[Path | bytes] | None,
) -> str | list[ContentPart]:
    """A user-message content value: plain text, or text plus image parts."""
    image_parts = _build_image_parts(image_urls, image_files)
    return [TextContent(text=prompt), *image_parts] if image_parts else prompt


def _response_format(response_type: type) -> ResponseFormat:
    """
    The response_format for a structured response: plain JSON mode for dict,
    otherwise a JSON schema generated from the type for the server to
    enforce via guided decoding.
    """
    if response_type is dict:
        return ResponseFormat(type="json_object")
    return ResponseFormat(
        type="json_schema",
        json_schema=JsonSchemaResponseFormat(
            name=getattr(response_type, "__name__", "response"),
            json_schema=msgspec.json.schema(response_type),
        ),
    )


def _default_instruction(model: EmbeddingModel | str, query: bool) -> str | None:
    """
    The model's recommended embedding instruction, if known: its query
    instruction when query is set, its document instruction otherwise.
    """
    params = EMBEDDING_PARAMS.get(model)
    if params is None:
        return None
    return params.query_instruction if query else params.document_instruction


def _check_dimensions(model: EmbeddingModel | str, dimensions: int | None) -> None:
    """
    Warn when a requested Matryoshka truncation exceeds or does not divide
    evenly into the model's native output dimension (when both are known).
    """
    if dimensions is None:
        return
    if dimensions < 1:
        raise ValueError("dimensions must be a positive integer")
    params = EMBEDDING_PARAMS.get(model)
    if params is None or params.dimensions is None:
        return
    if dimensions > params.dimensions:
        warnings.warn(
            f"dimensions={dimensions} exceeds the native {params.dimensions} "
            f"dimensions of '{model}'",
            stacklevel=3,
        )
    elif params.dimensions % dimensions != 0:
        warnings.warn(
            f"dimensions={dimensions} does not divide evenly into the native "
            f"{params.dimensions} dimensions of '{model}'; use a power-of-two "
            f"divisor (e.g. {params.dimensions // 2}, {params.dimensions // 4})",
            stacklevel=3,
        )


def _float_embedding(embedding: list[float] | str) -> list[float]:
    """
    Ensure a decoded embedding is the float list the client requested (the
    server returns strings for non-float encoding formats).
    """
    if isinstance(embedding, str):
        raise TypeError("expected a float embedding, got an encoded string")
    return embedding


class Client:
    """
    The main client class for interacting with the SthAI API.
    """

    def __init__(
        self,
        api_key: str = getenv("STHAI_KEY", ""),
        *,
        fqdn: str = "ai.sitehost.nz",
        secure: bool = True,
        session_pin: str | None = None,
        auto_session: bool = False,
        write_history: bool = True,
    ) -> None:
        """
        Create a client. api_key defaults to the STHAI_KEY environment
        variable. session_pin (or auto_session=True to generate one) pins
        requests to a server session; write_history controls whether chat()
        records conversation turns.
        """
        self.fqdn = fqdn
        self.secure = secure
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._session_pin = session_pin
        if not self._session_pin and auto_session:
            self._session_pin = token_hex(24)
        self._write_history = write_history
        self._chat_history: list[ChatMessage] = []
        self._last_response: InferenceResponse | None = None

    def healthy(self) -> bool:
        """Check the server's health status."""
        response = self._make_request(HttpMethod.GET, HEALTH_ENDPOINT)
        return response.ok

    def models(self) -> list[ModelCard]:
        """List the models available on the server."""
        response = self._make_request(HttpMethod.GET, MODELS_ENDPOINT)
        response.raise_for_status()
        return msgspec.json.decode(response.content or b"", type=ModelList).data

    def clear_history(self) -> None:
        """Discard the stored chat history."""
        self._chat_history = []

    def last_response(self) -> InferenceResponse | None:
        """The full response from the most recent chat() or response() call."""
        return self._last_response

    def last_reasoning(self) -> str | None:
        """The reasoning from the most recent chat() or response() call, if any."""
        if self._last_response is None:
            return None
        return self._last_response.output().reasoning

    def chat(
        self,
        prompt: str,
        *,
        model: InferenceModel | str = InferenceModel.QWEN_3_6_27B,
        max_tokens: int | None = None,
        temperature: float | None = None,
        use_thinking: bool = False,
        system_prompt: str | None = None,
        image_urls: list[str] | None = None,
        image_files: list[Path | bytes] | None = None,
        use_history: bool = True,
    ) -> InferenceResponse:
        """
        Send a chat message and return the full inference response.

        With write_history=True (the default), each successful call appends
        the user and assistant turns to the stored history, and later calls
        send that history. Pass use_history=False for a standalone call that
        neither sends nor updates it.
        """
        user_message = UserMessage(
            content=_prompt_content(prompt, image_urls, image_files)
        )

        messages: list[ChatMessage] = []
        if system_prompt:
            # The system prompt is prepended per-call rather than stored in
            # history, so changing it between calls behaves predictably
            messages.append(SystemMessage(content=system_prompt))
        if use_history:
            messages.extend(self._chat_history)
        messages.append(user_message)

        body = InferenceRequest(
            messages=messages,
            model=model,
            # max_tokens is deprecated upstream in favor of max_completion_tokens
            max_completion_tokens=max_tokens if max_tokens is not None else UNSET,
            temperature=temperature if temperature is not None else UNSET,
            chat_template_kwargs={"enable_thinking": use_thinking},
        )
        decoded = self._inference_request(body)

        # A call that didn't see the history must not write to it either,
        # or the stored conversation would gain a turn with missing context
        if use_history and self._write_history and decoded.choices:
            self._chat_history.append(user_message)
            self._chat_history.append(
                AssistantMessage(content=decoded.choices[0].message.content)
            )
        return decoded

    def response[T](
        self,
        prompt: str,
        *,
        response_type: type[T] | None = None,
        model: InferenceModel | str = InferenceModel.QWEN_3_6_27B,
        max_tokens: int | None = None,
        temperature: float | None = None,
        use_thinking: bool = False,
        system_prompt: str | None = None,
        image_urls: list[str] | None = None,
        image_files: list[Path | bytes] | None = None,
    ) -> T | InferenceResponse:
        """
        One-off inference: like chat(), but the stored chat history is
        neither sent nor updated. The response remains available through
        last_response() and last_reasoning().

        Pass response_type for a structured response: a msgspec Struct type
        becomes a JSON schema the server enforces during generation, and the
        response text is decoded and validated into that type and returned.
        response_type=dict just constrains the output to valid JSON. Without
        response_type the full InferenceResponse is returned, as with chat().

        Thinking combines with structured responses (reasoning stays
        unconstrained) but consumes max_tokens, so budget generously. The
        server occasionally skips the schema when thinking is enabled;
        parsing then raises a ValueError - retry, or disable thinking.
        """
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(
            UserMessage(content=_prompt_content(prompt, image_urls, image_files))
        )

        body = InferenceRequest(
            messages=messages,
            model=model,
            max_completion_tokens=max_tokens if max_tokens is not None else UNSET,
            temperature=temperature if temperature is not None else UNSET,
            chat_template_kwargs={"enable_thinking": use_thinking},
            response_format=(
                _response_format(response_type) if response_type is not None else UNSET
            ),
        )
        decoded = self._inference_request(body)
        if response_type is None:
            return decoded
        return decoded.parse(response_type)

    def _inference_request(self, body: InferenceRequest) -> InferenceResponse:
        """POST an inference request and record the decoded last response."""
        response = self._make_request(
            HttpMethod.POST,
            INFERENCE_ENDPOINT,
            data=msgspec.json.encode(body),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        decoded = msgspec.json.decode(response.content or b"", type=InferenceResponse)
        self._last_response = decoded
        return decoded

    def embed(
        self,
        text: str | None = None,
        *,
        model: EmbeddingModel | str = EmbeddingModel.QWEN_3_VL_8B,
        query: bool = False,
        instruction: str | None = None,
        image_urls: list[str] | None = None,
        image_files: list[Path | bytes] | None = None,
        dimensions: int | None = None,
    ) -> list[float]:
        """
        Embed a single input (text, images, or both) and return its vector.

        The instruction-trained model is steered by a default instruction
        from EMBEDDING_PARAMS in sthai.const: the model's document
        instruction, or its query instruction when query=True (use this when
        embedding search queries). Passing instruction overrides either.

        Each call produces exactly ONE vector - multimodal content rolls into
        it; use batch_embed() to embed many texts in one request. dimensions
        truncates the vector server-side (Matryoshka); powers of two work
        best, up to the model's native dimension.
        """
        image_parts = _build_image_parts(image_urls, image_files)
        parts: list[ContentPart] = []
        # An empty string is treated as no text: embedding it would produce a
        # meaningless vector, so it falls through to the guard below instead
        if text:
            parts.append(TextContent(text=text))
        parts.extend(image_parts)
        if not parts:
            raise ValueError("embed() requires text and/or images")
        _check_dimensions(model, dimensions)
        content: str | list[ContentPart] = text if text and not image_parts else parts

        if instruction is None:
            instruction = _default_instruction(model, query)
        messages: list[ChatMessage] = []
        if instruction is not None:
            messages.append(SystemMessage(content=instruction))
        messages.append(UserMessage(content=content))
        # The open assistant turn is intentional: with continue_final_message
        # the template is left unterminated, matching how the model was
        # trained to embed
        messages.append(AssistantMessage(content=""))

        body = EmbeddingChatRequest(
            messages=messages,
            model=model,
            encoding_format="float",
            dimensions=dimensions if dimensions is not None else UNSET,
            continue_final_message=True,
            # True (not the chat-form server default of False) so tokenization
            # matches batch_embed's plain-input form, which defaults to True
            add_special_tokens=True,
        )
        decoded = self._embedding_request(body)
        outputs = decoded.output()
        if not outputs:
            raise ValueError("server returned no embedding data")
        return _float_embedding(outputs[0])

    def batch_embed(
        self,
        texts: list[str],
        *,
        model: EmbeddingModel | str = EmbeddingModel.QWEN_3_VL_8B,
        query: bool = False,
        instruction: str | None = None,
        template: str | None = None,
        dimensions: int | None = None,
    ) -> list[list[float]]:
        """
        Embed a batch of texts in one request, returning one vector per text
        in the same order. Text-only; use embed() for multimodal input.

        Only the plain-input request form batches, and it bypasses the
        server-side chat template, so each text is rendered through a local
        template first; with the built-in templates the results match calling
        embed() per text. template and instruction default from
        EMBEDDING_PARAMS (the query instruction when query=True, as with
        embed()). For models without known params, pass a template using
        {instruction} and {text} placeholders - "{text}" alone for raw
        untemplated input. dimensions truncates the vectors server-side.
        """
        if not texts:
            raise ValueError("batch_embed() requires at least one text")
        if not all(texts):
            raise ValueError("batch_embed() texts must be non-empty strings")
        if template is None:
            params = EMBEDDING_PARAMS.get(model)
            template = params.template if params is not None else None
            if template is None:
                raise ValueError(
                    f"no known embedding template for model '{model}'; pass "
                    'template= (use "{text}" for models that take raw '
                    "untemplated input)"
                )
        if "{instruction}" not in template and (instruction is not None or query):
            warnings.warn(
                "the template has no {instruction} placeholder, so the requested "
                "instruction steering will not be applied",
                stacklevel=2,
            )
        if instruction is None:
            instruction = _default_instruction(model, query)
            if instruction is None and "{instruction}" in template:
                raise ValueError(
                    f"no known embedding instruction for model '{model}' but "
                    "the template expects one; pass instruction="
                )
        _check_dimensions(model, dimensions)
        try:
            inputs = [
                template.format(instruction=instruction, text=text) for text in texts
            ]
        except (KeyError, IndexError) as exc:
            raise ValueError(
                "template must use only the {instruction} and {text} "
                "placeholders; escape literal braces as {{ and }}"
            ) from exc
        body = EmbeddingCompletionRequest(
            input=inputs,
            model=model,
            encoding_format="float",
            dimensions=dimensions if dimensions is not None else UNSET,
        )
        decoded = self._embedding_request(body)
        return [_float_embedding(embedding) for embedding in decoded.output()]

    def _embedding_request(
        self, body: EmbeddingChatRequest | EmbeddingCompletionRequest
    ) -> EmbeddingResponse:
        """POST an embedding request and decode the response."""
        response = self._make_request(
            HttpMethod.POST,
            EMBEDDING_ENDPOINT,
            data=msgspec.json.encode(body),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return msgspec.json.decode(response.content or b"", type=EmbeddingResponse)

    def rerank(
        self,
        query: str | ScoreMultiModalParam,
        documents: list[str | ScoreMultiModalParam],
        *,
        model: RerankingModel | str = RerankingModel.QWEN_3_VL_8B,
        top_n: int | None = None,
        instruction: str | None = None,
    ) -> list[RerankResult]:
        """
        Score each document against the query and return the results sorted
        by relevance score descending, each carrying the document, its
        relevance_score, and its index in the input documents list.

        All documents are returned unless top_n limits it. The
        instruction-trained model applies its own default instruction; pass
        instruction to steer relevance for a specific task. The query and
        each document may be a plain string or, for multimodal input, a
        ScoreMultiModalParam wrapping text/image content parts.
        """
        if not documents:
            raise ValueError("rerank() requires at least one document")
        if top_n is not None and top_n < 1:
            raise ValueError("top_n must be a positive integer")

        body = RerankRequest(
            query=query,
            documents=documents,
            top_n=top_n if top_n is not None else UNSET,
            model=model,
            instruction=instruction if instruction is not None else UNSET,
        )
        response = self._make_request(
            HttpMethod.POST,
            RERANKING_ENDPOINT,
            data=msgspec.json.encode(body),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        decoded = msgspec.json.decode(response.content or b"", type=RerankResponse)
        return decoded.results

    def _server_url(self) -> str:
        """The server's base URL."""
        scheme = "https" if self.secure else "http"
        return f"{scheme}://{self.fqdn}"

    def _endpoint_url(self, endpoint: str) -> str:
        """The absolute URL for an endpoint path."""
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        return f"{self._server_url()}{endpoint}"

    def _default_headers(self) -> dict[str, str]:
        """The auth (and session pin) headers sent with every request."""
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._session_pin:
            headers[SESSION_PIN_HEADER] = self._session_pin
        return headers

    def _make_request(
        self, method: HttpMethod, endpoint: str, **kwargs: Any
    ) -> Response:
        """Send a request with default headers; kwargs pass through to niquests."""
        headers = self._default_headers() | kwargs.pop("headers", {})
        return request(method, self._endpoint_url(endpoint), headers=headers, **kwargs)
