from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

import msgspec
from niquests import arequest
from niquests.exceptions import RequestException
from niquests.models import Response

from sthai.client import _BaseClient
from sthai.const import (
    EMBEDDING_ENDPOINT,
    HEALTH_ENDPOINT,
    INFERENCE_ENDPOINT,
    MODELS_ENDPOINT,
    RERANKING_ENDPOINT,
)
from sthai.exceptions import TransportError
from sthai.models import EmbeddingModel, InferenceModel, RerankingModel
from sthai.structs.completions import InferenceRequest, InferenceResponse
from sthai.structs.embeddings import (
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
    EmbeddingResponse,
)
from sthai.structs.models import ModelCard
from sthai.structs.rerank import RerankResult, ScoreMultiModalParam
from sthai.typing import HttpMethod

# TypeVar rather than PEP 695 syntax to stay compatible with Python 3.10
T = TypeVar("T")


class AsyncClient(_BaseClient):
    """
    The async counterpart to Client: identical constructor and methods, with
    every network method awaitable. Like Client it is session-less - each
    call sends a one-shot request, so there is no async context manager or
    close() to manage.
    """

    async def healthy(self) -> bool:
        """Check the server's health status."""
        response = await self._make_request(HttpMethod.GET, HEALTH_ENDPOINT)
        return response.ok

    async def models(self) -> list[ModelCard]:
        """List the models available on the server."""
        response = await self._make_request(HttpMethod.GET, MODELS_ENDPOINT)
        return self._finish_models(response)

    async def chat(
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
        body, user_message = self._build_chat_request(
            prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            use_thinking=use_thinking,
            system_prompt=system_prompt,
            image_urls=image_urls,
            image_files=image_files,
            use_history=use_history,
        )
        decoded = await self._inference_request(body)
        self._record_chat_turns(user_message, decoded, use_history)
        return decoded

    async def response(
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
        parsing then raises a ResponseParseError - retry, or disable thinking.
        """
        body = self._build_response_request(
            prompt,
            response_type=response_type,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            use_thinking=use_thinking,
            system_prompt=system_prompt,
            image_urls=image_urls,
            image_files=image_files,
        )
        decoded = await self._inference_request(body)
        if response_type is None:
            return decoded
        return decoded.parse(response_type)

    async def _inference_request(self, body: InferenceRequest) -> InferenceResponse:
        """POST an inference request and record the decoded last response."""
        response = await self._make_request(
            HttpMethod.POST,
            INFERENCE_ENDPOINT,
            data=msgspec.json.encode(body),
            headers={"Content-Type": "application/json"},
        )
        return self._finish_inference(response)

    async def embed(
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
        body = self._build_embed_request(
            text,
            model=model,
            query=query,
            instruction=instruction,
            image_urls=image_urls,
            image_files=image_files,
            dimensions=dimensions,
        )
        decoded = await self._embedding_request(body)
        return self._extract_single_embedding(decoded)

    async def batch_embed(
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
        body = self._build_batch_embed_request(
            texts,
            model=model,
            query=query,
            instruction=instruction,
            template=template,
            dimensions=dimensions,
        )
        decoded = await self._embedding_request(body)
        return self._extract_batch_embeddings(decoded)

    async def _embedding_request(
        self, body: EmbeddingChatRequest | EmbeddingCompletionRequest
    ) -> EmbeddingResponse:
        """POST an embedding request and decode the response."""
        response = await self._make_request(
            HttpMethod.POST,
            EMBEDDING_ENDPOINT,
            data=msgspec.json.encode(body),
            headers={"Content-Type": "application/json"},
        )
        return self._finish_embedding(response)

    async def rerank(
        self,
        query: str | ScoreMultiModalParam,
        # Sequence rather than list so a plain list[str] type-checks
        documents: Sequence[str | ScoreMultiModalParam],
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
        body = self._build_rerank_request(
            query,
            documents,
            model=model,
            top_n=top_n,
            instruction=instruction,
        )
        response = await self._make_request(
            HttpMethod.POST,
            RERANKING_ENDPOINT,
            data=msgspec.json.encode(body),
            headers={"Content-Type": "application/json"},
        )
        return self._finish_rerank(response)

    async def _make_request(
        self, method: HttpMethod, endpoint: str, **kwargs: Any
    ) -> Response:
        """Send a request with default headers; kwargs pass through to niquests."""
        headers = self._default_headers() | kwargs.pop("headers", {})
        # Non-streaming arequest calls always return a fully-read Response,
        # but the **kwargs call resolves to the Response | AsyncResponse
        # overload, hence the cast
        try:
            return cast(
                Response,
                await arequest(
                    method, self._endpoint_url(endpoint), headers=headers, **kwargs
                ),
            )
        except RequestException as exc:
            raise TransportError(f"request failed: {exc}") from exc
