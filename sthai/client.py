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
    HEALTH_ENDPOINT,
    INFERENCE_ENDPOINT,
    MODELS_ENDPOINT,
    SESSION_PIN_HEADER,
)
from sthai.models import InferenceModel
from sthai.structs.completions import (
    AssistantMessage,
    ChatMessage,
    ContentPart,
    ImageContent,
    ImageURL,
    InferenceRequest,
    InferenceResponse,
    SystemMessage,
    TextContent,
    UserMessage,
)
from sthai.structs.models import ModelCard, ModelList
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
        """The full response from the most recent chat() call."""
        return self._last_response

    def last_reasoning(self) -> str | None:
        """The reasoning from the most recent chat() call, if any."""
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
