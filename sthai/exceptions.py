"""
sthai-specific exceptions.

Every error this client raises subclasses SthaiError, so catching it is
enough to handle anything sthai raises - callers never need to import
niquests or msgspec to handle library errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
from msgspec import Struct

if TYPE_CHECKING:
    from niquests.models import Response


class SthaiError(Exception):
    """Base class for every exception sthai raises."""


class InputError(SthaiError):
    """
    Invalid arguments passed to a client method: bad image data, an empty
    document or text list, a non-positive dimensions or top_n, a missing
    api_key, or an embedding template with unsupported placeholders.
    """


class TransportError(SthaiError):
    """
    The request could not reach the server or complete: a connection
    failure, timeout, TLS error, or similar transport-level problem from
    the underlying HTTP library. The original exception is available as
    __cause__.
    """


class ResponseError(SthaiError):
    """
    The server returned a response sthai could not use: a success body that
    didn't decode into the expected shape, or one that decoded but was
    missing data the caller needs (e.g. no embedding vector returned).
    """


class ResponseParseError(ResponseError):
    """
    InferenceResponse.parse() could not produce response_type: the model
    produced no text, its output was cut off by the token limit, the text
    wasn't valid JSON, or the JSON didn't match response_type's schema.
    """


class APIStatusError(SthaiError):
    """
    Base class for HTTP error-status responses; catch this to handle
    ClientError and APIError together. Carries status_code, the raw
    response, and (when the server sent a structured error body)
    server_message and error_type.
    """

    def __init__(
        self,
        response: Response,
        *,
        server_message: str | None = None,
        error_type: str | None = None,
    ) -> None:
        self.status_code: int = response.status_code or 0
        self.response = response
        self.server_message = server_message
        self.error_type = error_type
        super().__init__(
            _format_message(self.status_code, server_message, error_type, response)
        )


class ClientError(APIStatusError):
    """A 4xx response: the request was invalid, unauthorized, or rejected."""


class APIError(APIStatusError):
    """A 5xx response: the server encountered an error or is unavailable."""


class _ErrorDetail(Struct):
    message: str
    type: str | None = None


class _ErrorEnvelope(Struct):
    """The OpenAI-style error body: {"error": {"message": ..., "type": ...}}."""

    error: _ErrorDetail


def _parse_error_envelope(response: Response) -> tuple[str | None, str | None]:
    """The server message and error type from the body, or (None, None)."""
    try:
        envelope = msgspec.json.decode(response.content or b"", type=_ErrorEnvelope)
    except (msgspec.DecodeError, msgspec.ValidationError):
        return None, None
    return envelope.error.message, envelope.error.type


def _format_message(
    status_code: int,
    server_message: str | None,
    error_type: str | None,
    response: Response,
) -> str:
    if server_message:
        suffix = f" ({error_type})" if error_type else ""
        return f"{status_code}: {server_message}{suffix}"
    reason = getattr(response, "reason", None)
    return f"{status_code}: {reason}" if reason else f"{status_code}: HTTP error"


def raise_for_status(response: Response) -> None:
    """
    Raise ClientError for a 4xx response or APIError for a 5xx response,
    parsing the error envelope from the body when present. Does nothing
    for success statuses.
    """
    status_code = response.status_code or 0
    if status_code < 400:
        return
    server_message, error_type = _parse_error_envelope(response)
    cls = ClientError if status_code < 500 else APIError
    raise cls(response, server_message=server_message, error_type=error_type)


__all__ = [
    "APIError",
    "APIStatusError",
    "ClientError",
    "InputError",
    "ResponseError",
    "ResponseParseError",
    "SthaiError",
    "TransportError",
]
