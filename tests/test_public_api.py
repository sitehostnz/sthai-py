"""The package's public surface: what `from sthai import ...` provides."""

import inspect

import pytest

import sthai
from sthai import AsyncClient, Client, SthaiError

# Every public API method the two clients must expose identically
API_METHODS = [
    "healthy",
    "models",
    "chat",
    "response",
    "embed",
    "batch_embed",
    "rerank",
]


def test_all_names_are_importable() -> None:
    for name in sthai.__all__:
        assert getattr(sthai, name) is not None


def test_public_surface() -> None:
    # The names the README examples rely on
    assert set(sthai.__all__) == {
        "APIError",
        "APIStatusError",
        "AsyncClient",
        "Client",
        "ClientError",
        "EmbeddingModel",
        "InferenceModel",
        "InputError",
        "RerankingModel",
        "ResponseError",
        "ResponseParseError",
        "SthaiError",
        "TransportError",
        "image_content",
    }


@pytest.mark.parametrize(
    ("child", "parent"),
    [
        (sthai.InputError, SthaiError),
        (sthai.TransportError, SthaiError),
        (sthai.APIStatusError, SthaiError),
        (sthai.ClientError, sthai.APIStatusError),
        (sthai.APIError, sthai.APIStatusError),
        (sthai.ResponseError, SthaiError),
        (sthai.ResponseParseError, sthai.ResponseError),
    ],
)
def test_exception_hierarchy(child: type, parent: type) -> None:
    # `except SthaiError` must be enough to catch anything sthai raises
    assert issubclass(child, parent)


def test_async_client_mirrors_client_signatures() -> None:
    # The twins must not drift: same signature per method, async on one side
    for name in API_METHODS:
        sync_method = getattr(Client, name)
        async_method = getattr(AsyncClient, name)
        # Compare rendered signatures: each module has its own T TypeVar, so
        # response()'s signatures are equal in spelling but not by object
        assert str(inspect.signature(async_method)) == str(
            inspect.signature(sync_method)
        ), name
        assert inspect.iscoroutinefunction(async_method), name
        assert not inspect.iscoroutinefunction(sync_method), name
