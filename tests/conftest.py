"""
Shared test fixtures: a mock backend that replays captured live responses.

The fixtures in tests/fixtures/ are real exchanges recorded from the live
API by tests/capture_fixtures.py. MockBackend replaces sthai.client.request
(the module-level niquests import), so tests exercise the full client path -
URL building, headers, body encoding - without any network access.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from niquests.exceptions import HTTPError

from sthai.client import Client

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    """The raw captured exchange from tests/fixtures/<name>.json."""
    return json.loads((FIXTURES_DIR / f"{name}.json").read_text())


class FakeResponse:
    """Stand-in for niquests.models.Response covering what the client uses."""

    def __init__(self, status_code: int, content: bytes) -> None:
        self.status_code = status_code
        self.content = content

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise HTTPError(f"{self.status_code} Error")


@dataclass
class Call:
    """One request the client sent to the mock backend."""

    method: str
    url: str
    path: str
    headers: dict[str, str]
    # The request body decoded from JSON, or None for body-less requests
    body: Any


class MockBackend:
    """Routes (method, path) to canned responses and records every call."""

    def __init__(self) -> None:
        self._routes: dict[tuple[str, str], tuple[int, bytes]] = {}
        self.calls: list[Call] = []

    def register(self, name: str) -> dict[str, Any]:
        """Route a captured fixture by name; returns it for assertions."""
        fixture = load_fixture(name)
        self.respond(
            fixture["method"],
            fixture["endpoint"],
            fixture["response"],
            status_code=fixture["status_code"],
        )
        return fixture

    def respond(
        self,
        method: str,
        endpoint: str,
        body: Any,
        *,
        status_code: int = 200,
    ) -> None:
        """Route an arbitrary response body (dict/list encoded as JSON)."""
        content = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self._routes[(method, endpoint)] = (status_code, content)

    @property
    def last_call(self) -> Call:
        return self.calls[-1]

    def handle(self, method: Any, url: str, **kwargs: Any) -> FakeResponse:
        """Replacement for niquests.request: replay the registered route."""
        path = urlsplit(url).path
        data = kwargs.get("data")
        self.calls.append(
            Call(
                method=str(method),
                url=url,
                path=path,
                headers=dict(kwargs.get("headers") or {}),
                body=json.loads(data) if data else None,
            )
        )
        route = self._routes.get((str(method), path))
        if route is None:
            raise AssertionError(f"no fixture registered for {method} {path}")
        return FakeResponse(*route)


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> MockBackend:
    """A fresh MockBackend patched in as the client's transport."""
    mock = MockBackend()
    monkeypatch.setattr("sthai.client.request", mock.handle)
    return mock


@pytest.fixture
def client() -> Client:
    """A client with a dummy key; combine with backend to serve responses."""
    return Client(api_key="test-key")
