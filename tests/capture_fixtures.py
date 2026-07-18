"""
Capture real server responses as test fixtures.

Not a test module: run it manually against the live API whenever the
fixtures need refreshing (e.g. after a server upgrade), then commit the
regenerated tests/fixtures/*.json files:

    env $(grep -v '^#' .env | xargs) PYTHONPATH=. uv run python tests/capture_fixtures.py

Each fixture records one request/response exchange: the endpoint, method,
decoded request body, response status, and response body. Scenarios are kept
cheap (low max_tokens, dimensions=32) - the suite asserts on shapes and
wiring, not on content quality.
"""

import json
import struct
import zlib
from pathlib import Path
from typing import Any

from msgspec import Struct
from niquests.exceptions import HTTPError
from niquests.models import Response

from sthai.client import Client
from sthai.typing import HttpMethod

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class CityInfo(Struct):
    """Small schema for the structured-response scenarios (mirrored in tests)."""

    name: str
    country: str
    population: int


def tiny_png(width: int = 4, height: int = 4) -> bytes:
    """Generate a minimal valid solid-red PNG for the multimodal scenario."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data))
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class RecordingClient(Client):
    """Client that records every exchange passing through _make_request."""

    records: list[dict[str, Any]] = []

    def _make_request(
        self, method: HttpMethod, endpoint: str, **kwargs: Any
    ) -> Response:
        response = super()._make_request(method, endpoint, **kwargs)
        request_body = kwargs.get("data")
        try:
            response_body = json.loads(response.content or b"")
        except json.JSONDecodeError:
            response_body = (response.content or b"").decode(errors="replace")
        self.records.append(
            {
                "endpoint": endpoint,
                "method": str(method),
                "status_code": response.status_code,
                "request": json.loads(request_body) if request_body else None,
                "response": response_body,
            }
        )
        return response


def save(name: str) -> None:
    """Write the most recent exchange to tests/fixtures/<name>.json."""
    record = RecordingClient.records[-1]
    path = FIXTURES_DIR / f"{name}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"{name}: {record['method']} {record['endpoint']} -> "
        f"{record['status_code']} ({path.stat().st_size} bytes)"
    )


def main() -> None:
    FIXTURES_DIR.mkdir(exist_ok=True)
    client = RecordingClient()

    client.healthy()
    save("health")

    client.models()
    save("models")

    client.chat("Reply with exactly: kia ora", max_tokens=50, use_history=False)
    save("chat_simple")

    client.chat(
        "What is 17 + 25? Answer with just the number.",
        max_tokens=400,
        use_thinking=True,
        use_history=False,
    )
    save("chat_thinking")

    client.response(
        "Give me basic facts about Wellington, New Zealand.",
        response_type=CityInfo,
        max_tokens=200,
    )
    save("response_struct")

    client.response(
        'Return a JSON object with a single key "answer" whose value is 42.',
        response_type=dict,
        max_tokens=100,
    )
    save("response_json_object")

    try:
        client.response(
            "Give me basic facts about Wellington, New Zealand.",
            response_type=CityInfo,
            max_tokens=10,
        )
    except ValueError as exc:
        print(f"  (expected truncation error: {exc})")
    save("response_truncated")

    client.embed("The Beehive is New Zealand's parliament building.", dimensions=32)
    save("embed_single")

    client.embed("A small red square.", image_files=[tiny_png()], dimensions=32)
    save("embed_multimodal")

    client.batch_embed(
        [
            "Wellington is the capital of New Zealand.",
            "Auckland is the largest city in New Zealand.",
            "The kiwi is a flightless bird.",
        ],
        dimensions=32,
    )
    save("batch_embed")

    documents = [
        "The capital of New Zealand is Wellington.",
        "Auckland has the largest population in New Zealand.",
        "Kiwi are nocturnal flightless birds native to New Zealand.",
        "The All Blacks are New Zealand's national rugby team.",
    ]
    client.rerank("What is the capital of New Zealand?", documents)
    save("rerank")

    client.rerank("What is the capital of New Zealand?", documents, top_n=2)
    save("rerank_top_n")

    try:
        client.chat("hello", model="does-not-exist", use_history=False)
    except HTTPError as exc:
        print(f"  (expected HTTP error: {exc})")
    save("error_bad_model")

    print(f"\nDone: {len(RecordingClient.records)} exchanges captured.")


if __name__ == "__main__":
    main()
