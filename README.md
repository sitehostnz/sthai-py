# sthai-py

[![Tests](https://github.com/sitehostnz/sthai-py/actions/workflows/tests.yml/badge.svg)](https://github.com/sitehostnz/sthai-py/actions/workflows/tests.yml)

A Python client for the [SiteHost AI Platform](https://kb.sitehost.nz/ai-platform): inference, embeddings and reranking with typed requests and responses built on [msgspec](https://jcristharif.com/msgspec/).
Check out our [PHP client port](https://github.com/sitehostnz/sthai-php) as well!

## The SiteHost AI Platform

The [SiteHost AI Platform](https://kb.sitehost.nz/ai-platform) serves capable open-weight models at their full context windows - not heavily quantised cut-downs - with full control over system prompts and outputs. Everything runs on SiteHost's own hardware in their own New Zealand data centres, so your data never leaves the country, and request bodies are never stored: only usage metrics are kept for billing and performance monitoring.

Models are stable targets, too: each served model has a minimum one-year retention window and at least three months' deprecation notice, with guidance on any adjustments needed.

The platform currently serves three models, one per capability (see the [models page](https://kb.sitehost.nz/ai-platform/models) for the source of truth):

| Model | Purpose | Context window |
|-------|---------|----------------|
| `Qwen/Qwen3.6-27B` | Inference (chat, multimodal, thinking) | 262K |
| `Qwen/Qwen3-VL-Embedding-8B` | Embeddings (multimodal, Matryoshka, 4096 dims) | 32K |
| `Qwen/Qwen3-VL-Reranker-8B` | Reranking (multimodal, instruction-trained) | 32K |

## Installation

Requires Python 3.10+. Install from [PyPI](https://pypi.org/project/sthai/):

```bash
pip install sthai
# or, in a uv project
uv add sthai
```

## Getting started

Create an API key in the SiteHost Control Panel (see [API keys](https://kb.sitehost.nz/ai-platform/api-keys)). The client reads it from the `STHAI_KEY` environment variable, or you can pass `api_key=` explicitly:

```python
from sthai import Client

client = Client()  # or Client(api_key="...")
response = client.chat("What's the tallest mountain in New Zealand?")
print(response.output().text)
```

## Usage

### Chat and history

`chat()` keeps a conversation going: each successful call appends the user and assistant turns to the client's history, and later calls send it back. The system prompt is applied per call rather than stored.

```python
client.chat("I'm planning a tramping trip to Fiordland.")
client.chat("What should I pack?")  # the model sees the earlier turn

client.chat("Standalone question.", use_history=False)  # neither sends nor records
client.clear_history()

client.chat("Be brief: why is the sky blue?", system_prompt="You are terse.")
```

Thinking models can reason before answering; the reasoning rides along on the response:

```python
response = client.chat("What is 17 * 23?", use_thinking=True, max_tokens=2000)
print(response.output().reasoning)  # or client.last_reasoning()
print(response.output().text)
```

### Images

Chat and embedding inputs can include images, given as URLs or as local files/bytes (PNG, JPEG, GIF or WEBP - files are inlined as data URIs):

```python
from pathlib import Path

client.chat("What's in this image?", image_files=[Path("photo.png")])
client.chat("Compare these.", image_urls=["https://example.com/a.jpg", "https://example.com/b.jpg"])
```

### One-off and structured responses

`response()` mirrors `chat()` without the back-and-forth: the stored history is neither sent nor updated. Pass `response_type=` to get structured output - a msgspec Struct becomes a JSON schema the server enforces during generation, and the decoded, validated instance is returned:

```python
from msgspec import Struct

class CityInfo(Struct):
    name: str
    country: str
    population: int

city = client.response(
    "Give me basic facts about Wellington.",
    response_type=CityInfo,
)
print(city.population)

data = client.response("List three NZ birds as JSON.", response_type=dict)
```

If the output is cut off by the token limit, or (rarely, with thinking enabled) the server skips the schema, parsing raises a `ValueError` naming the cause.

### Embeddings

`embed()` turns one input - text, images, or both - into a single vector. The embedding model is instruction-trained: document embedding is the default, and `query=True` switches to the query instruction for search-style lookups. `dimensions=` truncates the vector server-side (Matryoshka - powers of two work best):

```python
vector = client.embed("The Beehive is New Zealand's parliament building.")
query_vector = client.embed("Where does NZ parliament sit?", query=True)
small = client.embed("Compact vector, please.", dimensions=512)
```

`batch_embed()` embeds many texts in one request, returning one vector per text in order:

```python
vectors = client.batch_embed([
    "Wellington is the capital of New Zealand.",
    "Auckland is the largest city in New Zealand.",
])
```

### Reranking

`rerank()` scores each document against a query and returns results sorted by relevance, with each result's `index` mapping back to your input list:

```python
results = client.rerank(
    "What is the capital of New Zealand?",
    [
        "The capital of New Zealand is Wellington.",
        "Auckland has the largest population in New Zealand.",
        "The All Blacks are New Zealand's national rugby team.",
    ],
    top_n=2,
)
for result in results:
    print(result.relevance_score, result.document.text)
```

Pass `instruction=` to steer relevance for a specific task; the model applies a sensible default otherwise.

### Response helpers

Every response type has `usage()` (input/output/cached token counts) and `output()` (the useful payload). The full struct from the most recent inference call is available via `last_response()`:

```python
response = client.chat("Hello!")
print(response.usage().input_tokens, response.usage().output_tokens)
print(client.last_response().model)
```

### Sessions

Pinning requests to a server session keeps routing consistent and helps caching. Pass `session_pin=` with your own identifier, or let the client generate one:

```python
client = Client(auto_session=True)
```

### Models and health

```python
client.healthy()  # True if the server is up
for card in client.models():
    print(card.id)
```

### Async

`AsyncClient` mirrors `Client` method-for-method, with every network method awaitable. Like `Client` it is session-less: construct it and go - there is no `async with` or `close()` to manage.

```python
import asyncio
from sthai import AsyncClient

async def main() -> None:
    client = AsyncClient()
    response = await client.chat("What's the tallest mountain in New Zealand?")
    print(response.output().text)

asyncio.run(main())
```

Async is most useful for concurrent fan-out, such as `asyncio.gather` over many `embed()` or `rerank()` calls.

## Development

Development uses [uv](https://docs.astral.sh/uv/). The test suite runs entirely offline against fixtures captured from the live API:

```bash
git clone https://github.com/sitehostnz/sthai-py.git
cd sthai-py
uv sync
uv run pytest
uv run ruff check sthai/ tests/
uv run ruff format --check sthai/ tests/
uv run ty check sthai/ tests/
```

To refresh the fixtures against the live API (costs tokens, needs a real key), see `tests/capture_fixtures.py`.

## Licence

[MIT](LICENSE)
