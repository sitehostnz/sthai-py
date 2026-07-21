# Changelog

All notable changes to this project will be documented in this file.

## [1.2.0] - 2026-07-21

### Changed

- **Breaking:** `embed()` and `batch_embed()` now return the full `EmbeddingResponse` instead of bare vectors, on both clients. Get the vectors via `output()` (one per input, so `output()[0]` for `embed()`), and token usage via `usage()` - previously usage on embedding calls was unrecoverable.
- **Breaking:** `rerank()` now returns the full `RerankResponse` instead of a `list[RerankResult]`, on both clients. Get the sorted results via `output()` (or `results`), and token usage via `usage()`.
- **Breaking:** `EmbeddingResponse.output()` now returns validated float vectors (`list[list[float]]`) and raises `sthai.ResponseError` for non-float encoding formats or when the response carries no embeddings. The raw entries stay available on `data`.

## [1.1.0] - 2026-07-20

### Added

- Session pin accessors on both clients: `session_pin()` returns the pin in use (constructor-passed, `auto_session`-generated, or set later; `None` when unpinned), `set_session_pin()` pins subsequent requests to a session (`None` unpins), and `new_session()` switches to a freshly generated pin and returns it.
- Chat history accessors: `history()` returns the stored turns as role/content dicts, oldest first, and `set_history()` replaces them, for restoring a persisted conversation. Invalid turns raise `InputError`.
- `write_history()` and `set_write_history()` read and toggle turn recording after construction. Turning recording off keeps the stored history and still sends it; only recording stops.
- `history_usage()` returns summed token usage across the calls that built the stored history. The tally resets with `clear_history()` and `set_history()`. Each call resends the conversation so far, so input tokens count what the server processed (as billed), not unique tokens.
- Added a package-specific exception hierarchy in `sthai.exceptions`, exported from the top-level package: `SthaiError` (root), `InputError`, `TransportError`, `APIStatusError` with `ClientError` (4xx) and `APIError` (5xx), and `ResponseError` with `ResponseParseError`. Catching `SthaiError` is enough to handle anything the client raises.
- HTTP error responses now parse the server's error body: `ClientError` and `APIError` carry `status_code`, `server_message`, `error_type` and the raw `response`.

### Changed

- **Breaking:** argument validation errors now raise `sthai.InputError` instead of `ValueError` or `TypeError`.
- **Breaking:** HTTP 4xx/5xx responses now raise `sthai.ClientError`/`sthai.APIError` instead of `niquests.exceptions.HTTPError`.
- **Breaking:** connection, timeout and TLS failures now raise `sthai.TransportError` instead of raw niquests exceptions (the original is preserved as `__cause__`).
- **Breaking:** `InferenceResponse.parse()` failures now raise `sthai.ResponseParseError` instead of `ValueError` or `msgspec.ValidationError`.
- Success responses whose bodies fail to decode now raise `sthai.ResponseError` instead of leaking msgspec exceptions.
