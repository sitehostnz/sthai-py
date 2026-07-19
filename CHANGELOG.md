# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Added a package-specific exception hierarchy in `sthai.exceptions`, exported from the top-level package: `SthaiError` (root), `InputError`, `TransportError`, `APIStatusError` with `ClientError` (4xx) and `APIError` (5xx), and `ResponseError` with `ResponseParseError`. Catching `SthaiError` is enough to handle anything the client raises.
- HTTP error responses now parse the server's error body: `ClientError` and `APIError` carry `status_code`, `server_message`, `error_type` and the raw `response`.

### Changed

- **Breaking:** argument validation errors now raise `sthai.InputError` instead of `ValueError` or `TypeError`.
- **Breaking:** HTTP 4xx/5xx responses now raise `sthai.ClientError`/`sthai.APIError` instead of `niquests.exceptions.HTTPError`.
- **Breaking:** connection, timeout and TLS failures now raise `sthai.TransportError` instead of raw niquests exceptions (the original is preserved as `__cause__`).
- **Breaking:** `InferenceResponse.parse()` failures now raise `sthai.ResponseParseError` instead of `ValueError` or `msgspec.ValidationError`.
- Success responses whose bodies fail to decode now raise `sthai.ResponseError` instead of leaking msgspec exceptions.
