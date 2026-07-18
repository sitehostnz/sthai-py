"""Module-level helpers: image_content format sniffing and data URIs."""

from base64 import b64encode
from pathlib import Path

import pytest

from sthai.client import image_content

PNG = b"\x89PNG\r\n\x1a\nrest-of-file"
JPEG = b"\xff\xd8\xff\xe0rest-of-file"
GIF = b"GIF89arest-of-file"
WEBP = b"RIFF\x00\x00\x00\x00WEBPrest-of-file"


def test_url_string_passes_through() -> None:
    part = image_content("https://example.com/pic.png")
    assert part.image_url.url == "https://example.com/pic.png"


@pytest.mark.parametrize(
    ("data", "mime"),
    [
        (PNG, "image/png"),
        (JPEG, "image/jpeg"),
        (GIF, "image/gif"),
        (WEBP, "image/webp"),
    ],
)
def test_magic_bytes_to_data_uri(data: bytes, mime: str) -> None:
    part = image_content(data)
    expected = f"data:{mime};base64,{b64encode(data).decode('ascii')}"
    assert part.image_url.url == expected


def test_path_input_read_from_disk(tmp_path: Path) -> None:
    image_path = tmp_path / "pic.png"
    image_path.write_bytes(PNG)
    assert image_content(image_path).image_url.url == image_content(PNG).image_url.url


def test_unrecognized_format_raises() -> None:
    with pytest.raises(ValueError, match="unrecognized image format"):
        image_content(b"plain text, not an image")
