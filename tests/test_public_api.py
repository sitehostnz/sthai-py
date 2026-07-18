"""The package's public surface: what `from sthai import ...` provides."""

import sthai


def test_all_names_are_importable() -> None:
    for name in sthai.__all__:
        assert getattr(sthai, name) is not None


def test_public_surface() -> None:
    # The names the README examples rely on
    assert set(sthai.__all__) == {
        "Client",
        "EmbeddingModel",
        "InferenceModel",
        "RerankingModel",
        "image_content",
    }
