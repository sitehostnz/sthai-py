from typing import NamedTuple

from sthai.models import EmbeddingModel

INFERENCE_ENDPOINT = "/v1/chat/completions"
EMBEDDING_ENDPOINT = "/v1/embeddings"
RERANKING_ENDPOINT = "/v1/rerank"
MODELS_ENDPOINT = "/v1/models"
HEALTH_ENDPOINT = "/health"

SESSION_PIN_HEADER = "X-Session-Id"

# Chat templates applied locally for batched embedding requests, since only
# the plain-input request form batches and it bypasses the server-side
# template. Each matches what the model's own chat template renders for a
# single-turn request; the open assistant turn is intentional.
QWEN_3_VL_EMBEDDING_TEMPLATE = (
    "<|im_start|>system\n{instruction}<|im_end|>\n"
    "<|im_start|>user\n{text}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


# Recommended task instructions for the embedding model
EMBEDDING_DOCUMENT_INSTRUCTION = "Represent the user's input."
EMBEDDING_QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class EmbeddingParams(NamedTuple):
    """Client-side defaults for a known embedding model."""

    # Chat template applied locally for batched requests (see batch_embed)
    template: str | None = None
    # The model's native output dimension; a requested Matryoshka truncation
    # should divide evenly into this
    dimensions: int | None = None
    # Recommended task instructions: the default when embedding documents,
    # and the one to pass explicitly when embedding search queries
    document_instruction: str | None = None
    query_instruction: str | None = None


EMBEDDING_PARAMS: dict[str, EmbeddingParams] = {
    EmbeddingModel.QWEN_3_VL_8B: EmbeddingParams(
        template=QWEN_3_VL_EMBEDDING_TEMPLATE,
        dimensions=4096,
        document_instruction=EMBEDDING_DOCUMENT_INSTRUCTION,
        query_instruction=EMBEDDING_QUERY_INSTRUCTION,
    ),
}
