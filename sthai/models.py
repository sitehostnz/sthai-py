from sthai.typing import StrEnum


class InferenceModel(StrEnum):
    QWEN_3_6_27B = "Qwen/Qwen3.6-27B"


class EmbeddingModel(StrEnum):
    QWEN_3_VL_8B = "Qwen/Qwen3-VL-Embedding-8B"


class RerankingModel(StrEnum):
    QWEN_3_VL_8B = "Qwen/Qwen3-VL-Reranker-8B"
