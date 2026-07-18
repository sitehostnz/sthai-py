"""Response structs for the /v1/models endpoint.

These mirror the vLLM 0.22.1 model list schema
(vllm/entrypoints/openai/engine/protocol.py). Response-side fields use plain
defaults because vLLM always serializes every field.
"""

from msgspec import Struct, field


class ModelPermission(Struct):
    id: str
    object: str = "model_permission"
    created: int = 0
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: str | None = None
    is_blocking: bool = False


class ModelCard(Struct):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "vllm"
    root: str | None = None
    parent: str | None = None
    max_model_len: int | None = None
    permission: list[ModelPermission] = field(default_factory=list)


class ModelList(Struct):
    object: str = "list"
    data: list[ModelCard] = field(default_factory=list)
