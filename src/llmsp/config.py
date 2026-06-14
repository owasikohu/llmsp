"""Server configuration.

Settings arrive via the LSP ``initialize`` request's ``initialization_options``
(and may be refreshed live with ``workspace/configuration``). Everything has a
default so the server starts with zero config against the offline mock backend.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Optional


@dataclass
class ContextConfig:
    """Context-engineering pipeline settings."""

    max_prompt_tokens: int = 1536
    # Relative weights of the single-file budget (normalised). Prefix is favoured.
    prefix_ratio: float = 0.7
    suffix_ratio: float = 0.3
    # Layer 2 — cross-file retrieval. The reserve is additionally capped at half
    # of max_prompt_tokens by the assembler so the current file is never starved.
    cross_file: bool = True
    cross_file_tokens: int = 512
    retrieval: str = "jaccard"  # "jaccard" | "bm25" | "none"
    max_snippets: int = 4
    ring_chunks: int = 16
    ring_chunk_lines: int = 64
    # Layer 3 — structural (tree-sitter) context / single-vs-multi-line decision.
    structural: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ContextConfig":
        return _merge(cls(), data)


@dataclass
class Config:
    """Top-level server configuration."""

    backend: str = "mock"  # mock|ollama|deepseek|openai-instruct|codestral|llamacpp|vllm|tgi
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    model_family: str = ""  # overrides FIM-template inference from `model`
    fim_template: Optional[Mapping[str, Any]] = None  # explicit sentinel override

    max_tokens: int = 128
    temperature: float = 0.1
    stop: tuple[str, ...] = ()
    debounce_ms: int = 250
    request_timeout_ms: int = 2000
    multiline: str = "auto"  # "auto" | "single" | "multi"

    context: ContextConfig = field(default_factory=ContextConfig)

    # --- construction -------------------------------------------------------
    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "Config":
        cfg = cls()
        if not data:
            return cfg.with_env_defaults()
        ctx_data = data.get("context") if isinstance(data.get("context"), Mapping) else None
        cfg = _merge(cfg, {k: v for k, v in data.items() if k != "context"})
        if ctx_data:
            cfg.context = ContextConfig.from_mapping(ctx_data)
        # normalise list -> tuple for stop
        if isinstance(cfg.stop, list):
            cfg.stop = tuple(cfg.stop)
        return cfg.with_env_defaults()

    def with_env_defaults(self) -> "Config":
        """Fill an empty ``api_key`` from a backend-appropriate env var."""
        if self.api_key:
            return self
        env_key = {
            "deepseek": "DEEPSEEK_API_KEY",
            "codestral": "CODESTRAL_API_KEY",
            "openai-instruct": "OPENAI_API_KEY",
            "openai": "OPENAI_API_KEY",
        }.get(self.backend)
        key = ""
        if env_key:
            key = os.environ.get(env_key, "") or os.environ.get("LLMSP_API_KEY", "")
        return replace(self, api_key=key) if key else self

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stop"] = list(self.stop)
        return d


def _merge(obj: Any, data: Mapping[str, Any]) -> Any:
    """Shallow-merge known dataclass fields from ``data`` onto ``obj`` in place."""
    valid = {f for f in obj.__dataclass_fields__}  # type: ignore[attr-defined]
    for k, v in data.items():
        if k in valid and v is not None:
            setattr(obj, k, v)
    return obj
