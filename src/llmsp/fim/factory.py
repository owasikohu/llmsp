"""Build a :class:`FIMBackend` and resolve its FIM template from config.

Imports of optional, backend-specific dependencies (``openai``) are done lazily
inside each branch so that, e.g., the Ollama or mock path never requires them.
"""

from __future__ import annotations

from typing import Optional

from llmsp.config import Config
from llmsp.fim.base import FIMBackend
from llmsp.fim.templates import FimTemplate, custom_template, template_for


def resolve_template(cfg: Config) -> Optional[FimTemplate]:
    """Resolve the FIM template: explicit override → family → infer from model.

    Returns ``None`` when nothing matches; that is fine for convenience backends
    (Ollama / llama.cpp / DeepSeek / Codestral) which template server-side. The
    template is still useful for deriving stop sentinels and cleaning leaked
    tokens from output.
    """
    if cfg.fim_template:
        t = cfg.fim_template
        return custom_template(
            t["fim_prefix"],
            t["fim_suffix"],
            t["fim_middle"],
            order=t.get("order", "psm"),
            stop=t.get("stop", ()),
            file_sep=t.get("file_sep"),
            repo_name=t.get("repo_name"),
        )
    return template_for(cfg.model_family or None, cfg.model)


def build_backend(cfg: Config) -> FIMBackend:
    """Instantiate the backend named by ``cfg.backend``."""
    name = (cfg.backend or "mock").lower()

    if name == "mock":
        from llmsp.fim.backends.mock import MockBackend

        return MockBackend()

    if name == "ollama":
        from llmsp.fim.backends.ollama import OllamaBackend

        return OllamaBackend(
            base_url=cfg.base_url or "http://localhost:11434",
            model=cfg.model or "qwen2.5-coder",
        )

    if name in ("deepseek", "openai-instruct", "openai"):
        from llmsp.fim.backends.openai_suffix import (
            DEEPSEEK_BASE_URL,
            OpenAISuffixBackend,
        )

        if name == "deepseek":
            base_url = cfg.base_url or DEEPSEEK_BASE_URL
            model = cfg.model or "deepseek-coder"
        else:
            base_url = cfg.base_url or None
            model = cfg.model or "gpt-3.5-turbo-instruct"
        return OpenAISuffixBackend(model=model, api_key=cfg.api_key, base_url=base_url)

    if name == "codestral":
        from llmsp.fim.backends.codestral import CODESTRAL_BASE_URL, CodestralBackend

        return CodestralBackend(
            api_key=cfg.api_key,
            model=cfg.model or "codestral-2508",
            base_url=cfg.base_url or CODESTRAL_BASE_URL,
        )

    if name == "llamacpp":
        from llmsp.fim.backends.llamacpp import LlamaCppInfillBackend

        return LlamaCppInfillBackend(base_url=cfg.base_url or "http://localhost:8080")

    if name in ("vllm", "tgi", "raw", "raw-template"):
        from llmsp.fim.backends.raw_template import RawTemplateBackend

        template = resolve_template(cfg)
        if template is None:
            raise ValueError(
                f"backend {name!r} needs a FIM template; set 'model_family' "
                "(e.g. qwen/starcoder2/deepseek/codellama) or 'fim_template'."
            )
        if not cfg.base_url:
            raise ValueError(f"backend {name!r} requires 'base_url'.")
        return RawTemplateBackend(
            base_url=cfg.base_url,
            model=cfg.model or "default",
            template=template,
            api_key=cfg.api_key,
        )

    raise ValueError(
        f"unknown backend {cfg.backend!r}; expected one of: mock, ollama, "
        "deepseek, openai-instruct, codestral, llamacpp, vllm, tgi."
    )
