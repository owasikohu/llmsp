"""Per-model FIM sentinel registry.

Only the *raw-template* backend (vLLM / TGI, which do not honour a ``suffix``
parameter) needs to render these by hand. Convenience backends — Ollama,
llama.cpp ``/infill``, DeepSeek ``/beta``, Codestral ``/v1/fim/completions`` —
apply the correct template server-side, so they take plain prefix/suffix.

The sentinel strings are exact and must be copied byte-for-byte. In particular
DeepSeek uses the full-width bar ``｜`` (U+FF5C) and ``▁`` (U+2581), Qwen and
StarCoder2 differ only by pipe-wrapping (``<|fim_prefix|>`` vs ``<fim_prefix>``).
All modern code models use PSM (Prefix, Suffix, Middle) ordering; CodeLlama also
supports SPM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from llmsp.fim.base import Snippet


@dataclass(frozen=True)
class FimTemplate:
    """A model family's FIM special tokens and assembly rules."""

    family: str
    fim_prefix: str
    fim_suffix: str
    fim_middle: str
    order: str = "psm"  # "psm" | "spm"
    stop: tuple[str, ...] = ()
    file_sep: Optional[str] = None  # repo-level snippet separator
    repo_name: Optional[str] = None  # repo-level repo-name marker

    @property
    def supports_repo(self) -> bool:
        return self.file_sep is not None

    def render(
        self,
        prefix: str,
        suffix: str,
        extra: Sequence[Snippet] = (),
        repo: str = "",
    ) -> str:
        """Render the full FIM prompt string for a raw completion endpoint."""
        head = ""
        if extra and self.supports_repo:
            parts: list[str] = []
            if self.repo_name:
                parts.append(f"{self.repo_name}{repo or 'repo'}")
            for sn in extra:
                label = sn.path or "snippet"
                parts.append(f"{self.file_sep}{label}\n{sn.text.rstrip()}")
            # Begin the file being completed.
            parts.append(f"{self.file_sep}")
            head = "\n".join(parts) + "\n"
        if self.order == "spm":
            body = f"{self.fim_suffix}{suffix}{self.fim_prefix}{prefix}{self.fim_middle}"
        else:
            body = f"{self.fim_prefix}{prefix}{self.fim_suffix}{suffix}{self.fim_middle}"
        return head + body

    def sentinels(self) -> tuple[str, ...]:
        """All sentinel strings (for stripping leaked tokens from output)."""
        toks = [self.fim_prefix, self.fim_suffix, self.fim_middle]
        if self.file_sep:
            toks.append(self.file_sep)
        if self.repo_name:
            toks.append(self.repo_name)
        toks.extend(self.stop)
        return tuple(t for t in toks if t)


# DeepSeek-Coder: full-width bar U+FF5C and U+2581 — copy exactly.
_DEEPSEEK = FimTemplate(
    family="deepseek",
    fim_prefix="<｜fim▁begin｜>",
    fim_suffix="<｜fim▁hole｜>",
    fim_middle="<｜fim▁end｜>",
    order="psm",
    stop=("<｜end▁of▁sentence｜>",),
)

_QWEN = FimTemplate(
    family="qwen",
    fim_prefix="<|fim_prefix|>",
    fim_suffix="<|fim_suffix|>",
    fim_middle="<|fim_middle|>",
    order="psm",
    stop=("<|endoftext|>", "<|fim_pad|>"),
    file_sep="<|file_sep|>",
    repo_name="<|repo_name|>",
)

_STARCODER2 = FimTemplate(
    family="starcoder2",
    fim_prefix="<fim_prefix>",
    fim_suffix="<fim_suffix>",
    fim_middle="<fim_middle>",
    order="psm",
    stop=("<|endoftext|>", "<fim_pad>"),
    file_sep="<file_sep>",
    repo_name="<repo_name>",
)

_STARCODER = FimTemplate(
    family="starcoder",
    fim_prefix="<fim_prefix>",
    fim_suffix="<fim_suffix>",
    fim_middle="<fim_middle>",
    order="psm",
    stop=("<|endoftext|>",),
)

_CODELLAMA = FimTemplate(
    family="codellama",
    fim_prefix="<PRE> ",
    fim_suffix=" <SUF>",
    fim_middle=" <MID>",
    order="psm",
    stop=("<EOT>", "<｜end▁of▁sentence｜>"),
)


_REGISTRY: dict[str, FimTemplate] = {
    "deepseek": _DEEPSEEK,
    "deepseek-coder": _DEEPSEEK,
    "qwen": _QWEN,
    "qwen2": _QWEN,
    "qwen2.5-coder": _QWEN,
    "qwen3-coder": _QWEN,
    "starcoder2": _STARCODER2,
    "starcoder": _STARCODER,
    "starcoderbase": _STARCODER,
    "codellama": _CODELLAMA,
}


def get_template(family: str) -> FimTemplate:
    """Look up a template by family name (case-insensitive). Raises on unknown."""
    key = (family or "").strip().lower()
    if key in _REGISTRY:
        return _REGISTRY[key]
    raise KeyError(
        f"unknown FIM model family {family!r}; known families: "
        f"{sorted(set(t.family for t in _REGISTRY.values()))}. "
        "Pass an explicit fim_template in config to override."
    )


def custom_template(
    fim_prefix: str,
    fim_suffix: str,
    fim_middle: str,
    *,
    order: str = "psm",
    stop: Sequence[str] = (),
    file_sep: Optional[str] = None,
    repo_name: Optional[str] = None,
) -> FimTemplate:
    """Build a template from explicit sentinels (config ``fim_template`` override)."""
    return FimTemplate(
        family="custom",
        fim_prefix=fim_prefix,
        fim_suffix=fim_suffix,
        fim_middle=fim_middle,
        order=order,
        stop=tuple(stop),
        file_sep=file_sep,
        repo_name=repo_name,
    )


def infer_family(model: str) -> Optional[str]:
    """Best-effort guess of the FIM family from a model name string."""
    m = (model or "").lower()
    # Order matters: check more specific names first.
    if "deepseek" in m:
        return "deepseek"
    if "qwen" in m:
        return "qwen"
    if "starcoder2" in m:
        return "starcoder2"
    if "starcoder" in m:
        return "starcoder"
    if "codellama" in m or "code-llama" in m:
        return "codellama"
    return None


def template_for(family: Optional[str], model: str) -> Optional[FimTemplate]:
    """Resolve a template from an explicit family or by inferring from ``model``.

    Returns ``None`` when nothing matches (e.g. an unknown Ollama model that
    nonetheless applies its template server-side).
    """
    fam = family or infer_family(model)
    if not fam:
        return None
    try:
        return get_template(fam)
    except KeyError:
        return None
