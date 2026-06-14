"""Core FIM data types and the backend interface.

Every backend implements one method, :meth:`FIMBackend.complete`, which takes a
:class:`FimRequest` (prefix, suffix, cross-file snippets and generation knobs)
and yields the *middle* text as a stream of chunks. Streaming lets the server
cancel an in-flight request the moment the user keeps typing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Snippet:
    """A cross-file context snippet injected ahead of the FIM prefix.

    ``path`` is a repo-relative label used either for repo-level FIM tokens
    (``<file_sep>{path}``) or as a comment header when folded into a flat
    prefix. ``score`` is the retrieval relevance (higher is better); snippets
    are packed *ascending* so the most relevant sits nearest the cursor.
    """

    text: str
    path: str = ""
    score: float = 0.0


@dataclass(frozen=True)
class Knobs:
    """Generation knobs shared across backends."""

    max_tokens: int = 64
    temperature: float = 0.1
    stop: tuple[str, ...] = ()
    stream: bool = True
    timeout_ms: int = 2000


@dataclass(frozen=True)
class FimRequest:
    """Everything a backend needs to produce a completion."""

    prefix: str
    suffix: str
    extra: tuple[Snippet, ...] = ()
    knobs: Knobs = field(default_factory=Knobs)
    language_id: str = ""


@runtime_checkable
class FIMBackend(Protocol):
    """A provider of FIM completions.

    Implementations must:

    * respect ``req.knobs`` (max_tokens / temperature / stop / stream);
    * be cancellable — when the awaiting task is cancelled they must let
      :class:`asyncio.CancelledError` propagate (closing any open HTTP stream);
    * decide how to use ``req.extra`` — backends with native repo support pass
      it structurally, "flat" backends fold it via :func:`flatten_prefix`.
    """

    name: str

    def complete(self, req: FimRequest) -> AsyncIterator[str]:
        """Yield completion ("middle") text chunks for ``req``."""
        ...

    async def aclose(self) -> None:
        """Release any held resources (HTTP clients, etc.)."""
        ...


async def collect(stream: AsyncIterator[str]) -> str:
    """Drain a completion stream into a single string."""
    parts: list[str] = []
    async for chunk in stream:
        parts.append(chunk)
    return "".join(parts)


# --- folding cross-file snippets into a flat prefix -------------------------

# Line-comment token per LSP language id; used to label folded snippets so they
# read like real code to the model rather than opaque blobs.
_LINE_COMMENT = {
    "python": "#",
    "ruby": "#",
    "shellscript": "#",
    "bash": "#",
    "sh": "#",
    "yaml": "#",
    "toml": "#",
    "dockerfile": "#",
    "makefile": "#",
    "r": "#",
    "perl": "#",
    "javascript": "//",
    "javascriptreact": "//",
    "typescript": "//",
    "typescriptreact": "//",
    "c": "//",
    "cpp": "//",
    "objective-c": "//",
    "csharp": "//",
    "java": "//",
    "go": "//",
    "rust": "//",
    "php": "//",
    "kotlin": "//",
    "swift": "//",
    "scala": "//",
    "dart": "//",
    "zig": "//",
    "lua": "--",
    "haskell": "--",
    "elm": "--",
    "sql": "--",
}


def line_comment(language_id: str) -> str:
    """Return the line-comment token for ``language_id`` (default ``#``)."""
    return _LINE_COMMENT.get((language_id or "").lower(), "#")


def flatten_prefix(req: FimRequest) -> str:
    """Prepend ``req.extra`` to the prefix as comment-labelled code blocks.

    Used by backends that accept only ``prompt``/``suffix`` (Ollama, the OpenAI
    suffix endpoint, Codestral). Snippets arrive already ordered ascending by
    relevance, so the closest-to-cursor (best) one ends up nearest the prefix.
    """
    if not req.extra:
        return req.prefix
    cmt = line_comment(req.language_id)
    blocks: list[str] = []
    for sn in req.extra:
        header = f"{cmt} {sn.path}".rstrip() if sn.path else f"{cmt} context"
        blocks.append(f"{header}\n{sn.text.rstrip()}\n")
    return "\n".join(blocks) + "\n" + req.prefix


def join_stops(*groups: Sequence[str]) -> tuple[str, ...]:
    """Merge several stop-sequence groups, dropping blanks and duplicates."""
    seen: dict[str, None] = {}
    for group in groups:
        for s in group or ():
            if s and s not in seen:
                seen[s] = None
    return tuple(seen)
