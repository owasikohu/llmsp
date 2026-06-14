"""Token counting and budget-aware, line-boundary truncation.

Uses :mod:`tiktoken` when installed for accurate counts; otherwise falls back
to a ``chars / 4`` heuristic, which is good enough for budgeting and keeps the
core dependency-free. Truncation always happens at line boundaries and away
from the cursor: the *prefix* keeps its tail (lines nearest the cursor) and the
*suffix* keeps its head.
"""

from __future__ import annotations

from typing import Optional


class TokenCounter:
    """Counts tokens, accurately with tiktoken or via a char heuristic."""

    def __init__(self, model: Optional[str] = None) -> None:
        self._enc = None
        try:  # optional dependency
            import tiktoken

            try:
                self._enc = tiktoken.encoding_for_model(model) if model else None
            except KeyError:
                self._enc = None
            if self._enc is None:
                self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    @property
    def exact(self) -> bool:
        """Whether counts are tokenizer-exact (vs. the char heuristic)."""
        return self._enc is not None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(text, disallowed_special=()))
        # Heuristic: ~4 chars/token, but never under-count short strings.
        return max(1, (len(text) + 3) // 4)


def fit_prefix(text: str, max_tokens: int, counter: TokenCounter) -> str:
    """Trim ``text`` to ``max_tokens``, keeping the END (nearest the cursor).

    Drops whole lines from the top until the budget fits.
    """
    if max_tokens <= 0:
        return ""
    if counter.count(text) <= max_tokens:
        return text
    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    total = 0
    for line in reversed(lines):
        c = counter.count(line)
        if total + c > max_tokens and kept:
            break
        kept.append(line)
        total += c
    return "".join(reversed(kept))


def fit_suffix(text: str, max_tokens: int, counter: TokenCounter) -> str:
    """Trim ``text`` to ``max_tokens``, keeping the START (nearest the cursor).

    Drops whole lines from the bottom until the budget fits.
    """
    if max_tokens <= 0:
        return ""
    if counter.count(text) <= max_tokens:
        return text
    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    total = 0
    for line in lines:
        c = counter.count(line)
        if total + c > max_tokens and kept:
            break
        kept.append(line)
        total += c
    return "".join(kept)
