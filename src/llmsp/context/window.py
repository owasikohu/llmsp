"""Layer 1 — budgeted single-file windowing.

Splits the document at the cursor offset into prefix/suffix, then trims each to
its token budget at line boundaries, dropping from the prefix *top* and the
suffix *bottom* so the lines nearest the cursor always survive.
"""

from __future__ import annotations

from dataclasses import dataclass

from llmsp.tokenizer import TokenCounter, fit_prefix, fit_suffix


@dataclass(frozen=True)
class Window:
    prefix: str
    suffix: str


def split_budget(total: int, prefix_ratio: float, suffix_ratio: float) -> tuple[int, int]:
    """Split a token budget into (prefix, suffix) by normalised weights."""
    total = max(0, total)
    denom = prefix_ratio + suffix_ratio
    if denom <= 0:
        return total // 2, total - total // 2
    pre = int(round(total * (prefix_ratio / denom)))
    pre = max(0, min(total, pre))
    return pre, total - pre


def build_window(
    source: str,
    offset: int,
    prefix_budget: int,
    suffix_budget: int,
    counter: TokenCounter,
) -> Window:
    """Build the budgeted prefix/suffix around ``offset`` in ``source``."""
    offset = max(0, min(len(source), offset))
    prefix = fit_prefix(source[:offset], prefix_budget, counter)
    suffix = fit_suffix(source[offset:], suffix_budget, counter)
    return Window(prefix=prefix, suffix=suffix)
