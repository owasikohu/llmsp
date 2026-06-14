"""Clean a raw model completion before handing it to the editor.

Defends against the common FIM failure modes: leaked sentinel tokens, runaway
multi-line output, and duplicating code that already follows the cursor.
"""

from __future__ import annotations

from typing import Sequence

_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = {")": "(", "]": "[", "}": "{"}


def strip_sentinels(text: str, sentinels: Sequence[str]) -> str:
    for s in sentinels:
        if s:
            text = text.replace(s, "")
    return text


def trim_to_bracket_balance(text: str) -> str:
    """Cut the completion where it would close more brackets than it opened.

    A FIM model often emits a few trailing ``)``/``}`` that actually belong to
    the suffix. We stop at the first close that drops the running depth below
    zero, keeping a self-balanced fragment.
    """
    depth = 0
    out: list[str] = []
    for ch in text:
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            if depth == 0:
                break
            depth -= 1
        out.append(ch)
    return "".join(out)


def dedupe_against_suffix(text: str, suffix: str) -> str:
    """Drop a trailing overlap between the completion and the start of the suffix.

    If the completion ends with text that re-types the beginning of what already
    follows the cursor, remove that overlap. Also short-circuits the case where
    the whole completion is a prefix of the suffix (model "completed" by copying).
    """
    if not text or not suffix:
        return text
    suf = suffix.lstrip()
    stripped = text.rstrip()
    if suf and stripped and suf.startswith(stripped):
        return ""
    max_overlap = min(len(text), len(suffix))
    for n in range(max_overlap, 0, -1):
        if text[-n:] == suffix[:n]:
            return text[:-n]
    return text


def clean(
    text: str,
    *,
    suffix: str = "",
    sentinels: Sequence[str] = (),
    single_line: bool = False,
) -> str:
    """Full post-processing pipeline."""
    if not text:
        return ""
    text = strip_sentinels(text, sentinels)
    if single_line:
        # Keep one line of *content*. FIM models routinely emit a leading newline
        # (e.g. the cursor sits right after a block opener), so skip any leading
        # newlines first — otherwise we'd cut at index 0 and drop everything.
        text = text.lstrip("\n")
        idx = text.find("\n")
        if idx != -1:
            text = text[:idx]
    else:
        text = trim_to_bracket_balance(text)
    text = dedupe_against_suffix(text, suffix)
    return text
