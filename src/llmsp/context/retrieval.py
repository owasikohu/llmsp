"""Layer 2 ranking — pick the most relevant cross-file snippets for the cursor.

Default ranker is Jaccard over identifier sets (cheap, dependency-free, and what
Tabby/Continue use as their lexical baseline). BM25 is available when
``rank-bm25`` is installed and ``method="bm25"``. Either way the caller packs the
result *ascending* so the best snippet ends up nearest the cursor.
"""

from __future__ import annotations

import re
from typing import List, Sequence

from llmsp.fim.base import Snippet

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Identifiers that carry no retrieval signal.
_STOP = frozenset(
    {
        "the", "and", "for", "def", "return", "if", "else", "elif", "import",
        "from", "class", "self", "None", "True", "False", "in", "is", "not",
        "while", "with", "as", "try", "except", "finally", "pass", "function",
        "const", "let", "var", "this", "new", "public", "private", "static",
    }
)


def symbols(text: str) -> set:
    """Extract the set of meaningful identifiers from ``text``."""
    return {t for t in _IDENT.findall(text or "") if len(t) > 1 and t not in _STOP}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _rank_jaccard(query: set, candidates: Sequence[Snippet]) -> List[Snippet]:
    scored = [Snippet(c.text, c.path, jaccard(query, symbols(c.text))) for c in candidates]
    return [s for s in scored if s.score > 0.0]


def _rank_bm25(query_text: str, candidates: Sequence[Snippet]) -> List[Snippet]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return _rank_jaccard(symbols(query_text), candidates)
    corpus = [list(symbols(c.text)) for c in candidates]
    if not any(corpus):
        return []
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(list(symbols(query_text)))
    out = [Snippet(c.text, c.path, float(s)) for c, s in zip(candidates, scores) if s > 0]
    return out


def rank(
    query_text: str,
    candidates: Sequence[Snippet],
    *,
    k: int,
    method: str = "jaccard",
) -> List[Snippet]:
    """Return the top-``k`` snippets, ordered ASCENDING (best last)."""
    if not candidates or k <= 0 or method == "none":
        return []
    if method == "bm25":
        scored = _rank_bm25(query_text, candidates)
    else:
        scored = _rank_jaccard(symbols(query_text), candidates)
    scored.sort(key=lambda s: s.score, reverse=True)
    top = scored[:k]
    # Pack ascending so the most relevant snippet sits closest to the prefix.
    top.sort(key=lambda s: s.score)
    return top
