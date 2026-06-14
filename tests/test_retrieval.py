"""Layer-2 lexical retrieval and ranking."""

from llmsp.context.retrieval import jaccard, rank, symbols
from llmsp.fim.base import Snippet


def test_symbols_drops_stopwords_and_short():
    s = symbols("def compute_total(items): return sum(items)")
    assert "compute_total" in s and "items" in s
    assert "def" not in s and "return" not in s  # stopwords excluded


def test_jaccard_bounds():
    assert jaccard(set(), {"a"}) == 0.0
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert 0 < jaccard({"a", "b"}, {"b", "c"}) < 1


def test_rank_orders_ascending_best_last():
    query = "compute_total items sum"
    cands = [
        Snippet("unrelated banana orchard", "far.py"),
        Snippet("def compute_total(items): return sum(items)", "near.py"),
        Snippet("compute_total helper", "mid.py"),
    ]
    out = rank(query, cands, k=3, method="jaccard")
    # ascending => the most relevant snippet is LAST (nearest the cursor).
    assert out[-1].path == "near.py"
    assert [s.score for s in out] == sorted(s.score for s in out)


def test_rank_respects_k():
    cands = [Snippet(f"sym{i} compute", f"f{i}.py") for i in range(10)]
    out = rank("compute", cands, k=3, method="jaccard")
    assert len(out) <= 3


def test_rank_method_none_returns_empty():
    cands = [Snippet("compute", "f.py")]
    assert rank("compute", cands, k=3, method="none") == []


def test_rank_drops_zero_score():
    cands = [Snippet("totally different words", "f.py")]
    assert rank("compute_total", cands, k=3, method="jaccard") == []
