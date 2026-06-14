"""Token counting and budgeted, line-boundary windowing."""

from llmsp.context.window import build_window, split_budget
from llmsp.tokenizer import TokenCounter, fit_prefix, fit_suffix


def test_counter_counts_something():
    c = TokenCounter()
    assert c.count("") == 0
    assert c.count("hello world") >= 1


def test_fit_prefix_keeps_tail_lines():
    c = TokenCounter()
    text = "".join(f"line{i}\n" for i in range(100))
    out = fit_prefix(text, max_tokens=5, counter=c)
    # Keeps the END of the prefix (lines nearest the cursor).
    assert out.endswith("line99\n")
    assert "line0\n" not in out
    assert c.count(out) <= c.count(text)


def test_fit_suffix_keeps_head_lines():
    c = TokenCounter()
    text = "".join(f"line{i}\n" for i in range(100))
    out = fit_suffix(text, max_tokens=5, counter=c)
    # Keeps the START of the suffix (lines nearest the cursor).
    assert out.startswith("line0\n")
    assert "line99" not in out


def test_fit_keeps_at_least_one_line():
    c = TokenCounter()
    huge = "x" * 10000 + "\n"
    assert fit_prefix(huge, max_tokens=1, counter=c) != ""
    assert fit_suffix(huge, max_tokens=1, counter=c) != ""


def test_split_budget_weights_favor_prefix():
    pre, suf = split_budget(1000, 0.7, 0.3)
    assert pre == 700 and suf == 300
    pre, suf = split_budget(0, 0.7, 0.3)
    assert pre == 0 and suf == 0


def test_build_window_splits_at_offset():
    c = TokenCounter()
    src = "AAAA\nBBBB\nCCCC\nDDDD\n"
    off = src.index("CCCC")
    w = build_window(src, off, 100, 100, c)
    assert w.prefix == "AAAA\nBBBB\n"
    assert w.suffix == "CCCC\nDDDD\n"
