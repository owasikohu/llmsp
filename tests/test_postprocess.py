"""Completion post-processing — the FIM-failure-mode defenses."""

from llmsp.context.postprocess import (
    clean,
    dedupe_against_suffix,
    strip_sentinels,
    trim_to_bracket_balance,
)


def test_strip_sentinels():
    assert strip_sentinels("a<|fim_middle|>b", ["<|fim_middle|>"]) == "ab"


def test_single_line_cuts_at_newline():
    assert clean("a + b\nmore stuff\nyet more", single_line=True) == "a + b"


def test_single_line_skips_leading_newline():
    # FIM models often emit "\n<indent>content" — keep the content, not "".
    assert clean("\n    return x\nmore", single_line=True) == "    return x"
    assert clean("\n\nfoo()", single_line=True) == "foo()"


def test_multiline_trims_unbalanced_closers():
    # Model emitted an extra closing brace/paren that belongs to the suffix.
    assert trim_to_bracket_balance("foo(1, 2))") == "foo(1, 2)"
    assert trim_to_bracket_balance("a + b)") == "a + b"
    assert trim_to_bracket_balance("def f():\n    return (1 + 2)") == "def f():\n    return (1 + 2)"


def test_dedupe_overlap_with_suffix():
    # Completion re-types the start of what already follows the cursor.
    assert dedupe_against_suffix("return a + b)", ") + c") == "return a + b"


def test_dedupe_completion_is_prefix_of_suffix():
    assert dedupe_against_suffix("a + b", "a + b + c") == ""


def test_dedupe_noop_when_no_overlap():
    assert dedupe_against_suffix("xyz", "abc") == "xyz"


def test_clean_strips_then_dedupes():
    out = clean(
        "a + b<|fim_middle|>)",
        suffix=") + c",
        sentinels=["<|fim_middle|>"],
        single_line=False,
    )
    assert out == "a + b"


def test_clean_empty_input():
    assert clean("", single_line=True) == ""
