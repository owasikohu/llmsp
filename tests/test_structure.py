"""Layer-3 single/multi/empty mode decision (heuristic + tree-sitter refinement)."""

import pytest

from llmsp.context.structure import EMPTY, MULTI, SINGLE, Structure

# Heuristic-only (tree-sitter disabled) — must work with no optional deps.
heuristic = Structure(enabled=False)


def _mode(s: Structure, src: str, marker: str, lang: str = "python") -> str:
    return s.decide_mode(src, src.index(marker) + len(marker), lang)


def test_heuristic_empty_on_midtoken():
    assert heuristic.decide_mode("foobar = 1\n", 3, "python") == EMPTY


def test_heuristic_single_when_code_follows_on_line():
    assert _mode(heuristic, "x = | + 1\n".replace("|", ""), "x = ") == SINGLE


def test_heuristic_multi_after_block_opener():
    assert _mode(heuristic, "def f():\n    ", "def f():\n    ") == MULTI


def test_heuristic_single_plain_statement():
    assert _mode(heuristic, "print(x)\nreturn ", "print(x)\n") in (SINGLE, MULTI)


# Tree-sitter refinement — skipped when the grammar pack isn't installed.
ts = Structure(enabled=True)
needs_ts = pytest.mark.skipif(
    not ts._ensure_loader(), reason="tree-sitter-language-pack not installed"
)


@needs_ts
def test_treesitter_downgrades_multi_inside_comment():
    # Comment line ending in ':' would be MULTI by heuristic; tree-sitter knows
    # we're in a comment and forces single-line.
    assert _mode(ts, "x = 1  # config:\n", "# config:") == SINGLE
    assert _mode(ts, "# TODO:\n", "# TODO:") == SINGLE


@needs_ts
def test_treesitter_downgrades_multi_inside_string():
    assert _mode(ts, 'msg = "a = " + y\n', 'msg = "a = ') == SINGLE


@needs_ts
def test_treesitter_keeps_multi_for_real_block_opener():
    assert _mode(ts, "def f():\n    \n", "def f():\n    ") == MULTI
