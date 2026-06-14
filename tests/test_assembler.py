"""The context-assembly pipeline (L1 + L2 + mode decision)."""

from llmsp.config import Config
from llmsp.context.assembler import ContextAssembler
from llmsp.context.structure import EMPTY, MULTI
from llmsp.fim.base import Snippet


def _assembler(**ctx):
    cfg = Config.from_mapping({"backend": "mock", "context": {"cross_file_tokens": 300, **ctx}})
    return ContextAssembler(cfg)


def test_window_split_at_cursor():
    asm = _assembler(cross_file=False)
    src = "import os\n\nx = \nprint(x)\n"
    off = src.index("x = ") + len("x = ")
    built = asm.build(source=src, offset=off, language_id="python")
    assert built.prefix.endswith("x = ")
    assert built.suffix.startswith("\nprint(x)")
    assert built.extra == ()


def test_cross_file_selects_relevant_snippet():
    asm = _assembler(cross_file=True)
    src = "from helpers import compute_total\n\nresult = compute_total(\n"
    off = len(src)
    other = [
        ("file:///far.py", "def banana(): return 'fruit'\n"),
        ("file:///near.py", "def compute_total(items):\n    return sum(items)\n"),
    ]
    built = asm.build(
        source=src, offset=off, language_id="python",
        current_path="file:///main.py", open_docs=other,
    )
    assert built.extra, "expected at least one cross-file snippet"
    # Most relevant snippet is packed last (nearest the cursor).
    assert "compute_total" in built.extra[-1].text


def test_cross_file_dedupes_content_already_in_window():
    asm = _assembler(cross_file=True)
    body = "def compute_total(items):\n    return sum(items)"
    src = body + "\n\nresult = compute_total(\n"  # the def is already in the prefix
    off = len(src)
    built = asm.build(
        source=src, offset=off, language_id="python",
        current_path="file:///main.py",
        open_docs=[("file:///dup.py", body + "\n")],
    )
    # The snippet duplicates what's already in the prefix → excluded.
    assert all("def compute_total(items):" not in s.text.splitlines()[0] for s in built.extra)


def test_empty_mode_midtoken_yields_no_context():
    asm = _assembler(cross_file=True)
    src = "foobar = 1\n"
    off = src.index("foobar") + 3  # cursor inside the identifier
    built = asm.build(source=src, offset=off, language_id="python")
    assert built.mode == EMPTY
    assert built.extra == ()


def test_multiline_mode_after_block_opener():
    asm = _assembler(cross_file=False)
    src = "def f():\n    \n"
    off = src.index("def f():\n    ") + len("def f():\n    ")
    built = asm.build(source=src, offset=off, language_id="python")
    assert built.mode == MULTI


def test_window_not_starved_when_no_cross_file_candidates():
    # Regression: with cross_file on but no open docs/ring, the reserve must not
    # shrink (let alone zero) the single-file window.
    cfg = Config.from_mapping({"backend": "mock"})  # defaults: cross_file on
    asm = ContextAssembler(cfg)
    src = "\n".join(f"line_{i} = {i}" for i in range(50)) + "\nx = \n"
    off = src.rindex("x = ") + len("x = ")
    built = asm.build(
        source=src, offset=off, language_id="python",
        current_path="file:///main.py", open_docs=[],
    )
    assert built.extra == ()
    assert built.prefix.count("\n") > 10  # got a real window, not a starved one


def test_dedupe_drops_body_duplicate_not_only_first_line():
    # First line differs, but the body duplicates the suffix -> must be dropped.
    snip = Snippet("# helper comment\n    return x + y\n    print(done)", "h.py")
    out = ContextAssembler._dedupe([snip], "", "    return x + y\n    print(done)\n")
    assert out == []


def test_dedupe_keeps_snippet_with_only_short_token_overlap():
    # Unanchored substring matching previously dropped this spuriously.
    snip = Snippet("def compute_area(r):\n    return 3 * r * r", "c.py")
    out = ContextAssembler._dedupe([snip], "identifier = make_identifier()\nwhile True:\n", "")
    assert len(out) == 1


def test_cross_file_budget_is_respected():
    asm = _assembler(cross_file=True, cross_file_tokens=20, max_snippets=10)
    src = "use_thing(\n"
    big = [
        (f"file:///f{i}.py", f"def thing_{i}(): use_thing()\n" * 30) for i in range(10)
    ]
    built = asm.build(
        source=src, offset=len(src), language_id="python",
        current_path="file:///main.py", open_docs=big,
    )
    # Each candidate far exceeds the 20-token reserve, so at most the single
    # best one is kept (we always keep the first, even if it alone overflows).
    assert len(built.extra) <= 1
