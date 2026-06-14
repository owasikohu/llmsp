#!/usr/bin/env python3
"""Measure whether the context layers actually improve FIM accuracy.

Masks the middle of real lines in a codebase and asks the configured backend to
fill them back in under three context configurations — L1 (single-file only),
L1+L2 (cross-file retrieval), L1+L2+L3 (+ tree-sitter mode) — then reports
exact-match rate and mean edit-distance similarity for each.

Run against a real FIM backend to get meaningful numbers, e.g.::

    python scripts/eval_fim.py --backend ollama --model qwen2.5-coder --n 60
    python scripts/eval_fim.py --backend deepseek --model deepseek-coder --n 60

The default (mock backend) only checks that the harness runs end to end.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from llmsp.config import Config
from llmsp.context import postprocess
from llmsp.context.assembler import ContextAssembler
from llmsp.fim import factory
from llmsp.fim.base import FimRequest, Knobs, collect
from llmsp.tokenizer import TokenCounter

SKIP_LINE_PREFIXES = ("#", "//", "/*", "*", '"""', "'''", "import ", "from ")


@dataclass
class Example:
    path: str
    holed_source: str
    offset: int
    truth: str  # the masked middle (single line, code only)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    m = max(len(a), len(b))
    return 1.0 if m == 0 else 1.0 - levenshtein(a, b) / m


def gather_files(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*.py")
        if "/.venv/" not in str(p) and "/tests/" not in str(p) and "__pycache__" not in str(p)
    )


def make_examples(files: Sequence[Path], limit: int) -> Tuple[List[Example], dict]:
    examples: List[Example] = []
    sources = {str(p): p.read_text(encoding="utf-8", errors="ignore") for p in files}
    for path, src in sources.items():
        lines = src.splitlines(keepends=True)
        if len(lines) < 6:
            continue
        pos = 0
        # Deterministic sampling: every 7th eligible line.
        eligible = 0
        for i, line in enumerate(lines):
            start_of_line = pos
            pos += len(line)
            if i < 2 or i > len(lines) - 2:
                continue
            code = line.strip()
            if len(code) < 8 or code.startswith(SKIP_LINE_PREFIXES):
                continue
            eligible += 1
            if eligible % 7 != 0:
                continue
            indent = len(line) - len(line.lstrip())
            code_start = start_of_line + indent
            line_end = start_of_line + len(line.rstrip("\n"))
            truth = src[code_start:line_end]
            if not truth.strip():
                continue
            holed = src[:code_start] + src[line_end:]
            examples.append(Example(path, holed, code_start, truth))
            if len(examples) >= limit:
                return examples, sources
    return examples, sources


@dataclass
class Score:
    n: int = 0
    exact: int = 0
    sim_total: float = 0.0
    ms_total: float = 0.0

    def add(self, pred: str, truth: str, ms: float) -> None:
        self.n += 1
        p = pred.strip()
        t = truth.strip()
        if p == t:
            self.exact += 1
        self.sim_total += similarity(p, t)
        self.ms_total += ms

    def row(self, label: str) -> str:
        if self.n == 0:
            return f"{label:<14} (no examples)"
        return (
            f"{label:<14} exact={self.exact / self.n * 100:5.1f}%  "
            f"sim={self.sim_total / self.n * 100:5.1f}%  "
            f"lat={self.ms_total / self.n:6.0f}ms  (n={self.n})"
        )


def config_for(args, *, cross_file: bool, structural: bool) -> Config:
    return Config.from_mapping(
        {
            "backend": args.backend,
            "model": args.model,
            "base_url": args.base_url,
            "model_family": args.model_family,
            "max_tokens": 64,
            "temperature": 0.0,
            "context": {
                "cross_file": cross_file,
                "structural": structural,
                "retrieval": args.retrieval,
                "max_prompt_tokens": args.max_prompt_tokens,
            },
        }
    )


async def run_config(
    label: str, cfg: Config, examples: Sequence[Example], sources: dict
) -> Score:
    backend = factory.build_backend(cfg)
    template = factory.resolve_template(cfg)
    sentinels = template.sentinels() if template else ()
    counter = TokenCounter(cfg.model or None)
    assembler = ContextAssembler(cfg, counter=counter)
    score = Score()
    try:
        for ex in examples:
            open_docs = [(p, s) for p, s in sources.items() if p != ex.path]
            built = assembler.build(
                source=ex.holed_source,
                offset=ex.offset,
                language_id="python",
                current_path=ex.path,
                open_docs=open_docs,
            )
            knobs = Knobs(max_tokens=64, temperature=0.0, stop=("\n",), timeout_ms=10000)
            req = FimRequest(
                prefix=built.prefix, suffix=built.suffix, extra=built.extra,
                knobs=knobs, language_id="python",
            )
            t0 = time.monotonic()
            try:
                raw = await asyncio.wait_for(collect(backend.complete(req)), timeout=15)
            except Exception as exc:  # network/backend issues shouldn't abort the run
                print(f"  ! {ex.path}: {exc!r}")
                raw = ""
            ms = (time.monotonic() - t0) * 1000
            pred = postprocess.clean(raw, suffix=built.suffix, sentinels=sentinels, single_line=True)
            score.add(pred, ex.truth, ms)
    finally:
        if hasattr(backend, "aclose"):
            await backend.aclose()
    return score


async def amain(args) -> int:
    root = Path(args.path).resolve()
    files = gather_files(root)
    if not files:
        print(f"no .py files under {root}")
        return 1
    examples, sources = make_examples(files, args.n)
    print(f"FIM eval: {len(examples)} examples from {len(files)} files under {root}")
    print(f"backend={args.backend} model={args.model or '(default)'}\n")

    configs = [
        ("L1", dict(cross_file=False, structural=False)),
        ("L1+L2", dict(cross_file=True, structural=False)),
        ("L1+L2+L3", dict(cross_file=True, structural=True)),
    ]
    print("results:")
    for label, flags in configs:
        score = await run_config(label, config_for(args, **flags), examples, sources)
        print("  " + score.row(label))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--model", default="")
    ap.add_argument("--model-family", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--retrieval", default="jaccard", choices=["jaccard", "bm25", "none"])
    ap.add_argument("--max-prompt-tokens", type=int, default=1024)
    ap.add_argument("--path", default=str(Path(__file__).resolve().parents[1] / "src"))
    ap.add_argument("--n", type=int, default=40, help="number of FIM examples")
    args = ap.parse_args(argv)
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
