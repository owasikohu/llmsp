"""The context-engineering pipeline — the project's differentiator.

Orchestrates the three layers into a single :class:`Built` result the server
hands to a backend:

* **L1** budgeted single-file window (always on).
* **L2** cross-file snippets from the recently-edited ring buffer and other open
  documents, ranked against the cursor and packed ascending within a token
  budget (deduped against what's already in the window).
* **L3** the single/multi/empty completion mode decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from llmsp.config import Config
from llmsp.context import retrieval
from llmsp.context.ringbuffer import Chunk, RingBuffer
from llmsp.context.structure import EMPTY, Structure
from llmsp.context.window import build_window, split_budget
from llmsp.fim.base import Snippet
from llmsp.tokenizer import TokenCounter

# Cursor window used as the retrieval query (research: ~500 chars, prefix-weighted).
_QUERY_PREFIX_CHARS = 500
_QUERY_SUFFIX_CHARS = 200
_MAX_CANDIDATES = 200


@dataclass(frozen=True)
class Built:
    prefix: str
    suffix: str
    extra: Tuple[Snippet, ...]
    mode: str


class ContextAssembler:
    def __init__(
        self,
        cfg: Config,
        *,
        counter: Optional[TokenCounter] = None,
        ring: Optional[RingBuffer] = None,
        structure: Optional[Structure] = None,
    ) -> None:
        self.cfg = cfg
        self.counter = counter or TokenCounter(cfg.model or None)
        self.ring = ring
        self.structure = structure or Structure(enabled=cfg.context.structural)

    # --- main entry ---------------------------------------------------------
    def build(
        self,
        *,
        source: str,
        offset: int,
        language_id: str = "",
        current_path: str = "",
        open_docs: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> Built:
        ctx = self.cfg.context
        mode = self.structure.decide_mode(source, offset, language_id)

        # Build cross-file context FIRST so the single-file window can reclaim any
        # reserve the snippets don't actually use. The reserve is capped at half
        # the prompt so the current file is never starved (the default config has
        # cross_file_tokens == max_prompt_tokens, which would otherwise zero it).
        raw_prefix = source[:offset]
        raw_suffix = source[offset:]
        extra: Tuple[Snippet, ...] = ()
        used = 0
        if ctx.cross_file and mode != EMPTY:
            cap = min(ctx.cross_file_tokens, ctx.max_prompt_tokens // 2)
            if cap > 0:
                extra = self._cross_file(
                    raw_prefix,
                    raw_suffix,
                    current_path=current_path,
                    open_docs=open_docs or (),
                    budget=cap,
                )
                used = sum(self._snippet_cost(s) for s in extra)

        single_budget = max(0, ctx.max_prompt_tokens - used)
        pre_budget, suf_budget = split_budget(
            single_budget, ctx.prefix_ratio, ctx.suffix_ratio
        )
        window = build_window(source, offset, pre_budget, suf_budget, self.counter)
        return Built(prefix=window.prefix, suffix=window.suffix, extra=extra, mode=mode)

    # --- layer 2 ------------------------------------------------------------
    def _cross_file(
        self,
        prefix: str,
        suffix: str,
        *,
        current_path: str,
        open_docs: Iterable[Tuple[str, str]],
        budget: int,
    ) -> Tuple[Snippet, ...]:
        ctx = self.cfg.context
        candidates = self._candidates(current_path, open_docs)
        if not candidates:
            return ()

        query = prefix[-_QUERY_PREFIX_CHARS:] + "\n" + suffix[:_QUERY_SUFFIX_CHARS]
        ranked = retrieval.rank(
            query, candidates, k=ctx.max_snippets, method=ctx.retrieval
        )
        ranked = self._dedupe(ranked, prefix, suffix)
        return self._fit_budget(ranked, budget)

    def _candidates(
        self, current_path: str, open_docs: Iterable[Tuple[str, str]]
    ) -> List[Snippet]:
        cands: List[Snippet] = []
        if self.ring is not None:
            for ch in self.ring.chunks():
                cands.append(Snippet(ch.text, ch.path))
        for path, text in open_docs:
            if path == current_path:
                continue
            for ch in _chunk_document(path, text, self.cfg.context.ring_chunk_lines):
                cands.append(Snippet(ch.text, ch.path))
        # Deduplicate identical snippet bodies, keep bounded.
        seen = set()
        uniq: List[Snippet] = []
        for s in cands:
            key = (s.path, s.text)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(s)
            if len(uniq) >= _MAX_CANDIDATES:
                break
        return uniq

    @staticmethod
    def _dedupe(snippets: Sequence[Snippet], prefix: str, suffix: str) -> List[Snippet]:
        """Drop snippets whose body substantially overlaps the current file.

        Compares whole stripped lines (not just the first line, and not raw
        substrings — which produced false positives for short tokens) and drops a
        snippet when at least half of its non-trivial lines already appear in the
        prefix/suffix. This avoids wasting budget on near-duplicates and re-feeding
        post-cursor code back to the model as "context".
        """
        window_lines = {
            ln.strip()
            for blk in (prefix, suffix)
            for ln in blk.splitlines()
            if len(ln.strip()) >= 4
        }
        out: List[Snippet] = []
        for s in snippets:
            lines = [ln.strip() for ln in s.text.splitlines() if len(ln.strip()) >= 4]
            if lines and sum(ln in window_lines for ln in lines) / len(lines) >= 0.5:
                continue
            out.append(s)
        return out

    def _snippet_cost(self, s: Snippet) -> int:
        # +2 for the path/header tokens added when the snippet is packed.
        return self.counter.count(s.text) + self.counter.count(s.path) + 2

    def _fit_budget(self, ranked: Sequence[Snippet], budget: int) -> Tuple[Snippet, ...]:
        """Keep the best snippets (from the ascending tail) that fit ``budget``."""
        kept: List[Snippet] = []
        total = 0
        for s in reversed(ranked):  # best first
            cost = self._snippet_cost(s)
            if total + cost > budget and kept:
                break
            kept.append(s)
            total += cost
        kept.reverse()  # back to ascending (best last / nearest cursor)
        return tuple(kept)


def _chunk_document(path: str, source: str, chunk_lines: int) -> List[Chunk]:
    lines = source.splitlines()
    if not lines:
        return []
    step = max(1, chunk_lines)
    chunks: List[Chunk] = []
    for start in range(0, len(lines), step):
        text = "\n".join(lines[start : start + step]).strip()
        if text:
            chunks.append(Chunk(path=path, text=text))
    return chunks
