"""Layer 2 source — a ring buffer of recently-edited code chunks.

Research across llm-ls / llama.vscode / Continue shows that *recently-edited*
ranges are the single highest-value cross-file signal. On each ``didChange`` we
record the neighbourhood of the edit as a chunk; the most recent ``capacity``
chunks form the candidate pool that retrieval ranks against the cursor.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List


@dataclass(frozen=True)
class Chunk:
    path: str  # document uri or repo-relative path
    text: str


class RingBuffer:
    def __init__(self, capacity: int = 16, chunk_lines: int = 64) -> None:
        self._capacity = max(1, capacity)
        self._chunk_lines = max(1, chunk_lines)
        self._chunks: Deque[Chunk] = deque(maxlen=self._capacity)

    def record_edit(self, path: str, source: str, line: int) -> None:
        """Capture ~``chunk_lines`` of context around an edited ``line``."""
        lines = source.splitlines()
        if not lines:
            return
        half = self._chunk_lines // 2
        start = max(0, line - half)
        end = min(len(lines), start + self._chunk_lines)
        text = "\n".join(lines[start:end]).strip()
        if text:
            self._push(Chunk(path=path, text=text))

    def _push(self, chunk: Chunk) -> None:
        # Drop an identical existing chunk so the freshest copy moves to the end.
        existing = [c for c in self._chunks if not (c.path == chunk.path and c.text == chunk.text)]
        self._chunks = deque(existing, maxlen=self._capacity)
        self._chunks.append(chunk)

    def chunks(self) -> List[Chunk]:
        """Most-recent-last list of candidate chunks."""
        return list(self._chunks)

    def extend(self, chunks: Iterable[Chunk]) -> None:
        for c in chunks:
            if c.text.strip():
                self._push(c)

    def clear(self) -> None:
        self._chunks.clear()
