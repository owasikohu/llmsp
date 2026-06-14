"""Deterministic, offline backend for tests and zero-config startup.

It captures the last :class:`FimRequest` it received (so context-assembly tests
can assert on what reached the model) and streams back a canned response one
character at a time to exercise the streaming/cancellation path.
"""

from __future__ import annotations

from typing import AsyncIterator, Optional

from llmsp.fim.base import FimRequest


class MockBackend:
    name = "mock"

    def __init__(self, response: str = "pass  # llmsp-mock", *, echo: bool = False) -> None:
        self._response = response
        self._echo = echo
        self.last_request: Optional[FimRequest] = None

    async def complete(self, req: FimRequest) -> AsyncIterator[str]:
        self.last_request = req
        out = self._render(req)
        # Honour max_tokens loosely (1 char per "token") so timeout/budget paths
        # behave; stream char-by-char so cancellation can interrupt mid-stream.
        limit = max(1, req.knobs.max_tokens)
        for ch in out[:limit]:
            yield ch

    def _render(self, req: FimRequest) -> str:
        if not self._echo:
            return self._response
        # Echo mode: surface what the model "saw" for assertions/debugging.
        names = ",".join(sn.path or "?" for sn in req.extra)
        return f"[extra={names}|pre={req.prefix[-20:]!r}|suf={req.suffix[:20]!r}]"

    async def aclose(self) -> None:  # pragma: no cover - nothing to release
        return None
