"""The pygls glue: documents, debouncing, cancellation → FIM backend.

Handlers are async coroutines (pygls runs each as an :class:`asyncio.Task`) so an
in-flight LLM request can be cancelled the instant the user keeps typing. Two
completion surfaces share one engine (:meth:`LlmspServer._suggest`):

* ``textDocument/completion`` — the universal popup, works in any LSP client.
* ``textDocument/inlineCompletion`` — LSP 3.18 ghost text (Neovim 0.12, VS Code).
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from collections import OrderedDict
from typing import List, Optional, Tuple

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from llmsp.config import Config
from llmsp.context import postprocess
from llmsp.context.assembler import Built, ContextAssembler
from llmsp.context.ringbuffer import RingBuffer
from llmsp.context.structure import EMPTY, MULTI, SINGLE, Structure
from llmsp.fim import factory
from llmsp.fim.base import FimRequest, Knobs, collect, join_stops
from llmsp.tokenizer import TokenCounter

_WORD_TAIL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")
_CACHE_MAX = 256
_SINGLE_LINE_MAX_TOKENS = 80


class LlmspServer(LanguageServer):
    def __init__(self) -> None:
        super().__init__(
            "llmsp", "0.1.0", text_document_sync_kind=types.TextDocumentSyncKind.Incremental
        )
        self._inflight: dict[Tuple[str, str], asyncio.Task] = {}
        self._cache: "OrderedDict[Tuple, str]" = OrderedDict()
        self.configure(Config())

    # --- (re)configuration --------------------------------------------------
    def configure(self, cfg: Config) -> None:
        old = getattr(self, "backend", None)
        self.config = cfg
        self.counter = TokenCounter(cfg.model or None)
        self.ring = RingBuffer(
            capacity=cfg.context.ring_chunks, chunk_lines=cfg.context.ring_chunk_lines
        )
        self.structure = Structure(enabled=cfg.context.structural)
        self.assembler = ContextAssembler(
            cfg, counter=self.counter, ring=self.ring, structure=self.structure
        )
        self.backend = factory.build_backend(cfg)
        self.template = factory.resolve_template(cfg)
        self.sentinels = self.template.sentinels() if self.template else ()
        self._cache.clear()
        if old is not None and hasattr(old, "aclose"):
            self._close_later(old)
        self._maybe_warmup()

    def _close_later(self, backend) -> None:
        try:
            asyncio.get_running_loop().create_task(backend.aclose())
        except RuntimeError:  # no loop yet (startup) — nothing to close
            pass

    def _maybe_warmup(self) -> None:
        # Preload local models so the user's FIRST completion isn't the slow
        # cold-load (several seconds). Hosted backends are skipped to avoid a
        # needless billed call.
        if getattr(self.backend, "name", "") not in ("ollama", "llamacpp"):
            return
        try:
            asyncio.get_running_loop().create_task(self._warmup())
        except RuntimeError:  # no running loop yet (startup before INITIALIZE)
            pass

    async def _warmup(self) -> None:
        try:
            req = FimRequest(
                prefix="def ",
                suffix="",
                knobs=Knobs(max_tokens=1, temperature=0.0, stop=("\n",), timeout_ms=60000),
            )
            await collect(self.backend.complete(req))
            self.log(f"warmed up {self.backend.name}")
        except Exception as exc:  # backend may be down — not fatal
            self.log(f"warmup skipped: {exc!r}", types.MessageType.Warning)

    def log(self, message: str, level: types.MessageType = types.MessageType.Log) -> None:
        self.window_log_message(types.LogMessageParams(type=level, message=f"[llmsp] {message}"))

    # --- the shared completion engine --------------------------------------
    async def _suggest(
        self, uri: str, position: types.Position, surface: str = "completion"
    ) -> Optional[str]:
        # Debounce + cancel the previous request *for this surface and document*.
        # Keying by (uri, surface) is essential: a single keystroke makes some
        # clients fire both completion and inlineCompletion for the same uri, and
        # keying by uri alone would make the two surfaces cancel each other so
        # only one ever returns.
        inflight_key = (uri, surface)
        prev = self._inflight.get(inflight_key)
        if prev is not None and not prev.done():
            prev.cancel()
        task = asyncio.current_task()
        if task is not None:
            self._inflight[inflight_key] = task
        try:
            await asyncio.sleep(self.config.debounce_ms / 1000)

            doc = self.workspace.get_text_document(uri)
            try:
                offset = doc.offset_at_position(position)
            except Exception:
                return None

            built = self.assembler.build(
                source=doc.source,
                offset=offset,
                language_id=doc.language_id or "",
                current_path=uri,
                open_docs=self._open_docs(),
            )
            mode = self._effective_mode(built.mode)
            if mode == EMPTY:
                return None

            # Cache on what actually drives the request: the assembled prompt
            # (window + cross-file snippets) and the mode — not a fixed-size raw
            # slice that ignores the far window and cross-file context.
            key = self._cache_key(uri, mode, built)
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key] or None

            knobs, single_line = self._knobs_for(mode)
            req = FimRequest(
                prefix=built.prefix,
                suffix=built.suffix,
                extra=built.extra,
                knobs=knobs,
                language_id=doc.language_id or "",
            )

            started = time.monotonic()
            raw = await asyncio.wait_for(
                collect(self.backend.complete(req)), timeout=knobs.timeout_ms / 1000
            )
            text = postprocess.clean(
                raw, suffix=built.suffix, sentinels=self.sentinels, single_line=single_line
            )
            elapsed = (time.monotonic() - started) * 1000
            self.log(
                f"{self.backend.name} {surface} {mode} {len(built.extra)} snippets "
                f"{elapsed:.0f}ms -> {len(text)} chars"
            )
            self._cache_put(key, text)
            return text or None
        except asyncio.TimeoutError:
            self.log("backend timed out", types.MessageType.Warning)
            return None
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # never let a handler crash the server
            self.log(f"completion error: {exc!r}", types.MessageType.Error)
            return None
        finally:
            if self._inflight.get(inflight_key) is task:
                self._inflight.pop(inflight_key, None)

    # --- helpers ------------------------------------------------------------
    def _open_docs(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for uri, doc in self.workspace.text_documents.items():
            out.append((uri, doc.source))
        return out

    def _effective_mode(self, mode: str) -> str:
        if mode == EMPTY:
            return EMPTY
        forced = self.config.multiline
        if forced == "single":
            return SINGLE
        if forced == "multi":
            return MULTI
        return mode

    def _knobs_for(self, mode: str) -> Tuple[Knobs, bool]:
        cfg = self.config
        single_line = mode == SINGLE
        max_tokens = (
            min(cfg.max_tokens, _SINGLE_LINE_MAX_TOKENS) if single_line else cfg.max_tokens
        )
        line_stop = ("\n",) if single_line else ("\n\n",)
        stop = join_stops(cfg.stop, line_stop, self.sentinels)
        knobs = Knobs(
            max_tokens=max_tokens,
            temperature=cfg.temperature,
            stop=stop,
            stream=True,
            timeout_ms=cfg.request_timeout_ms,
        )
        return knobs, single_line

    @staticmethod
    def _cache_key(uri: str, mode: str, built: Built) -> Tuple:
        h = hashlib.blake2b(digest_size=16)
        h.update(built.prefix.encode("utf-8", "ignore"))
        h.update(b"\x00")
        h.update(built.suffix.encode("utf-8", "ignore"))
        for s in built.extra:
            h.update(b"\x01")
            h.update(s.path.encode("utf-8", "ignore"))
            h.update(b"\x02")
            h.update(s.text.encode("utf-8", "ignore"))
        return (uri, mode, h.hexdigest())

    def _cache_put(self, key: Tuple, value: str) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > _CACHE_MAX:
            self._cache.popitem(last=False)


server = LlmspServer()


# --- lifecycle --------------------------------------------------------------
@server.feature(types.INITIALIZE)
def on_initialize(ls: LlmspServer, params: types.InitializeParams) -> None:
    opts = params.initialization_options
    cfg = Config.from_mapping(opts if isinstance(opts, dict) else None)
    ls.configure(cfg)
    ls.log(f"initialized backend={cfg.backend} model={cfg.model or '(default)'}")


@server.feature(types.SHUTDOWN)
async def on_shutdown(ls: LlmspServer, *_: object) -> None:
    backend = getattr(ls, "backend", None)
    if backend is not None and hasattr(backend, "aclose"):
        try:
            await backend.aclose()
        except Exception:
            pass


# --- document tracking (feeds the L2 ring buffer) ---------------------------
@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def on_did_change(ls: LlmspServer, params: types.DidChangeTextDocumentParams) -> None:
    uri = params.text_document.uri
    doc = ls.workspace.get_text_document(uri)
    line = 0
    for change in params.content_changes:
        rng = getattr(change, "range", None)
        if rng is not None:
            line = rng.start.line
    ls.ring.record_edit(uri, doc.source, line)


# --- completion surfaces ----------------------------------------------------
@server.feature(types.TEXT_DOCUMENT_COMPLETION)
async def on_completion(
    ls: LlmspServer, params: types.CompletionParams
) -> Optional[types.CompletionList]:
    uri = params.text_document.uri
    text = await ls._suggest(uri, params.position, surface="completion")
    if not text:
        return types.CompletionList(is_incomplete=True, items=[])

    doc = ls.workspace.get_text_document(uri)
    line_prefix = _line_prefix(doc.source, params.position)
    m = _WORD_TAIL.search(line_prefix)
    word = m.group(0) if m else ""
    first_line = (text.splitlines() or [""])[0]
    label = (text.strip().splitlines() or [text])[0][:60] or "llmsp"
    # Keep the edit range and filterText describing the SAME span. When a partial
    # word precedes the cursor, replace it (word..cursor) with word+text so the
    # client filters on the typed word; filterText stays single-line.
    if word:
        start = types.Position(
            line=params.position.line, character=params.position.character - len(word)
        )
        edit_range = types.Range(start=start, end=params.position)
        new_text = word + text
        filter_text: Optional[str] = word + first_line
    else:
        edit_range = types.Range(start=params.position, end=params.position)
        new_text = text
        filter_text = None
    item = types.CompletionItem(
        label=label,
        kind=types.CompletionItemKind.Text,
        detail="llmsp",
        insert_text=new_text,
        insert_text_format=types.InsertTextFormat.PlainText,
        filter_text=filter_text,
        text_edit=types.TextEdit(range=edit_range, new_text=new_text),
        preselect=True,
    )
    return types.CompletionList(is_incomplete=True, items=[item])


# Pass explicit options: pygls only advertises `inlineCompletionProvider` in the
# server capabilities when the feature is registered WITH an options object
# (its capability default is None). Without this, real clients like VS Code never
# register the ghost-text provider even though the handler works when called.
@server.feature(types.TEXT_DOCUMENT_INLINE_COMPLETION, types.InlineCompletionOptions())
async def on_inline_completion(
    ls: LlmspServer, params: types.InlineCompletionParams
) -> Optional[types.InlineCompletionList]:
    uri = params.text_document.uri
    text = await ls._suggest(uri, params.position, surface="inline")
    if not text:
        return types.InlineCompletionList(items=[])
    item = types.InlineCompletionItem(
        insert_text=text,
        range=types.Range(start=params.position, end=params.position),
    )
    return types.InlineCompletionList(items=[item])


def _line_prefix(source: str, position: types.Position) -> str:
    lines = source.splitlines()
    if 0 <= position.line < len(lines):
        return lines[position.line][: position.character]
    return ""
