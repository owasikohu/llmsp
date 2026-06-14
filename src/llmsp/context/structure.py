"""Layer 3 — structural context and the single/multi-line decision.

Uses tree-sitter (via ``tree-sitter-language-pack``) when available to refine the
completion *mode*: force single-line continuations inside comments/strings where
a multi-line block would be wrong. Everything degrades gracefully to a pure
heuristic when tree-sitter or a grammar is unavailable, so the layer is always
safe to enable.

The tree-sitter Python bindings are not API-stable across versions (``Node.type``
vs ``Node.kind``; ``tree.root_node`` as property vs method; ``parse`` taking
``bytes`` vs ``str``; presence of ``descendant_for_byte_range``). The thin
adapter below feature-detects all of these and does its own descendant search so
the layer works regardless of which binding is installed.
"""

from __future__ import annotations

from typing import List

# Modes the assembler/server act on.
SINGLE = "single"
MULTI = "multi"
EMPTY = "empty"

# Tokens that, when they end the line before the cursor, open a block.
_BLOCK_OPENERS = (":", "{", "(", "[", "->", "=>", "=", ",")

# LSP language id -> tree-sitter grammar name (only the ones that differ).
_TS_LANG = {
    "csharp": "c_sharp",
    "shellscript": "bash",
    "sh": "bash",
    "objective-c": "objc",
    "javascriptreact": "javascript",
    "typescriptreact": "tsx",
    "dockerfile": "dockerfile",
    "makefile": "make",
}


def _is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _heuristic_mode(source: str, offset: int) -> str:
    offset = max(0, min(len(source), offset))
    after = source[offset:]
    before = source[:offset]

    # Mid-token: cursor sits inside an identifier -> don't suggest.
    if after[:1] and _is_ident_char(after[:1]) and before[-1:] and _is_ident_char(before[-1:]):
        return EMPTY

    line_start = before.rfind("\n") + 1
    line_prefix = before[line_start:]
    nl = after.find("\n")
    line_suffix = after if nl == -1 else after[:nl]

    # Substantial code already after the cursor on this line -> fill one line.
    if line_suffix.strip() and any(c.isalnum() for c in line_suffix):
        return SINGLE

    stripped = line_prefix.rstrip()
    if stripped.endswith(_BLOCK_OPENERS):
        return MULTI
    # Cursor on a blank line whose previous non-blank line opened a block.
    if not stripped:
        prev = before[:line_start].rstrip()
        if prev.endswith(_BLOCK_OPENERS):
            return MULTI
    return SINGLE


# --- portable tree-sitter adapter ------------------------------------------
def _maybe_call(value):
    return value() if callable(value) else value


def _kind(node) -> str:
    # Mainline binding: .type (str property) ; some bindings: .kind() (method).
    value = getattr(node, "type", None)
    if value is None:
        value = getattr(node, "kind", None)
    value = _maybe_call(value)
    return value if isinstance(value, str) else ""


def _start(node) -> int:
    return int(_maybe_call(getattr(node, "start_byte", 0)))


def _end(node) -> int:
    return int(_maybe_call(getattr(node, "end_byte", 0)))


def _parent(node):
    return _maybe_call(getattr(node, "parent", None))


def _children(node) -> List:
    kids = getattr(node, "children", None)
    if kids is not None and not callable(kids):
        return list(kids)
    count = _maybe_call(getattr(node, "child_count", 0)) or 0
    child = getattr(node, "child", None)
    if child is None:
        return []
    out = []
    for i in range(int(count)):
        try:
            out.append(child(i))
        except Exception:
            break
    return out


def _root(tree):
    return _maybe_call(getattr(tree, "root_node", None))


def _descendant_at(root, byte_off: int):
    """Smallest node whose half-open ``[start, end)`` byte span contains ``byte_off``."""
    node = root
    if node is None or not (_start(node) <= byte_off < max(_end(node), _start(node) + 1)):
        return node
    while True:
        for child in _children(node):
            if _start(child) <= byte_off < _end(child):
                node = child
                break
        else:
            return node


class Structure:
    """Lazy tree-sitter wrapper with a heuristic fallback."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._get_parser = None
        self._tried = False
        self._parsers: dict = {}

    def _ensure_loader(self) -> bool:
        if not self._enabled:
            return False
        if self._tried:
            return self._get_parser is not None
        self._tried = True
        try:
            from tree_sitter_language_pack import get_parser

            self._get_parser = get_parser
        except Exception:
            self._get_parser = None
        return self._get_parser is not None

    def _parser(self, language_id: str):
        name = _TS_LANG.get((language_id or "").lower(), (language_id or "").lower())
        if not name:
            return None
        if name in self._parsers:
            return self._parsers[name]
        try:
            parser = self._get_parser(name)  # type: ignore[misc]
        except Exception:
            parser = None
        self._parsers[name] = parser
        return parser

    @staticmethod
    def _parse(parser, source: str):
        # str-taking bindings vs bytes-taking bindings.
        try:
            return parser.parse(source)
        except TypeError:
            return parser.parse(source.encode("utf-8"))

    def decide_mode(self, source: str, offset: int, language_id: str = "") -> str:
        """Return ``SINGLE`` / ``MULTI`` / ``EMPTY`` for the cursor position."""
        base = _heuristic_mode(source, offset)
        # Tree-sitter only refines the MULTI decision (downgrade in comments /
        # strings); it can't add information to SINGLE / EMPTY.
        if base != MULTI or not self._ensure_loader():
            return base
        parser = self._parser(language_id)
        if parser is None:
            return base
        try:
            if self._in_comment_or_string(self._parse(parser, source), source, offset):
                return SINGLE
        except Exception:
            return base
        return base

    @staticmethod
    def _in_comment_or_string(tree, source: str, offset: int) -> bool:
        """Is the cursor inside (or at the trailing edge of) a comment/string?"""
        root = _root(tree)
        if root is None:
            return False
        byte_off = len(source[:offset].encode("utf-8"))
        # Probe the cursor and the byte just before it (handles end-of-token
        # boundaries, e.g. the cursor at the end of a comment line); walk a few
        # ancestors since a string node may wrap an inner token.
        for probe in (byte_off, byte_off - 1):
            if probe < 0:
                continue
            node = _descendant_at(root, probe)
            depth = 0
            while node is not None and depth < 4:
                k = _kind(node)
                if "comment" in k or "string" in k:
                    return True
                node = _parent(node)
                depth += 1
        return False
