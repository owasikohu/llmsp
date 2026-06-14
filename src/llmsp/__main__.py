"""Console entry point: ``llmsp`` / ``python -m llmsp``.

Speaks LSP over stdio, the standard transport for an editor-spawned server.
"""

from __future__ import annotations

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmsp", description="FIM LLM code-completion language server."
    )
    parser.add_argument(
        "--log",
        default="WARNING",
        help="logging level for stderr (DEBUG/INFO/WARNING/ERROR).",
    )
    parser.add_argument(
        "--version", action="store_true", help="print version and exit."
    )
    # LSP clients (e.g. vscode-languageclient) append a transport flag such as
    # `--stdio` to the server command. We only speak stdio, so accept it as a
    # no-op; other transport/handshake flags (`--clientProcessId=...`, `--pipe`,
    # `--socket`, `--node-ipc`) are tolerated via parse_known_args below.
    parser.add_argument(
        "--stdio", action="store_true", help="use stdio transport (default)."
    )
    return parser


def main(argv: "list[str] | None" = None) -> int:
    parser = build_parser()
    args, _unknown = parser.parse_known_args(argv)

    if args.version:
        from llmsp import __version__

        print(__version__)
        return 0

    # Log to stderr only — stdout is the LSP channel.
    logging.basicConfig(
        level=getattr(logging, str(args.log).upper(), logging.WARNING),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from llmsp.server import server

    server.start_io()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
