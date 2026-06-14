"""CLI argument handling.

Regression: editor LSP clients (vscode-languageclient) append a transport flag
like ``--stdio`` to the server command. argparse must accept it instead of
exiting with code 2 (which made VS Code kill/retry the server until it gave up).
"""

from llmsp.__main__ import build_parser


def test_parser_accepts_stdio():
    args, unknown = build_parser().parse_known_args(["--stdio"])
    assert args.stdio is True
    assert unknown == []


def test_parser_ignores_unknown_handshake_flags():
    # e.g. --clientProcessId=, --pipe=, --socket=, --node-ipc
    args, unknown = build_parser().parse_known_args(["--stdio", "--clientProcessId=42"])
    assert args.stdio is True
    assert "--clientProcessId=42" in unknown


def test_parser_version_still_works():
    args, _ = build_parser().parse_known_args(["--version"])
    assert args.version is True
