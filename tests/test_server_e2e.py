"""End-to-end LSP tests: spawn the real server over stdio via pytest-lsp.

Uses the offline mock backend so the whole initialize → didOpen → completion /
inlineCompletion round-trip is exercised without any network.
"""

import asyncio
import sys

import pytest_lsp
from lsprotocol import types
from pytest_lsp import ClientServerConfig, LanguageClient, client_capabilities

MOCK_TEXT = "pass  # llmsp-mock"


@pytest_lsp.fixture(
    config=ClientServerConfig(server_command=[sys.executable, "-m", "llmsp"]),
)
async def client(lsp_client: LanguageClient):
    result = await lsp_client.initialize_session(
        types.InitializeParams(
            capabilities=client_capabilities("visual-studio-code"),
            initialization_options={"backend": "mock", "debounce_ms": 10},
        )
    )
    lsp_client.__dict__["init_result"] = result
    yield
    await lsp_client.shutdown_session()


async def test_advertises_completion_and_inline_providers(client: LanguageClient):
    # Regression: real clients (VS Code) only register the suggest popup and the
    # ghost-text provider when the server advertises these capabilities. pygls
    # only emits inlineCompletionProvider when the feature is registered with an
    # explicit options object.
    caps = client.init_result.capabilities
    assert caps.completion_provider is not None
    assert caps.inline_completion_provider is not None


def _open(client: LanguageClient, uri: str, text: str, language_id: str = "python") -> None:
    client.text_document_did_open(
        types.DidOpenTextDocumentParams(
            text_document=types.TextDocumentItem(
                uri=uri, language_id=language_id, version=1, text=text
            )
        )
    )


async def test_completion_returns_mock_item(client: LanguageClient):
    uri = "file:///single.py"
    _open(client, uri, "x = \n")
    result = await client.text_document_completion_async(
        types.CompletionParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            position=types.Position(line=0, character=4),
        )
    )
    assert isinstance(result, types.CompletionList)
    assert len(result.items) == 1
    assert result.items[0].insert_text == MOCK_TEXT


async def test_inline_completion_returns_mock_item(client: LanguageClient):
    uri = "file:///inline.py"
    _open(client, uri, "x = \n")
    result = await client.text_document_inline_completion_async(
        types.InlineCompletionParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            position=types.Position(line=0, character=4),
            context=types.InlineCompletionContext(
                trigger_kind=types.InlineCompletionTriggerKind.Automatic
            ),
        )
    )
    assert isinstance(result, types.InlineCompletionList)
    assert len(result.items) == 1
    assert result.items[0].insert_text == MOCK_TEXT


async def test_empty_mode_midtoken_returns_no_items(client: LanguageClient):
    uri = "file:///empty.py"
    _open(client, uri, "foobar = 1\n")
    result = await client.text_document_completion_async(
        types.CompletionParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            position=types.Position(line=0, character=3),  # inside "foobar"
        )
    )
    assert isinstance(result, types.CompletionList)
    assert len(result.items) == 0


async def test_both_surfaces_return_for_same_keystroke(client: LanguageClient):
    """Regression: completion and inlineCompletion for the SAME uri must not
    cancel each other (they are keyed separately in _inflight)."""
    uri = "file:///both.py"
    _open(client, uri, "x = \n")
    pos = types.Position(line=0, character=4)
    comp, inline = await asyncio.gather(
        client.text_document_completion_async(
            types.CompletionParams(
                text_document=types.TextDocumentIdentifier(uri=uri), position=pos
            )
        ),
        client.text_document_inline_completion_async(
            types.InlineCompletionParams(
                text_document=types.TextDocumentIdentifier(uri=uri),
                position=pos,
                context=types.InlineCompletionContext(
                    trigger_kind=types.InlineCompletionTriggerKind.Automatic
                ),
            )
        ),
    )
    assert isinstance(comp, types.CompletionList) and comp.items
    assert comp.items[0].insert_text == MOCK_TEXT
    assert isinstance(inline, types.InlineCompletionList) and inline.items
    assert inline.items[0].insert_text == MOCK_TEXT


async def test_server_survives_rapid_superseding_requests(client: LanguageClient):
    """Rapid same-document requests cancel their predecessors; the server stays
    responsive and a subsequent request still returns a result."""
    uri = "file:///rapid.py"
    _open(client, uri, "x = \n")
    params = types.CompletionParams(
        text_document=types.TextDocumentIdentifier(uri=uri),
        position=types.Position(line=0, character=4),
    )
    # Fire several concurrently — earlier ones may come back cancelled (errors).
    await asyncio.gather(
        *(client.text_document_completion_async(params) for _ in range(4)),
        return_exceptions=True,
    )
    # Liveness: a fresh request after the storm still works.
    final = await client.text_document_completion_async(params)
    assert isinstance(final, types.CompletionList)
    assert final.items and final.items[0].insert_text == MOCK_TEXT
