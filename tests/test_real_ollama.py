"""End-to-end test against a REAL local Ollama FIM backend.

Skipped automatically unless an Ollama server is reachable on localhost:11434
with the model below pulled. Run manually with:

    ollama serve & ; ollama pull qwen2.5-coder:0.5b
    pytest tests/test_real_ollama.py -v -s
"""

import sys
import urllib.request

import pytest
import pytest_lsp
from lsprotocol import types
from pytest_lsp import ClientServerConfig, LanguageClient, client_capabilities

MODEL = "qwen2.5-coder:0.5b"
OLLAMA = "http://127.0.0.1:11434"


def _ollama_has_model() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=2) as r:
            import json

            names = {m["name"] for m in json.load(r).get("models", [])}
            return MODEL in names
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_has_model(), reason=f"Ollama not running or {MODEL} not pulled"
)


@pytest_lsp.fixture(
    config=ClientServerConfig(server_command=[sys.executable, "-m", "llmsp"]),
)
async def client(lsp_client: LanguageClient):
    await lsp_client.initialize_session(
        types.InitializeParams(
            capabilities=client_capabilities("visual-studio-code"),
            initialization_options={
                "backend": "ollama",
                "model": MODEL,
                "model_family": "qwen",
                "debounce_ms": 10,
                "request_timeout_ms": 30000,
                "max_tokens": 64,
                "temperature": 0.0,
            },
        )
    )
    yield
    await lsp_client.shutdown_session()


async def test_real_fim_completion_fills_fibonacci(client: LanguageClient):
    uri = "file:///fib.py"
    src = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return \n\nprint(fibonacci(10))\n"
    client.text_document_did_open(
        types.DidOpenTextDocumentParams(
            text_document=types.TextDocumentItem(
                uri=uri, language_id="python", version=1, text=src
            )
        )
    )
    # Cursor right after "return " on line 3.
    result = await client.text_document_completion_async(
        types.CompletionParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            position=types.Position(line=3, character=11),
        )
    )
    assert isinstance(result, types.CompletionList)
    assert result.items, "expected a real completion from the model"
    text = result.items[0].insert_text or result.items[0].text_edit.new_text
    print(f"\n[real ollama] completion -> {text!r}")
    assert "fibonacci" in text  # the model should recurse


async def test_real_inline_completion_returns_text(client: LanguageClient):
    uri = "file:///gcd.py"
    src = "def gcd(a, b):\n    while b:\n        a, b = \n    return a\n"
    client.text_document_did_open(
        types.DidOpenTextDocumentParams(
            text_document=types.TextDocumentItem(
                uri=uri, language_id="python", version=1, text=src
            )
        )
    )
    result = await client.text_document_inline_completion_async(
        types.InlineCompletionParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            position=types.Position(line=2, character=15),
            context=types.InlineCompletionContext(
                trigger_kind=types.InlineCompletionTriggerKind.Automatic
            ),
        )
    )
    assert isinstance(result, types.InlineCompletionList)
    assert result.items
    print(f"\n[real ollama] inline -> {result.items[0].insert_text!r}")
    assert result.items[0].insert_text.strip()
