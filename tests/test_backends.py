"""Backend adapters — request shape and stream parsing, mocked over httpx."""

import json

import httpx

from llmsp.fim.backends.codestral import CodestralBackend
from llmsp.fim.backends.llamacpp import LlamaCppInfillBackend
from llmsp.fim.backends.mock import MockBackend
from llmsp.fim.backends.ollama import OllamaBackend
from llmsp.fim.backends.raw_template import RawTemplateBackend
from llmsp.fim.base import FimRequest, Knobs, Snippet, collect
from llmsp.fim.templates import get_template


def _mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_mock_backend_echo_captures_request():
    b = MockBackend(echo=True)
    req = FimRequest(prefix="abc", suffix="xyz", extra=(Snippet("t", "f.py"),))
    out = await collect(b.complete(req))
    assert "f.py" in out and b.last_request is req


async def test_ollama_sends_suffix_and_folds_extra():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        lines = [
            json.dumps({"response": "a + ", "done": False}),
            json.dumps({"response": "b", "done": False}),
            json.dumps({"response": "", "done": True}),
        ]
        return httpx.Response(200, content="\n".join(lines).encode())

    b = OllamaBackend(model="qwen2.5-coder")
    b._client = _mock_client(handler)
    req = FimRequest(
        prefix="result = ",
        suffix="\nprint(result)",
        extra=(Snippet("def add(): ...", "u.py"),),
        knobs=Knobs(max_tokens=32, stop=("\n",)),
        language_id="python",
    )
    out = await collect(b.complete(req))
    assert out == "a + b"
    assert seen["url"].endswith("/api/generate")
    assert seen["body"]["suffix"] == "\nprint(result)"
    assert seen["body"]["options"]["stop"] == ["\n"]
    # Keep the model resident to avoid repeated cold-load latency.
    assert seen["body"]["keep_alive"] == "30m"
    # Cross-file snippet folded into the prompt with a comment header.
    assert "# u.py" in seen["body"]["prompt"]
    assert "def add(): ..." in seen["body"]["prompt"]
    await b.aclose()


async def test_codestral_parses_sse_delta():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert str(request.url).endswith("/v1/fim/completions")
        assert body["suffix"] == "SUF"
        lines = [
            'data: {"choices":[{"delta":{"content":"a + "}}]}',
            'data: {"choices":[{"delta":{"content":"b"}}]}',
            "data: [DONE]",
        ]
        return httpx.Response(200, content="\n".join(lines).encode())

    b = CodestralBackend(api_key="k", model="codestral-2508")
    b._client = _mock_client(handler)
    out = await collect(b.complete(FimRequest(prefix="PRE", suffix="SUF")))
    assert out == "a + b"
    await b.aclose()


async def test_raw_template_renders_fim_and_parses_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        lines = [
            'data: {"choices":[{"text":"a + "}]}',
            'data: {"choices":[{"text":"b"}]}',
            "data: [DONE]",
        ]
        return httpx.Response(200, content="\n".join(lines).encode())

    b = RawTemplateBackend(
        base_url="http://localhost:8000", model="qwen", template=get_template("qwen")
    )
    b._client = _mock_client(handler)
    out = await collect(
        b.complete(FimRequest(prefix="PRE", suffix="SUF", knobs=Knobs(stop=("\n",))))
    )
    assert out == "a + b"
    # The rendered prompt uses the model's FIM sentinels.
    assert seen["body"]["prompt"] == "<|fim_prefix|>PRE<|fim_suffix|>SUF<|fim_middle|>"
    # The model's own stop tokens are always added (anti-runaway).
    assert "<|endoftext|>" in seen["body"]["stop"]
    await b.aclose()


async def test_raw_template_repo_packs_extra():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=b'data: {"choices":[{"text":"x"}]}\ndata: [DONE]')

    b = RawTemplateBackend(
        base_url="http://localhost:8000", model="qwen", template=get_template("qwen")
    )
    b._client = _mock_client(handler)
    req = FimRequest(prefix="P", suffix="S", extra=(Snippet("helper()", "h.py"),))
    await collect(b.complete(req))
    assert "<|file_sep|>h.py" in seen["body"]["prompt"]
    await b.aclose()


async def test_llamacpp_infill_passes_structured_extra():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=b'data: {"content":"a","stop":false}\ndata: {"content":"b","stop":true}')

    b = LlamaCppInfillBackend(base_url="http://localhost:8080")
    b._client = _mock_client(handler)
    req = FimRequest(prefix="PRE", suffix="SUF", extra=(Snippet("body", "x.py"),))
    out = await collect(b.complete(req))
    assert out == "ab"
    assert seen["body"]["input_prefix"] == "PRE"
    assert seen["body"]["input_suffix"] == "SUF"
    assert seen["body"]["input_extra"] == [{"filename": "x.py", "text": "body"}]
    await b.aclose()
