"""Ollama backend — local FIM via ``POST /api/generate``.

Ollama applies the model's own FIM template internally when you pass a
``suffix`` field, so we send plain prefix/suffix. Note this is Ollama's *native*
endpoint, NOT its ``/v1`` OpenAI-compatible shim (which cannot take ``suffix``).

Cross-file ``extra`` snippets are folded into the prefix as comment-labelled
blocks (Ollama has no structured repo-context field).

Streaming responses are newline-delimited JSON; each line carries a ``response``
fragment and a ``done`` flag.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from llmsp.fim.base import FimRequest, flatten_prefix


class OllamaBackend:
    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5-coder",
        keep_alive: str = "30m",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        # Keep the model resident between requests so we don't pay the cold-load
        # cost (several seconds) again after an idle pause. Ollama's default is 5m.
        self._keep_alive = keep_alive
        self._client = httpx.AsyncClient(timeout=None)

    async def complete(self, req: FimRequest) -> AsyncIterator[str]:
        prompt = flatten_prefix(req)
        payload = {
            "model": self._model,
            "prompt": prompt,
            "suffix": req.suffix,
            "stream": req.knobs.stream,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": req.knobs.temperature,
                "num_predict": req.knobs.max_tokens,
            },
        }
        if req.knobs.stop:
            payload["options"]["stop"] = list(req.knobs.stop)

        url = f"{self._base_url}/api/generate"
        if not req.knobs.stream:
            resp = await self._client.post(url, json=payload)
            resp.raise_for_status()
            yield resp.json().get("response", "")
            return

        async with self._client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = obj.get("response", "")
                if chunk:
                    yield chunk
                if obj.get("done"):
                    break

    async def aclose(self) -> None:
        await self._client.aclose()
