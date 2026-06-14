"""llama.cpp server backend — the dedicated ``POST /infill`` endpoint.

``/infill`` takes ``input_prefix`` / ``input_suffix`` and an ``input_extra``
array of ``{filename, text}`` for repo context; the server builds the correct
FIM special tokens from the GGUF metadata. This is the cleanest local repo-aware
path, so cross-file snippets are passed *structurally* rather than folded.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from llmsp.fim.base import FimRequest


class LlamaCppInfillBackend:
    name = "llamacpp"

    def __init__(self, *, base_url: str = "http://localhost:8080") -> None:
        self._url = base_url.rstrip("/") + "/infill"
        self._client = httpx.AsyncClient(timeout=None)

    async def complete(self, req: FimRequest) -> AsyncIterator[str]:
        payload = {
            "input_prefix": req.prefix,
            "input_suffix": req.suffix,
            "input_extra": [
                {"filename": sn.path or "snippet", "text": sn.text} for sn in req.extra
            ],
            "n_predict": req.knobs.max_tokens,
            "temperature": req.knobs.temperature,
            "stream": req.knobs.stream,
        }
        if req.knobs.stop:
            payload["stop"] = list(req.knobs.stop)

        if not req.knobs.stream:
            resp = await self._client.post(self._url, json=payload)
            resp.raise_for_status()
            yield resp.json().get("content", "")
            return

        async with self._client.stream("POST", self._url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                chunk = obj.get("content", "")
                if chunk:
                    yield chunk
                if obj.get("stop"):
                    break

    async def aclose(self) -> None:
        await self._client.aclose()
