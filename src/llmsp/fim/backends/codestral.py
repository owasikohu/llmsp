"""Mistral Codestral backend — the dedicated ``POST /v1/fim/completions``.

This endpoint is NOT the OpenAI ``completions.create`` shape, so we issue a raw
httpx request. Body fields: ``model``, ``prompt`` (code before the cursor),
``suffix`` (code after), plus ``stop``/``temperature``/``max_tokens``. The
code-specific host ``codestral.mistral.ai`` is the low-latency autocomplete
endpoint; the general ``api.mistral.ai`` host works too.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from llmsp.fim.base import FimRequest, flatten_prefix

CODESTRAL_BASE_URL = "https://codestral.mistral.ai"


class CodestralBackend:
    name = "codestral"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "codestral-2508",
        base_url: str = CODESTRAL_BASE_URL,
    ) -> None:
        self._model = model
        self._url = base_url.rstrip("/") + "/v1/fim/completions"
        self._client = httpx.AsyncClient(
            timeout=None,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

    async def complete(self, req: FimRequest) -> AsyncIterator[str]:
        payload = {
            "model": self._model,
            "prompt": flatten_prefix(req),
            "suffix": req.suffix,
            "max_tokens": req.knobs.max_tokens,
            "temperature": req.knobs.temperature,
            "stream": req.knobs.stream,
        }
        if req.knobs.stop:
            payload["stop"] = list(req.knobs.stop)

        if not req.knobs.stream:
            resp = await self._client.post(self._url, json=payload)
            resp.raise_for_status()
            yield resp.json()["choices"][0]["message"]["content"]
            return

        async with self._client.stream("POST", self._url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in obj.get("choices", []):
                    # FIM streaming uses delta.content.
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                    if text:
                        yield text

    async def aclose(self) -> None:
        await self._client.aclose()
