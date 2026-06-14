"""Raw-template backend for vLLM / TGI.

These servers expose an OpenAI-compatible ``/v1/completions`` but do NOT honour
the ``suffix`` field (TGI even rejects it with HTTP 422). So we render the
model's FIM special-token string ourselves (via :mod:`llmsp.fim.templates`) and
send it as a plain ``prompt``. Repo-capable families (Qwen/StarCoder2) get
``extra`` packed with ``<file_sep>`` tokens; others fall back to folding.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from llmsp.fim.base import FimRequest, flatten_prefix
from llmsp.fim.templates import FimTemplate


class RawTemplateBackend:
    name = "raw-template"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        template: FimTemplate,
        api_key: str = "",
    ) -> None:
        self._url = base_url.rstrip("/") + "/v1/completions"
        self._model = model
        self._template = template
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(timeout=None, headers=headers)

    def _render(self, req: FimRequest) -> str:
        if req.extra and self._template.supports_repo:
            return self._template.render(req.prefix, req.suffix, req.extra)
        # Non-repo families: fold snippets into the prefix, then render PSM/SPM.
        flat = flatten_prefix(req)
        return self._template.render(flat, req.suffix)

    async def complete(self, req: FimRequest) -> AsyncIterator[str]:
        prompt = self._render(req)
        # Always stop on the model's own sentinels to defend against
        # non-stopping FIM (a known Qwen2.5-Coder bug).
        stop = list(dict.fromkeys([*req.knobs.stop, *self._template.stop]))
        payload = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": req.knobs.max_tokens,
            "temperature": req.knobs.temperature,
            "stream": req.knobs.stream,
            "stop": stop,
        }

        if not req.knobs.stream:
            resp = await self._client.post(self._url, json=payload)
            resp.raise_for_status()
            yield resp.json()["choices"][0]["text"]
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
                    text = choice.get("text") or ""
                    if text:
                        yield text

    async def aclose(self) -> None:
        await self._client.aclose()
