"""OpenAI-compatible suffix backend.

Covers the two hosted endpoints that do true FIM through the stock ``openai``
client's legacy Completions call (``prompt`` + ``suffix``):

* OpenAI ``gpt-3.5-turbo-instruct`` (the only first-party model with FIM).
* DeepSeek, via ``base_url="https://api.deepseek.com/beta"``.

Both apply the model's FIM template server-side, so we pass plain prefix/suffix
and fold cross-file ``extra`` into the prefix.
"""

from __future__ import annotations

from typing import AsyncIterator, Optional

from llmsp.fim.base import FimRequest, flatten_prefix

DEEPSEEK_BASE_URL = "https://api.deepseek.com/beta"


class OpenAISuffixBackend:
    name = "openai-suffix"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "the OpenAI/DeepSeek backend needs the 'openai' package: "
                "pip install 'llmsp[openai]'"
            ) from exc
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key or "EMPTY", base_url=base_url)

    async def complete(self, req: FimRequest) -> AsyncIterator[str]:
        prompt = flatten_prefix(req)
        kwargs = {
            "model": self._model,
            "prompt": prompt,
            "suffix": req.suffix,
            "max_tokens": req.knobs.max_tokens,
            "temperature": req.knobs.temperature,
            "stream": req.knobs.stream,
        }
        if req.knobs.stop:
            kwargs["stop"] = list(req.knobs.stop)

        if not req.knobs.stream:
            resp = await self._client.completions.create(**kwargs)
            yield resp.choices[0].text or ""
            return

        stream = await self._client.completions.create(**kwargs)
        async for event in stream:
            if not event.choices:
                continue
            text = event.choices[0].text
            if text:
                yield text

    async def aclose(self) -> None:
        await self._client.close()
