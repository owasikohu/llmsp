# llmsp

A **Fill-In-the-Middle (FIM)** LLM code-completion **language server**, written in
Python with [pygls](https://github.com/openlawlibrary/pygls) and a
provider-agnostic OpenAI-compatible client.

The differentiator is **context engineering**: instead of sending the model only
the raw text around your cursor, llmsp assembles a smarter prompt — budgeted
single-file windowing, cross-file retrieval of relevant snippets, and a
structural single/multi-line decision — to lift completion accuracy.

It is **editor-agnostic**: it implements both the standard `textDocument/completion`
popup (works in any LSP client, no extension required) and LSP 3.18
`textDocument/inlineCompletion` ghost text (VS Code, Neovim 0.12+).

## How it works

```
keystroke ─► debounce + cancel-previous ─► context assembler ─► FIM backend ─► post-process ─► editor
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                                ▼                               ▼
   L1  budgeted prefix/suffix      L2  cross-file snippets        L3  single/multi/empty
       (token-aware, trims             (recently-edited ring          mode via tree-sitter
        away from the cursor)           buffer + open files,          (degrades to a heuristic)
                                        ranked by Jaccard/BM25,
                                        packed best-nearest-cursor)
```

### Provider-agnostic FIM backends

One interface, [`FIMBackend.complete()`](src/llmsp/fim/base.py), with adapters for
every practical FIM endpoint (see [src/llmsp/fim/backends/](src/llmsp/fim/backends/)):

| `backend` | Endpoint | Notes |
|---|---|---|
| `ollama` *(default to dogfood)* | `POST /api/generate` (`suffix`) | local, no key; applies the model's FIM template server-side |
| `deepseek` | `…/beta` Completions (`prompt`+`suffix`) | hosted, OpenAI-client-native, cheap true FIM |
| `openai-instruct` | legacy Completions (`suffix`) | only `gpt-3.5-turbo-instruct` does FIM on OpenAI |
| `codestral` | `…/v1/fim/completions` | best-in-class autocomplete; dedicated endpoint |
| `llamacpp` | `POST /infill` (`input_extra`) | cleanest local repo-aware path |
| `vllm` / `tgi` | `/v1/completions` (raw FIM tokens) | these ignore `suffix`; llmsp renders the [model's special tokens](src/llmsp/fim/templates.py) |
| `mock` | — | offline, deterministic; zero-config default |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core + test deps
pip install -e ".[all]"            # + openai, tiktoken, rank-bm25, tree-sitter
```

## Quickstart (Ollama, local)

```bash
ollama pull qwen2.5-coder:0.5b     # any FIM-capable code model
llmsp                              # speaks LSP over stdio
```

`qwen2.5-coder:0.5b` is small and fast (~0.4 s/completion warm on CPU); use
`:1.5b`/`:7b` for higher quality. The server preloads the model on startup
(warm-up) and keeps it resident, so the first completion isn't a slow cold load.

## Editors

| Editor | Setup | Ghost text |
|---|---|---|
| **VS Code** (incl. Remote-WSL) | build & install the extension in [editors/vscode/](editors/vscode/) | ✅ |
| **Neovim** 0.11+ | source [examples/nvim/llmsp.lua](examples/nvim/llmsp.lua) (no plugins) | ✅ (0.12+) |
| **Kate** | paste [examples/kate/lspclient-settings.json](examples/kate/lspclient-settings.json) into LSP Client settings | popup only |
| any LSP client | point it at the `llmsp` command | popup; ghost text if supported |

See each directory's README for exact steps.

## Configuration

Settings arrive in the LSP `initialize` request's `initialization_options` (and
can be refreshed via `workspace/configuration`). Everything has a default, so the
server starts with zero config against the offline `mock` backend. Full schema:
[src/llmsp/config.py](src/llmsp/config.py).

```jsonc
{
  "backend": "ollama",
  "model": "qwen2.5-coder:0.5b",
  "model_family": "qwen",          // drives FIM stop-tokens / leaked-sentinel cleanup
  "max_tokens": 128,
  "temperature": 0.1,
  "debounce_ms": 200,
  "request_timeout_ms": 30000,
  "multiline": "auto",             // auto | single | multi
  "context": {
    "max_prompt_tokens": 1536,
    "cross_file": true,            // Layer 2
    "retrieval": "jaccard",        // jaccard | bm25 | none
    "max_snippets": 4,
    "structural": true             // Layer 3 (needs the `treesitter` extra)
  }
}
```

Hosted backends read their key from the environment (`DEEPSEEK_API_KEY`,
`CODESTRAL_API_KEY`, `OPENAI_API_KEY`, or `LLMSP_API_KEY`).

## Develop & verify

```bash
pytest                             # 69 unit + e2e tests (pytest-lsp drives the real server)
ruff check src tests scripts
python scripts/eval_fim.py --backend ollama --model qwen2.5-coder:0.5b --model-family qwen --n 60
```

[scripts/eval_fim.py](scripts/eval_fim.py) masks the middle of real lines and
reports exact-match and edit-distance similarity for **L1**, **L1+L2** and
**L1+L2+L3**, so you can measure whether the context layers help on your own
codebase and model. An illustrative run on this repo (qwen2.5-coder:0.5b, n=24)
showed cross-file retrieval lifting exact-match 12.5% → 16.7% and similarity
47.1% → 49.2%; numbers are noisy at small n / tiny models — run it on your own
project for a real signal.

## Status

Implemented: all three context layers, every backend above, debounce +
cancel-on-keystroke, prefix-keyed caching, both completion surfaces, local-model
warm-up/keep-alive, and a full test suite (unit + `pytest-lsp` end-to-end, plus a
skipped-by-default live-Ollama test). Roadmap: embedding/RRF retrieval and
RepoCoder-style draft-then-retrieve as opt-in accuracy modes.
