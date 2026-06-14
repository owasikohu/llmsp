# Using llmsp in KDE Kate

Kate ships an **LSP Client** plugin built in, so no extension is needed — you
just register the server command. Kate supports `textDocument/completion` (the
completion popup), which is exactly the editor-agnostic path llmsp targets. Kate
does **not** support `inlineCompletion` (ghost text), so completions appear in
the popup rather than as gray inline text.

## Setup

1. **Start the backend** (Ollama, in this example):
   ```bash
   OLLAMA_HOST=127.0.0.1:11434 ~/.local/ollama/bin/ollama serve &
   ollama pull qwen2.5-coder:0.5b   # if not already pulled
   ```

2. **Enable the LSP Client plugin**: *Settings → Configure Kate… → Plugins →*
   check **LSP Client**.

3. **Register llmsp**: *Settings → Configure Kate… → LSP Client →* open the
   **User Server Settings** tab and paste the contents of
   [`lspclient-settings.json`](lspclient-settings.json). Click **OK**.
   (Adjust the `command` path if your virtualenv lives elsewhere.)

## Use it

1. Open a Python file, e.g. `kate /tmp/try.py`.
2. Put the cursor where you want a completion (e.g. right after `return `).
3. Press **Ctrl+Space** to request a completion — the FIM suggestion appears in
   the popup; press **Enter** to accept.
   (Kate has no trigger characters configured for llmsp, so explicit Ctrl+Space
   is the reliable trigger.)

## Tips & troubleshooting

- **Cross-file context (Layer 2)**: open a few project files in other tabs so
  llmsp can pull relevant snippets from them into the prompt.
- **Server log**: *View → Tool Views → Show LSP Client* shows the server's
  `window/logMessage` output (backend used, mode, latency, snippet count) — handy
  for confirming it's really hitting Ollama.
- **First completion is slow**: the model loads into RAM on the first request;
  `request_timeout_ms` is set to 30 s to allow for it.
- **Bigger model**: change `model` to `qwen2.5-coder:7b` (or `deepseek-coder`,
  etc.) for stronger completions; pull it in Ollama first.
