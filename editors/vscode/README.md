# llmsp for VS Code

A thin client that runs the [llmsp](../../) FIM language server inside VS Code
(including Remote-WSL) and surfaces its completions two ways:

- **Suggest popup** — via the standard `textDocument/completion` (works in every
  VS Code, no extra setup).
- **Inline ghost text** — via `textDocument/inlineCompletion`; press **Tab** to
  accept. Toggle with the `llmsp.inlineCompletion` setting.

The extension is a **workspace (remote) extension** so, under Remote-WSL, it runs
on the Linux side and can spawn the server from your virtualenv.

## Configure

All settings live under `llmsp.*` (see *Settings → Extensions → llmsp*). The
important one is the server command:

```jsonc
"llmsp.serverCommand": ["/home/owata/llmsp/.venv/bin/llmsp"],
"llmsp.backend": "ollama",
"llmsp.model": "qwen2.5-coder:0.5b",
"llmsp.modelFamily": "qwen"
```

Make sure your backend is reachable (e.g. `ollama serve` running and the model
pulled). For inline ghost text, keep VS Code's `editor.inlineSuggest.enabled`
on (default).

### Ghost-text-first (recommended)

For a clean Copilot-style experience — ghost text only, no popup competing for
attention, and one model call per keystroke instead of two — turn off the suggest
popup's auto-trigger in your workspace `.vscode/settings.json`:

```jsonc
{
  "editor.inlineSuggest.enabled": true,
  "editor.quickSuggestions": { "other": false, "comments": false, "strings": false }
}
```

`Ctrl+Space` still opens the popup on demand.

## Build & install (Remote-WSL)

```bash
cd editors/vscode
npm install
npm run build              # bundles src/extension.ts -> out/extension.js
npx @vscode/vsce package   # produces llmsp-vscode-0.1.0.vsix
code --install-extension llmsp-vscode-0.1.0.vsix   # WSL-remote code CLI
```

Then reload the window (*Developer: Reload Window*). Open a Python file and start
typing — accept ghost text with **Tab**, or open the suggest popup with
**Ctrl+Space**. Use the command **llmsp: Restart Server** after changing settings.
