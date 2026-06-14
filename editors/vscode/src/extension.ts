import {
  ExtensionContext,
  workspace,
  window,
  commands,
  ProviderResult,
} from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  TransportKind,
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;

function cfg() {
  return workspace.getConfiguration("llmsp");
}

/** Map the extension's `llmsp.*` settings to the server's initializationOptions. */
function buildInitOptions(): Record<string, unknown> {
  const c = cfg();
  const opts: Record<string, unknown> = {
    backend: c.get("backend", "ollama"),
    model: c.get("model", ""),
    model_family: c.get("modelFamily", ""),
    max_tokens: c.get("maxTokens", 128),
    temperature: c.get("temperature", 0.1),
    debounce_ms: c.get("debounceMs", 200),
    request_timeout_ms: c.get("requestTimeoutMs", 30000),
    multiline: c.get("multiline", "auto"),
    context: {
      cross_file: c.get("context.crossFile", true),
      retrieval: c.get("context.retrieval", "jaccard"),
      structural: c.get("context.structural", true),
    },
  };
  const baseUrl = c.get<string>("baseUrl", "");
  if (baseUrl) {
    opts.base_url = baseUrl;
  }
  return opts;
}

function buildServerOptions(): ServerOptions {
  const command = cfg().get<string[]>("serverCommand", ["llmsp"]);
  const [cmd, ...args] = command.length ? command : ["llmsp"];
  return {
    command: cmd,
    args,
    transport: TransportKind.stdio,
    options: { env: process.env },
  };
}

async function startClient(): Promise<void> {
  const languages = cfg().get<string[]>("languages", ["python"]);
  const clientOptions: LanguageClientOptions = {
    documentSelector: languages.map((language) => ({ scheme: "file", language })),
    initializationOptions: buildInitOptions(),
    // The language client auto-registers BOTH the suggest popup
    // (textDocument/completion) and inline ghost text
    // (textDocument/inlineCompletion) from the server's capabilities. This
    // middleware lets `llmsp.inlineCompletion` toggle the ghost text without a
    // duplicate provider.
    middleware: {
      provideInlineCompletionItems: (document, position, context, token, next): ProviderResult<any> => {
        if (!cfg().get<boolean>("inlineCompletion", true)) {
          return undefined;
        }
        return next(document, position, context, token);
      },
    },
  };

  client = new LanguageClient(
    "llmsp",
    "llmsp (FIM completion)",
    buildServerOptions(),
    clientOptions
  );
  await client.start();
}

async function stopClient(): Promise<void> {
  if (client) {
    await client.stop();
    client = undefined;
  }
}

export async function activate(context: ExtensionContext): Promise<void> {
  context.subscriptions.push(
    commands.registerCommand("llmsp.restart", async () => {
      await stopClient();
      await startClient();
      window.showInformationMessage("llmsp: server restarted");
    })
  );

  // Restart automatically when relevant settings change.
  context.subscriptions.push(
    workspace.onDidChangeConfiguration(async (e) => {
      if (e.affectsConfiguration("llmsp")) {
        // The inline toggle is read live by the middleware; everything else
        // (command, model, context) needs a server restart to take effect.
        const onlyInline =
          e.affectsConfiguration("llmsp.inlineCompletion") &&
          !e.affectsConfiguration("llmsp.serverCommand") &&
          !e.affectsConfiguration("llmsp.backend") &&
          !e.affectsConfiguration("llmsp.model") &&
          !e.affectsConfiguration("llmsp.modelFamily") &&
          !e.affectsConfiguration("llmsp.baseUrl") &&
          !e.affectsConfiguration("llmsp.languages") &&
          !e.affectsConfiguration("llmsp.context") &&
          !e.affectsConfiguration("llmsp.maxTokens") &&
          !e.affectsConfiguration("llmsp.temperature") &&
          !e.affectsConfiguration("llmsp.debounceMs") &&
          !e.affectsConfiguration("llmsp.requestTimeoutMs") &&
          !e.affectsConfiguration("llmsp.multiline");
        if (!onlyInline) {
          await stopClient();
          await startClient();
        }
      }
    })
  );

  await startClient();
}

export async function deactivate(): Promise<void> {
  await stopClient();
}
