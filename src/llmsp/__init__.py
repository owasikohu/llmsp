"""llmsp — a Fill-In-the-Middle (FIM) LLM code-completion language server.

The package is organised in three areas:

* :mod:`llmsp.fim` — provider-agnostic FIM backends behind one interface.
* :mod:`llmsp.context` — the context-engineering pipeline (the project's
  differentiator): budgeted single-file windowing, cross-file retrieval and
  structural context.
* :mod:`llmsp.server` — the pygls glue tying documents, debouncing and
  cancellation to the backend.
"""

__version__ = "0.1.0"
