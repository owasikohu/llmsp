"""Config parsing from initialization_options."""


from llmsp.config import Config


def test_defaults_are_offline_safe():
    cfg = Config.from_mapping(None)
    assert cfg.backend == "mock"
    assert cfg.context.cross_file is True
    assert cfg.context.max_prompt_tokens == 1536
    # The reserve must be at most half the budget so the file window isn't starved.
    assert cfg.context.cross_file_tokens <= cfg.context.max_prompt_tokens // 2


def test_nested_context_merge():
    cfg = Config.from_mapping(
        {
            "backend": "ollama",
            "model": "qwen2.5-coder",
            "max_tokens": 256,
            "context": {"cross_file": False, "max_snippets": 2, "retrieval": "bm25"},
        }
    )
    assert cfg.backend == "ollama"
    assert cfg.model == "qwen2.5-coder"
    assert cfg.max_tokens == 256
    assert cfg.context.cross_file is False
    assert cfg.context.max_snippets == 2
    assert cfg.context.retrieval == "bm25"
    # untouched nested defaults survive
    assert cfg.context.ring_chunks == 16


def test_stop_list_becomes_tuple():
    cfg = Config.from_mapping({"stop": ["\n", "\n\n"]})
    assert cfg.stop == ("\n", "\n\n")


def test_unknown_keys_ignored():
    cfg = Config.from_mapping({"backend": "mock", "totally_unknown": 123})
    assert cfg.backend == "mock"
    assert not hasattr(cfg, "totally_unknown")


def test_env_api_key_fallback(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
    cfg = Config.from_mapping({"backend": "deepseek"})
    assert cfg.api_key == "sk-test-123"


def test_explicit_api_key_wins(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
    cfg = Config.from_mapping({"backend": "deepseek", "api_key": "sk-explicit"})
    assert cfg.api_key == "sk-explicit"
