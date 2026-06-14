"""FIM template rendering — exact sentinels are load-bearing."""

from llmsp.fim.base import Snippet
from llmsp.fim.templates import (
    custom_template,
    get_template,
    infer_family,
    template_for,
)


def test_qwen_psm_render():
    t = get_template("qwen2.5-coder")
    assert t.render("PRE", "SUF") == "<|fim_prefix|>PRE<|fim_suffix|>SUF<|fim_middle|>"


def test_deepseek_uses_exact_fullwidth_tokens():
    t = get_template("deepseek")
    out = t.render("PRE", "SUF")
    # Full-width bar U+FF5C and U+2581 — verify exact code points, not lookalikes.
    assert out == "<｜fim▁begin｜>PRE<｜fim▁hole｜>SUF<｜fim▁end｜>"


def test_starcoder2_is_single_bracket_not_pipe():
    t = get_template("starcoder2")
    assert t.render("P", "S") == "<fim_prefix>P<fim_suffix>S<fim_middle>"
    assert "<|fim_prefix|>" not in t.render("P", "S")


def test_codellama_psm_spacing():
    t = get_template("codellama")
    assert t.render("P", "S") == "<PRE> P <SUF>S <MID>"
    assert "<EOT>" in t.stop


def test_spm_ordering_places_suffix_first():
    t = custom_template("<P>", "<S>", "<M>", order="spm")
    assert t.render("PRE", "SUF") == "<S>SUF<P>PRE<M>"


def test_repo_level_packing_with_file_sep():
    t = get_template("qwen")
    extra = [Snippet("def helper(): ...", "utils/h.py", 0.5)]
    out = t.render("PRE", "SUF", extra, repo="myrepo")
    assert "<|repo_name|>myrepo" in out
    assert "<|file_sep|>utils/h.py" in out
    assert "def helper(): ..." in out
    # The completed file's FIM body still follows.
    assert out.endswith("<|fim_prefix|>PRE<|fim_suffix|>SUF<|fim_middle|>")


def test_non_repo_family_ignores_extra_in_render():
    t = get_template("codellama")  # no file_sep
    out = t.render("PRE", "SUF", [Snippet("x", "f.py")])
    assert out == "<PRE> PRE <SUF>SUF <MID>"


def test_infer_family_and_template_for():
    assert infer_family("qwen2.5-coder:7b") == "qwen"
    assert infer_family("deepseek-coder-v2") == "deepseek"
    assert infer_family("starcoder2-3b") == "starcoder2"
    assert infer_family("gpt-4o") is None
    assert template_for(None, "qwen2.5-coder").family == "qwen"
    assert template_for("starcoder2", "whatever").family == "starcoder2"
    assert template_for(None, "unknown-model") is None


def test_sentinels_include_stop_tokens():
    t = get_template("qwen")
    s = t.sentinels()
    assert "<|fim_prefix|>" in s and "<|endoftext|>" in s and "<|file_sep|>" in s
