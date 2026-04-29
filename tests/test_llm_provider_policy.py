"""Tests for supported LLM provider policy."""

import importlib.util
import sys
from pathlib import Path

import pytest

from src.llm_client import OpenRouterClient, get_llm_client


def test_auto_provider_ignores_direct_provider_keys(monkeypatch):
    """Only OpenRouter should be auto-detected as a remote provider."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-direct-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-direct-anthropic")

    assert get_llm_client("auto") is None


def test_openrouter_provider_is_supported(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter")

    client = get_llm_client("openrouter")

    assert isinstance(client, OpenRouterClient)
    assert client.provider == "openrouter"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_direct_providers_are_not_supported(provider):
    with pytest.raises(ValueError, match="Valid options: 'auto', 'openrouter', 'none'"):
        get_llm_client(provider)


def test_query_cli_provider_choices_are_openrouter_only(monkeypatch):
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    query_path = scripts_dir / "query.py"
    sys.path.insert(0, str(scripts_dir))
    try:
        spec = importlib.util.spec_from_file_location("finance_rag_query_cli", query_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts_dir))

    monkeypatch.setattr(sys, "argv", ["query.py", "--provider", "openrouter", "hello"])
    args = module.parse_args()
    assert args.provider == "openrouter"

    monkeypatch.setattr(sys, "argv", ["query.py", "--provider", "openai", "hello"])
    with pytest.raises(SystemExit):
        module.parse_args()
