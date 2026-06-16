"""Backend selection and dispatch tests (no network)."""

import pytest

import slop_frames as sf
from slop_frames import llm
from slop_frames.config import Settings, settings as global_settings
from slop_frames.providers import (
    AnthropicProvider,
    GoogleProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
    get_provider,
    resolve_model,
)


def test_resolve_model_defaults():
    assert resolve_model("anthropic", "") == "claude-opus-4-8"
    assert resolve_model("openai", "") == "gpt-4o"
    assert resolve_model("google", "") == "gemini-2.0-flash"
    assert resolve_model("ollama", "") == "llama3"
    # An explicit model always wins.
    assert resolve_model("openai", "gpt-4o-mini") == "gpt-4o-mini"


@pytest.mark.parametrize(
    "name,cls",
    [
        ("anthropic", AnthropicProvider),
        ("openai", OpenAIProvider),
        ("google", GoogleProvider),
        ("ollama", OllamaProvider),
    ],
)
def test_get_provider_constructs_each_backend(name, cls):
    provider = get_provider(Settings(provider=name, api_key="key"))
    assert isinstance(provider, cls)
    assert provider.name == name


def test_get_provider_unknown_raises():
    with pytest.raises(ProviderError):
        get_provider(Settings(provider="grok"))


def test_ollama_provider_receives_host():
    provider = get_provider(
        Settings(provider="ollama", ollama_url="http://example:1234/api/generate")
    )
    assert isinstance(provider, OllamaProvider)
    assert provider.host == "http://example:1234/api/generate"


def test_explicit_model_overrides_provider_default():
    provider = get_provider(
        Settings(provider="anthropic", model="claude-haiku-4-5", api_key="key")
    )
    assert provider.model == "claude-haiku-4-5"


def test_openai_ready_requires_credential(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert OpenAIProvider("gpt-4o", None).ready() is False
    assert OpenAIProvider("gpt-4o", "sk-test").ready() is True


def test_live_without_credential_degrades_to_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SLOP_FRAMES_API_KEY", raising=False)

    previous = (global_settings.provider, global_settings.api_key, global_settings.live)
    try:
        global_settings.provider = "openai"
        global_settings.api_key = None
        global_settings.live = True
        out = llm.complete("anything", fallback=lambda: "OFFLINE")
        assert out == "OFFLINE"
    finally:
        (
            global_settings.provider,
            global_settings.api_key,
            global_settings.live,
        ) = previous


def test_configure_accepts_provider_kwargs():
    try:
        sf.configure(provider="google", model="gemini-2.5-pro")
        assert global_settings.provider == "google"
        assert global_settings.model == "gemini-2.5-pro"
    finally:
        sf.configure(provider="anthropic", model="")
