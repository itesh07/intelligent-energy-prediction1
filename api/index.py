"""
Vercel Serverless Function entry point.
Imports the Flask application from app.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app import app
