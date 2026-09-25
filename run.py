#!/usr/bin/env python3
"""
run.py

Single entry point for the IBVAP prototype.

Usage:
    python run.py

This adds the app/ directory to sys.path (so the modules' plain imports,
e.g. `from config import ...`, work regardless of the current working
directory) and then hands off to app/main.py.
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "app"
sys.path.insert(0, str(APP_DIR))

from main import main  # noqa: E402

if __name__ == "__main__":
    main()
