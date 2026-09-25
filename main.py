#!/usr/bin/env python3
"""
main.py

Root entry point alias for IBVAP prototype.
Hands off to app/main.py with app/ added to sys.path.
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "app"
sys.path.insert(0, str(APP_DIR))

from main import main  # noqa: E402

if __name__ == "__main__":
    main()
