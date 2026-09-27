import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
APP_DIR = ROOT_DIR / "app"

# Ensure app and project root are in sys.path
for p in (APP_DIR, ROOT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# If pytest was invoked with system python, automatically mount the project's venv site-packages
venv_dir = ROOT_DIR / "venv"
if venv_dir.exists():
    for site_pkg in venv_dir.glob("lib/python*/site-packages"):
        if site_pkg.is_dir() and str(site_pkg) not in sys.path:
            sys.path.insert(0, str(site_pkg))
