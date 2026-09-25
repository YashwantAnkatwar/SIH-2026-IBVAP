#!/usr/bin/env python3
"""Runs every test module in this directory and exits non-zero on any failure."""

import subprocess
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
test_files = sorted(TEST_DIR.glob("test_*.py"))

failures = []
for f in test_files:
    print(f"\n=== {f.name} ===")
    result = subprocess.run([sys.executable, str(f)])
    if result.returncode != 0:
        failures.append(f.name)

print("\n" + "=" * 50)
if failures:
    print(f"FAILED: {', '.join(failures)}")
    sys.exit(1)
else:
    print(f"ALL {len(test_files)} TEST MODULES PASSED")
