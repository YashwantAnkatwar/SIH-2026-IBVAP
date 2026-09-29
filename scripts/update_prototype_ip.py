#!/usr/bin/env python3
"""
update_prototype_ip.py

Quick utility to update the public Prototype IP across README.md and documentation
whenever your VM or static IP changes.

Usage:
    python3 scripts/update_prototype_ip.py <NEW_IP_OR_DOMAIN>

Example:
    python3 scripts/update_prototype_ip.py 8.234.88.96
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
README_FILE = PROJECT_ROOT / "README.md"


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/update_prototype_ip.py <NEW_IP_OR_DOMAIN>")
        print("Example: python3 scripts/update_prototype_ip.py 8.234.88.96")
        sys.exit(1)

    new_target = sys.argv[1].strip().replace("http://", "").replace(":8000/", "").replace(":8000", "").strip("/")
    new_url = f"http://{new_target}:8000/"

    if not README_FILE.exists():
        print(f"Error: {README_FILE} not found.")
        sys.exit(1)

    content = README_FILE.read_text(encoding="utf-8")

    # Match any http://<ip_or_domain>:8000/ pattern in README
    pattern = r"http://(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|[a-zA-Z0-9.-]+):8000/?"
    updated_content, count = re.subn(pattern, new_url, content)

    if count == 0:
        print("No matching prototype URLs found in README.md to replace.")
    else:
        README_FILE.write_text(updated_content, encoding="utf-8")
        print(f"[✓] Successfully updated {count} URL instance(s) in README.md!")
        print(f"[✓] New Prototype URL: {new_url}")
        print("\nNext step: Commit and push the updated README to GitHub:")
        print("   git add README.md")
        print(f"   git commit -m \"Update live prototype URL to {new_url}\"")
        print("   git push origin main")


if __name__ == "__main__":
    main()
