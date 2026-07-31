#!/usr/bin/env python3
"""Build the static S5 mixture-composer submission."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).parent
DIST = ROOT / "dist"
FILES = (
    "README.md",
    "app.js",
    "favicon.svg",
    "index.html",
    "mixture-plan.json",
    "styles.css",
)


def main() -> int:
    DIST.mkdir(exist_ok=True)
    for name in FILES:
        shutil.copy2(ROOT / name, DIST / name)
    print(f"Built {DIST} with {len(FILES)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
