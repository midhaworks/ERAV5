#!/usr/bin/env python3
"""Build the lightweight static S4 submission without corpus artifacts."""
from pathlib import Path
import shutil


ROOT = Path(__file__).parent
DIST = ROOT / "dist"
ASSETS = ("index.html", "styles.css", "app.js", "favicon.svg")


def main() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "data").mkdir(parents=True)
    for asset in ASSETS:
        shutil.copy2(ROOT / asset, DIST / asset)
    shutil.copy2(
        ROOT / "data" / "cleanup-report.json",
        DIST / "data" / "cleanup-report.json",
    )
    files = sorted(path.relative_to(DIST).as_posix() for path in DIST.rglob("*") if path.is_file())
    print(f"Built {len(files)} files: {', '.join(files)}")


if __name__ == "__main__":
    main()
