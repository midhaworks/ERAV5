#!/usr/bin/env python3
# Copyright 2026 Avnish Midha. All rights reserved.
# Author: Avnish Midha
# GitHub: avnishbm
# Purpose: Package the exhaustive BPE search winner and exact corpus.
"""Promote the BPE search winner and exact corpus into tokenizers/bpe."""
from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEARCH = ROOT / "search-results"
SOURCE_CORPUS = ROOT.parent / "piecevocab" / "candidate-corpus"
CORPUS = ROOT / "corpus"


def main() -> int:
    meta = json.loads((SEARCH / "winner.meta.json").read_text(encoding="utf-8"))
    languages = meta["languages"]
    shutil.copy2(SEARCH / "winner.tokenizer.json", ROOT / "tokenizer.json")
    (ROOT / "metrics.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    CORPUS.mkdir(parents=True, exist_ok=True)
    expected = {
        f"{code}.{suffix}"
        for code in languages
        for suffix in ("faithful.txt", "faithful.md", "meta.json")
    }
    for existing in CORPUS.iterdir():
        if existing.is_file() and existing.name not in expected:
            existing.unlink()
    for code in languages:
        for suffix in ("faithful.txt", "faithful.md", "meta.json"):
            source = SOURCE_CORPUS / f"{code}.{suffix}"
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, CORPUS / source.name)
    print(
        f"Promoted {meta['fourth_language']['code']} "
        f"({meta['fourth_language']['name']}), score={meta['metrics']['adjusted_score']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
