#!/usr/bin/env python3
# Copyright 2026 Avnish Midha. All rights reserved.
# Author: Avnish Midha
# GitHub: avnishbm
# Purpose: Rebuild the selected PieceVocab tokenizer from its packaged corpus.
"""Rebuild the selected 5,000-visible-word tokenizer from cached pages."""
from __future__ import annotations

import json
from pathlib import Path

from rank_fourth_languages import prepare, train


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "candidate-corpus"
RANKING = ROOT / "search-visible-5000" / "ranking.json"
TOKENIZER = ROOT / "search-visible-5000" / "winner.tokenizer.json"


def main() -> int:
    report = json.loads(RANKING.read_text(encoding="utf-8"))
    winner = report["ranking"][0]
    languages = ("en", "hi", "te", winner["code"])
    texts = {
        code: (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
        for code in languages
    }
    prepared = prepare(texts, languages)
    tokenizer, verified_metrics = train(
        prepared, winner["weights"], verify_roundtrip=True
    )
    # Preserve the exhaustive-search record in the generated artifact while
    # independently verifying its counts against the packaged final corpus.
    for code in languages:
        if verified_metrics["rows"][code] != winner["rows"][code]:
            raise AssertionError(f"{code}: rebuilt metrics differ from ranking")
    meta = {
        "description": "5,000-visible-word PieceVocab winner; all writing directions included",
        "languages": list(languages),
        "vocab_size": len(tokenizer.vocab),
        "metrics": winner,
    }
    TOKENIZER.write_text(
        json.dumps(tokenizer.to_dict(meta), ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"Rebuilt {TOKENIZER.name}: vocab={len(tokenizer.vocab)}, "
        f"fourth={winner['code']}, score={winner['adjusted_score']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
