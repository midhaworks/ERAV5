#!/usr/bin/env python3
# Copyright 2026 Avnish Midha. All rights reserved.
# Author: Avnish Midha
# GitHub: avnishbm
# Purpose: Package the winning PieceVocab tokenizer and its exact corpus.
"""Package the 5,000-visible-word winner and its exact four snapshots."""
from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CANDIDATE_CORPUS = ROOT / "candidate-corpus"
SEARCH = ROOT / "search-visible-5000"
FINAL_CORPUS = SEARCH / "corpus"


def main() -> int:
    report = json.loads((SEARCH / "ranking.json").read_text(encoding="utf-8"))
    if not report["ranking"]:
        raise ValueError("ranking contains no candidates")
    winner = report["ranking"][0]
    languages = ["en", "hi", "te", winner["code"]]
    tokenizer_path = SEARCH / "winner.tokenizer.json"
    bundle = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    if bundle["meta"]["languages"] != languages:
        raise AssertionError("ranking winner and winner.tokenizer.json disagree")

    FINAL_CORPUS.mkdir(parents=True, exist_ok=True)
    for code in languages:
        for suffix in ("faithful.txt", "faithful.md", "meta.json"):
            source = CANDIDATE_CORPUS / f"{code}.{suffix}"
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, FINAL_CORPUS / source.name)

    metrics = {
        "variant": "faithful_piecevocab_language_search_winner",
        "selection_policy": "maximum adjusted score after exhaustive 20-pass optimization",
        "candidate_count": report["candidate_count"],
        "languages": languages,
        **{key: value for key, value in winner.items() if key not in {"code", "name"}},
        "fourth_language": {"code": winner["code"], "name": winner["name"]},
    }
    (SEARCH / "winner.metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Packaged {winner['code']} ({winner['name']}), "
        f"score={winner['adjusted_score']:.2f}, languages={languages}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
