#!/usr/bin/env python3
# Copyright 2026 Avnish Midha. All rights reserved.
# Author: Avnish Midha
# GitHub: avnishbm
# Purpose: Rebuild the selected BPE tokenizer from its packaged corpus.
"""Rebuild the final shared BPE from its packaged four-page corpus."""
from __future__ import annotations

import json
from pathlib import Path

from bpe_common import evaluate, train_tokenizer


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
METRICS = ROOT / "metrics.json"
TOKENIZER = ROOT / "tokenizer.json"


def main() -> int:
    saved = json.loads(METRICS.read_text(encoding="utf-8"))
    languages = tuple(saved["languages"])
    weights = saved["weights"]
    texts = {
        code: (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
        for code in languages
    }
    tokenizer = train_tokenizer(texts, languages, weights)
    rebuilt = evaluate(tokenizer, texts, languages)
    for code in languages:
        if rebuilt["rows"][code] != saved["metrics"]["rows"][code]:
            raise AssertionError(f"{code}: rebuilt metrics differ from saved metrics")
    tokenizer.save(str(TOKENIZER))
    print(
        f"Rebuilt {TOKENIZER.name}: vocab={tokenizer.get_vocab_size()}, "
        f"fourth={languages[-1]}, score={rebuilt['adjusted_score']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
