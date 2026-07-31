#!/usr/bin/env python3
# Copyright 2026 Avnish Midha. All rights reserved.
# Author: Avnish Midha
# GitHub: avnishbm
# Purpose: Verify BPE fertility, unknown-token counts, and exact round trips.
"""Standalone score and round-trip evaluator for the final BPE."""
from __future__ import annotations

import json
import random
from pathlib import Path

from tokenizers import Tokenizer

from bpe_common import UNK, evaluate


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
TOKENIZER = ROOT / "tokenizer.json"
METRICS = ROOT / "metrics.json"
SAMPLES = {
    "en": "India's population is 1,428,627,663.",
    "hi": "भारत की जनसंख्या 1,428,627,663 है।",
    "te": "భారతదేశ జనాభా 1,428,627,663.",
    "sd": "ڀارت ڏکڻ ايشيا جو هڪ ملڪ آهي۔",
}


def main() -> int:
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    saved = json.loads(METRICS.read_text(encoding="utf-8"))
    languages = tuple(saved["languages"])
    texts = {
        code: (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
        for code in languages
    }
    result = evaluate(tokenizer, texts, languages, require_exact=True)
    probe_count = 0
    for code in languages:
        sample = SAMPLES[code]
        encoding = tokenizer.encode(sample)
        if UNK in encoding.tokens:
            raise AssertionError(f"{code}: sample contains unknown token")
        if tokenizer.decode(encoding.ids) != sample:
            raise AssertionError(f"{code}: explicit sample round-trip failed")
        probe_count += 1

    rng = random.Random(20260717)
    for code in languages:
        text = texts[code]
        for _ in range(50):
            start = rng.randrange(len(text) + 1)
            end = min(len(text), start + rng.randrange(301))
            sample = text[start:end]
            encoding = tokenizer.encode(sample)
            if UNK in encoding.tokens or tokenizer.decode(encoding.ids) != sample:
                raise AssertionError(f"{code}: substring round-trip failed")
            probe_count += 1
    result.update({
        "languages": list(languages),
        "rejection_sample_roundtrip": True,
        "independent_roundtrip_probes": probe_count,
    })
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
