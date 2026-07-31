#!/usr/bin/env python3
# Copyright 2026 Avnish Midha. All rights reserved.
# Author: Avnish Midha
# GitHub: avnishbm
# Purpose: Verify PieceVocab fertility, vocabulary limits, and exact round trips.
"""Standalone faithful evaluation of the selected PieceVocab tokenizer."""
from __future__ import annotations

import json
import math
import random
import unicodedata
from pathlib import Path

from piecevocab import load


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "candidate-corpus"
TOKENIZER = ROOT / "search-visible-5000" / "winner.tokenizer.json"
SAMPLE = "India's population is 1,428,627,663."
UNSEEN_SAMPLE = "Unseen Unicode survives exactly: 🧪 𐍈"
LANGUAGE_SAMPLES = {
    "en": "India's population is 1,428,627,663.",
    "hi": "भारत की जनसंख्या 1,428,627,663 है।",
    "te": "భారతదేశ జనాభా 1,428,627,663.",
    "yo": "Íńdíà jẹ́ orílẹ̀-èdè kan ní Gúúsù Éṣíà.",
}


def faithful_units(text: str) -> int:
    count = 0
    in_run = False
    for char in text:
        is_word = unicodedata.category(char)[0] in {"L", "M", "N"}
        if is_word:
            if not in_run:
                count += 1
            in_run = True
        else:
            in_run = False
            if not char.isspace():
                count += 1
    return count


def main() -> int:
    tokenizer = load(TOKENIZER)
    bundle = json.loads(TOKENIZER.read_text(encoding="utf-8"))
    languages = bundle["meta"]["languages"]
    if len(tokenizer.vocab) != 10_000:
        raise AssertionError(f"expected exactly 10000 vocabulary entries, got {len(tokenizer.vocab)}")
    probe_count = 0
    for sample in (SAMPLE, UNSEEN_SAMPLE, *LANGUAGE_SAMPLES.values()):
        if tokenizer.decode(tokenizer.encode(sample)) != sample:
            raise AssertionError(f"sample round-trip failed: {sample!r}")
        probe_count += 1

    rows = {}
    rng = random.Random(20260717)
    for code in languages:
        text = (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        if decoded != text:
            raise AssertionError(f"{code}: exact round-trip failed")
        # Exercise independent substring boundaries rather than relying only on
        # one whole-page encode call.
        for _ in range(50):
            start = rng.randrange(len(text) + 1)
            end = min(len(text), start + rng.randrange(301))
            probe = text[start:end]
            if tokenizer.decode(tokenizer.encode(probe)) != probe:
                raise AssertionError(f"{code}: substring round-trip failed")
            probe_count += 1
        units = faithful_units(text)
        rows[code] = {
            "tokens": len(ids),
            "faithful_units": units,
            "ratio": len(ids) / units,
            "exact_roundtrip": True,
        }
    ratios = {code: row["ratio"] for code, row in rows.items()}
    spread = max(ratios.values()) - min(ratios.values())
    raw_score = math.inf if spread == 0 else 1000 / spread
    hindi_penalty = math.exp(max(0.0, ratios["hi"] / 1.2 - 1.0))
    result = {
        "vocab_size": len(tokenizer.vocab),
        "languages": languages,
        "sample_roundtrip": True,
        "unseen_unicode_roundtrip": True,
        "independent_roundtrip_probes": probe_count,
        "rows": rows,
        "spread": spread,
        "raw_score": raw_score,
        "hindi_penalty": hindi_penalty,
        "adjusted_score": raw_score / hindi_penalty,
        "all_under_1_2": all(ratio <= 1.2 for ratio in ratios.values()),
        "all_exact_roundtrip": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
