# Copyright 2026 Avnish Midha. All rights reserved.
# Author: Avnish Midha
# GitHub: avnishbm
# Purpose: Define shared faithful BPE construction and evaluation helpers.

"""Shared construction and evaluation helpers for the faithful BPE approach."""
from __future__ import annotations

import math
from typing import Iterable

import regex
from tokenizers import Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer


VOCAB_SIZE = 10_000
UNK = "[UNK]"
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")


def faithful_units(text: str) -> int:
    return len(FAITHFUL_UNIT_RE.findall(text))


def make_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=UNK))
    # No normalizer: normalization would make exact reconstruction impossible
    # for canonically distinct input. Metaspace preserves corpus whitespace here.
    tokenizer.pre_tokenizer = Metaspace(
        replacement="▁", prepend_scheme="never", split=True
    )
    tokenizer.decoder = MetaspaceDecoder(
        replacement="▁", prepend_scheme="never"
    )
    return tokenizer


def train_tokenizer(
    texts: dict[str, str],
    languages: Iterable[str],
    weights: dict[str, int],
) -> Tokenizer:
    languages = tuple(languages)
    if set(languages) != set(weights):
        raise ValueError("weights must contain exactly the selected languages")
    if any(not isinstance(weight, int) or weight < 1 for weight in weights.values()):
        raise ValueError("all training weights must be positive integers")
    tokenizer = make_tokenizer()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=1,
        special_tokens=[UNK],
        show_progress=False,
    )
    tokenizer.train_from_iterator(
        (texts[code] for code in languages for _ in range(weights[code])),
        trainer=trainer,
        length=sum(weights.values()),
    )
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise AssertionError(
            f"expected {VOCAB_SIZE} vocabulary entries, got {tokenizer.get_vocab_size()}"
        )
    return tokenizer


def evaluate(
    tokenizer: Tokenizer,
    texts: dict[str, str],
    languages: Iterable[str],
    require_exact: bool = True,
) -> dict:
    rows = {}
    ratios = {}
    for code in languages:
        encoding = tokenizer.encode(texts[code])
        decoded = tokenizer.decode(encoding.ids)
        exact = decoded == texts[code]
        visible = "".join(decoded.split()) == "".join(texts[code].split())
        if require_exact and not exact:
            raise AssertionError(f"{code}: exact round-trip failed")
        if not visible:
            raise AssertionError(f"{code}: visible-text round-trip failed")
        unknown_tokens = encoding.tokens.count(UNK)
        if unknown_tokens:
            raise AssertionError(f"{code}: encoded with {unknown_tokens} unknown tokens")
        units = faithful_units(texts[code])
        ratio = len(encoding.ids) / units
        ratios[code] = ratio
        rows[code] = {
            "tokens": len(encoding.ids),
            "faithful_units": units,
            "ratio": ratio,
            "unknown_tokens": unknown_tokens,
            "exact_roundtrip": exact,
            "visible_roundtrip": visible,
        }
    spread = max(ratios.values()) - min(ratios.values())
    raw_score = math.inf if spread == 0 else 1000 / spread
    hindi_penalty = math.exp(max(0.0, ratios["hi"] / 1.2 - 1.0))
    return {
        "vocab_size": tokenizer.get_vocab_size(),
        "rows": rows,
        "spread": spread,
        "raw_score": raw_score,
        "hindi_penalty": hindi_penalty,
        "adjusted_score": raw_score / hindi_penalty,
        "all_under_1_2": all(ratio <= 1.2 for ratio in ratios.values()),
        "all_exact_roundtrip": all(row["exact_roundtrip"] for row in rows.values()),
    }
