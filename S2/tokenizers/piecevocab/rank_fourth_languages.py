#!/usr/bin/env python3
# Copyright 2026 Avnish Midha. All rights reserved.
# Author: Avnish Midha
# GitHub: avnishbm
# Purpose: Rank every fourth-language candidate for the PieceVocab approach.
"""Rank each available fourth language for the faithful PieceVocab tokenizer."""
from __future__ import annotations

import argparse
import heapq
import json
import math
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from piecevocab import (
    SPECIAL_FALLBACK_TOKENS,
    PieceVocab,
    split_pieces,
)


FIXED = ("en", "hi", "te")
VOCAB_SIZE = 10_000


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


@dataclass
class Prepared:
    languages: tuple[str, ...]
    texts: dict[str, str]
    units: dict[str, int]
    chars: list[str]
    piece_budget: int
    base_tokens: dict[str, int]
    # (piece, token savings in each language)
    candidates: list[tuple[str, tuple[int, ...]]]


def prepare(texts: dict[str, str], languages: tuple[str, ...]) -> Prepared:
    pieces = {code: split_pieces(texts[code]) for code in languages}
    frequencies = {code: Counter(pieces[code]) for code in languages}
    units = {code: faithful_units(texts[code]) for code in languages}
    chars = sorted(set().union(*(set(texts[code]) for code in languages)))
    piece_budget = VOCAB_SIZE - len(chars) - len(SPECIAL_FALLBACK_TOKENS)
    if piece_budget <= 0:
        raise ValueError("character and Unicode fallback tokens exceed vocabulary budget")
    all_pieces = set().union(*(set(frequencies[code]) for code in languages))
    candidates = []
    for piece in all_pieces:
        if len(piece) <= 1 or piece in chars:
            continue
        saving = len(piece) - 1
        candidates.append((
            piece,
            tuple(frequencies[code][piece] * saving for code in languages),
        ))
    candidates.sort(key=lambda item: item[0])  # deterministic heap tie behavior
    return Prepared(
        languages=languages,
        texts=texts,
        units=units,
        chars=chars,
        piece_budget=piece_budget,
        base_tokens={code: sum(map(len, pieces[code])) for code in languages},
        candidates=candidates,
    )


def train(
    prepared: Prepared,
    weights: dict[str, float],
    verify_roundtrip: bool = False,
) -> tuple[PieceVocab, dict]:
    languages = prepared.languages

    def utility(candidate: tuple[str, tuple[int, ...]]) -> tuple[float, str]:
        piece, savings = candidate
        value = sum(
            weights[code] * savings[index] / prepared.units[code]
            for index, code in enumerate(languages)
        )
        return value, piece

    selected_rows = heapq.nlargest(
        prepared.piece_budget, prepared.candidates, key=utility
    )
    selected = [piece for piece, _ in selected_rows]
    tokenizer = PieceVocab(
        prepared.chars + list(SPECIAL_FALLBACK_TOKENS) + selected,
        selected,
    )
    rows = {}
    ratios = {}
    for index, code in enumerate(languages):
        token_count = prepared.base_tokens[code] - sum(
            savings[index] for _, savings in selected_rows
        )
        if verify_roundtrip:
            ids = tokenizer.encode(prepared.texts[code])
            if len(ids) != token_count:
                raise AssertionError(f"{code}: analytic and encoded counts disagree")
            if tokenizer.decode(ids) != prepared.texts[code]:
                raise AssertionError(f"{code}: exact round-trip failed")
        ratio = token_count / prepared.units[code]
        ratios[code] = ratio
        rows[code] = {
            "tokens": token_count,
            "faithful_units": prepared.units[code],
            "ratio": ratio,
            "exact_roundtrip": True,
        }
    spread = max(ratios.values()) - min(ratios.values())
    raw_score = math.inf if spread == 0 else 1000 / spread
    hindi_penalty = math.exp(max(0.0, ratios["hi"] / 1.2 - 1.0))
    metrics = {
        "vocab_size": len(tokenizer.vocab),
        "character_tokens": len(prepared.chars),
        "unicode_fallback_tokens": len(SPECIAL_FALLBACK_TOKENS),
        "whole_piece_tokens": len(selected),
        "weights": dict(weights),
        "rows": rows,
        "spread": spread,
        "raw_score": raw_score,
        "hindi_penalty": hindi_penalty,
        "adjusted_score": raw_score / hindi_penalty,
        "all_under_1_2": all(ratio <= 1.2 for ratio in ratios.values()),
        "all_exact_roundtrip": True,
    }
    return tokenizer, metrics


def objective(metrics: dict) -> tuple[float, float]:
    # Maximize adjusted score, then prefer lower worst fertility.
    return metrics["adjusted_score"], -max(
        row["ratio"] for row in metrics["rows"].values()
    )


def baseline_weights(prepared: Prepared) -> dict[str, float]:
    # This makes normalized utility proportional to aggregate token savings.
    en_units = prepared.units["en"]
    return {code: prepared.units[code] / en_units for code in prepared.languages}


def optimize(
    prepared: Prepared,
    initial_weights: dict[str, float],
    passes: int,
) -> tuple[PieceVocab, dict]:
    weights = dict(initial_weights)
    weights["en"] = 1.0
    best_tokenizer, best_metrics = train(prepared, weights)
    # Multiplicatively give more selection utility to languages whose fertility
    # is above English and less to those below it. This converges much faster
    # than a Cartesian weight grid and keeps an exhaustive all-candidate run practical.
    for iteration in range(passes):
        _, metrics = train(prepared, weights)
        if objective(metrics) > objective(best_metrics):
            best_tokenizer, best_metrics = train(prepared, weights)
        ratios = {code: metrics["rows"][code]["ratio"] for code in prepared.languages}
        learning_rate = 3.0 * (0.9**iteration)
        for code in prepared.languages[1:]:
            weights[code] *= math.exp(
                learning_rate * (ratios[code] - ratios["en"])
            )
    return best_tokenizer, best_metrics


def read_names(corpus: Path) -> dict[str, str]:
    names = {}
    manifest = corpus / "languages.json"
    if manifest.exists():
        for page in json.loads(manifest.read_text(encoding="utf-8")):
            names[page["code"]] = page.get("name", page["code"])
    for meta_path in corpus.glob("*.meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        code = meta.get("code", meta.get("lang", meta_path.name.split(".")[0]))
        names.setdefault(code, meta.get("name", code))
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "search-results",
    )
    parser.add_argument("--include", nargs="*", help="Only evaluate these fourth languages")
    parser.add_argument("--min-units", type=int, default=1)
    parser.add_argument(
        "--optimize-top", type=int, default=25,
        help="Coordinate-optimize top N baseline candidates; 0 means all",
    )
    parser.add_argument("--passes", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing = [code for code in FIXED if not (args.corpus / f"{code}.faithful.txt").exists()]
    if missing:
        raise FileNotFoundError(f"fixed-language corpus files missing: {missing}")
    fixed_texts = {
        code: (args.corpus / f"{code}.faithful.txt").read_text(encoding="utf-8")
        for code in FIXED
    }
    candidates = sorted(
        path.name.removesuffix(".faithful.txt")
        for path in args.corpus.glob("*.faithful.txt")
        if path.name.removesuffix(".faithful.txt") not in FIXED
    )
    if args.include:
        candidates = [code for code in candidates if code in set(args.include)]
    names = read_names(args.corpus)

    def prepare_candidate(fourth: str) -> Prepared:
        text = (args.corpus / f"{fourth}.faithful.txt").read_text(encoding="utf-8")
        return prepare({**fixed_texts, fourth: text}, (*FIXED, fourth))

    baseline_results = []
    for index, fourth in enumerate(candidates, 1):
        text = (args.corpus / f"{fourth}.faithful.txt").read_text(encoding="utf-8")
        if faithful_units(text) < args.min_units:
            continue
        prepared = prepare({**fixed_texts, fourth: text}, (*FIXED, fourth))
        _, metrics = train(prepared, baseline_weights(prepared))
        baseline_results.append({"code": fourth, "name": names.get(fourth, fourth), **metrics})
        print(
            f"baseline [{index}/{len(candidates)}] {fourth}: "
            f"score={metrics['adjusted_score']:.2f} spread={metrics['spread']:.6f}"
        )

    baseline_results.sort(key=lambda row: row["adjusted_score"], reverse=True)
    optimize_count = len(baseline_results) if args.optimize_top == 0 else args.optimize_top
    optimized_codes = {row["code"] for row in baseline_results[:optimize_count]}
    final_results = []
    winner_tokenizer = None
    winner_metrics = None
    for row in baseline_results:
        code = row["code"]
        if code in optimized_codes:
            prepared = prepare_candidate(code)
            tokenizer, metrics = optimize(
                prepared, baseline_weights(prepared), args.passes
            )
            row = {"code": code, "name": names.get(code, code), "optimized": True, **metrics}
            print(f"optimized {code}: score={metrics['adjusted_score']:.2f}")
        else:
            tokenizer = None
            row["optimized"] = False
        final_results.append(row)
        if winner_metrics is None or row["adjusted_score"] > winner_metrics["adjusted_score"]:
            winner_tokenizer = tokenizer
            winner_metrics = row

    final_results.sort(key=lambda row: row["adjusted_score"], reverse=True)
    # If a non-optimized baseline somehow wins, optimize it before saving.
    if final_results and not final_results[0]["optimized"]:
        code = final_results[0]["code"]
        prepared = prepare_candidate(code)
        _, metrics = optimize(
            prepared, baseline_weights(prepared), args.passes
        )
        final_results[0] = {
            "code": code, "name": names.get(code, code), "optimized": True, **metrics
        }
        final_results.sort(key=lambda row: row["adjusted_score"], reverse=True)
    if final_results:
        winner_metrics = final_results[0]
        code = winner_metrics["code"]
        prepared = prepare_candidate(code)
        winner_tokenizer, winner_metrics_raw = train(
            prepared, winner_metrics["weights"], verify_roundtrip=True
        )
        winner_metrics = {
            "code": code,
            "name": names.get(code, code),
            "optimized": winner_metrics["optimized"],
            **winner_metrics_raw,
        }
        final_results[0] = winner_metrics

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "fixed_languages": list(FIXED),
        "candidate_count": len(final_results),
        "optimization": {
            "top_n": args.optimize_top,
            "passes": args.passes,
            "warning": "Candidates outside top_n have baseline scores only; use --optimize-top 0 for exhaustive optimization.",
        },
        "ranking": final_results,
    }
    (args.output / "ranking.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if winner_tokenizer is not None and winner_metrics is not None:
        meta = {
            "description": "Winning faithful PieceVocab from fourth-language search",
            "languages": [*FIXED, winner_metrics["code"]],
            "vocab_size": len(winner_tokenizer.vocab),
            "metrics": winner_metrics,
        }
        (args.output / "winner.tokenizer.json").write_text(
            json.dumps(winner_tokenizer.to_dict(meta), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        print(
            f"WINNER {winner_metrics['code']} ({winner_metrics['name']}): "
            f"adjusted score={winner_metrics['adjusted_score']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
