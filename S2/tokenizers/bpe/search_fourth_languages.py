#!/usr/bin/env python3
# Copyright 2026 Avnish Midha. All rights reserved.
# Author: Avnish Midha
# GitHub: avnishbm
# Purpose: Exhaustively rank fourth-language candidates for the BPE approach.
"""Train and rank a shared faithful 10k BPE for every fourth language."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bpe_common import evaluate, faithful_units, train_tokenizer


ROOT = Path(__file__).resolve().parent
FIXED = ("en", "hi", "te")
BASE_WEIGHTS = {"en": 3, "hi": 4, "te": 4}

# A small, declared second-stage search. Every candidate's comparable baseline
# remains the reference profile (3,4,4,2); these profiles tune only finalists.
FINALIST_PROFILES = (
    (3, 4, 4, 1), (3, 4, 4, 2), (3, 4, 4, 3), (3, 4, 4, 4),
    (3, 4, 5, 1), (3, 4, 5, 2), (3, 4, 5, 3), (3, 4, 5, 4),
    (3, 4, 6, 1), (3, 4, 6, 2), (3, 4, 6, 3), (3, 4, 6, 4),
    (3, 5, 6, 1), (3, 5, 6, 2), (3, 5, 6, 3), (3, 5, 6, 4),
)


def names(corpus: Path) -> dict[str, str]:
    manifest = corpus / "languages.json"
    if not manifest.exists():
        return {}
    return {
        row["code"]: row.get("name", row["code"])
        for row in json.loads(manifest.read_text(encoding="utf-8"))
    }


def load_text(corpus: Path, code: str) -> str:
    return (corpus / f"{code}.faithful.txt").read_text(encoding="utf-8")


def weight_dict(languages: tuple[str, ...], profile: tuple[int, ...]) -> dict[str, int]:
    return dict(zip(languages, profile, strict=True))


def optimize_candidate(
    corpus: Path,
    fixed_texts: dict[str, str],
    fourth: str,
) -> dict:
    languages = (*FIXED, fourth)
    texts = {**fixed_texts, fourth: load_text(corpus, fourth)}
    best_profile = None
    best_metrics = None
    for profile in FINALIST_PROFILES:
        weights = weight_dict(languages, profile)
        tokenizer = train_tokenizer(texts, languages, weights)
        metrics = evaluate(tokenizer, texts, languages)
        if best_metrics is None or metrics["adjusted_score"] > best_metrics["adjusted_score"]:
            best_profile = weights
            best_metrics = metrics
    return {"weights": best_profile, **best_metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", type=Path,
        default=ROOT.parent / "piecevocab" / "candidate-corpus",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "search-results")
    parser.add_argument("--include", nargs="*")
    parser.add_argument(
        "--optimize-top", type=int, default=0,
        help="Optimize top N baselines; 0 exhaustively optimizes every candidate",
    )
    parser.add_argument(
        "--reuse-baseline", action="store_true",
        help="Reuse baseline rows and completed profile searches from output/ranking.json",
    )
    parser.add_argument("--min-units", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    fixed_texts = {code: load_text(args.corpus, code) for code in FIXED}
    candidate_codes = sorted(
        path.name.removesuffix(".faithful.txt")
        for path in args.corpus.glob("*.faithful.txt")
        if path.name.removesuffix(".faithful.txt") not in FIXED
    )
    if args.include:
        requested = set(args.include)
        candidate_codes = [code for code in candidate_codes if code in requested]
    language_names = names(args.corpus)

    prior_optimized = {}
    prior_path = output / "ranking.json"
    checkpoint_path = output / "profile-checkpoint.json"
    if args.reuse_baseline and prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        prior_rows = {row["code"]: row for row in prior["ranking"]}
        baseline = []
        for fourth in candidate_codes:
            row = prior_rows.get(fourth)
            if row is None:
                raise ValueError(f"cached ranking has no baseline for {fourth}")
            baseline.append({
                "code": fourth,
                "name": row["name"],
                "baseline_weights": row["baseline_weights"],
                "baseline": row["baseline"],
            })
            if row.get("optimized") is not None:
                prior_optimized[fourth] = row["optimized"]
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            prior_optimized.update(checkpoint.get("optimized", {}))
        print(
            f"Reused {len(baseline)} baselines and {len(prior_optimized)} "
            "completed profile searches",
            flush=True,
        )
    else:
        baseline = []
        for index, fourth in enumerate(candidate_codes, 1):
            languages = (*FIXED, fourth)
            texts = {**fixed_texts, fourth: load_text(args.corpus, fourth)}
            if faithful_units(texts[fourth]) < args.min_units:
                continue
            weights = {**BASE_WEIGHTS, fourth: 2}
            tokenizer = train_tokenizer(texts, languages, weights)
            metrics = evaluate(tokenizer, texts, languages)
            baseline.append({
                "code": fourth,
                "name": language_names.get(fourth, fourth),
                "baseline_weights": weights,
                "baseline": metrics,
            })
            print(
                f"[{index}/{len(candidate_codes)}] {fourth}: "
                f"score={metrics['adjusted_score']:.2f} spread={metrics['spread']:.6f}",
                flush=True,
            )

    baseline.sort(key=lambda row: row["baseline"]["adjusted_score"], reverse=True)
    finalist_count = (
        len(baseline) if args.optimize_top == 0
        else min(args.optimize_top, len(baseline))
    )
    finalists = {row["code"] for row in baseline[:finalist_count]}
    rows_by_code = {row["code"]: row for row in baseline}
    pending = []
    for row in baseline:
        fourth = row["code"]
        if fourth not in finalists:
            row["optimized"] = None
        elif fourth in prior_optimized:
            row["optimized"] = prior_optimized[fourth]
        else:
            pending.append(fourth)

    if pending:
        for completed, fourth in enumerate(pending, 1):
            optimized = optimize_candidate(args.corpus, fixed_texts, fourth)
            rows_by_code[fourth]["optimized"] = optimized
            prior_optimized[fourth] = optimized
            checkpoint_path.write_text(
                json.dumps(
                    {"profiles": [list(profile) for profile in FINALIST_PROFILES],
                     "optimized": prior_optimized},
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            print(
                f"optimized [{completed}/{len(pending)}] {fourth}: "
                f"score={optimized['adjusted_score']:.2f}",
                flush=True,
            )

    final_rows = list(rows_by_code.values())
    final_rows.sort(
        key=lambda row: (
            row["optimized"]["adjusted_score"]
            if row["optimized"] is not None
            else row["baseline"]["adjusted_score"]
        ),
        reverse=True,
    )
    report = {
        "fixed_languages": list(FIXED),
        "candidate_count": len(final_rows),
        "baseline_policy": {**BASE_WEIGHTS, "fourth": 2},
        "finalist_count": finalist_count,
        "optimization_scope": (
            "exhaustive_all_candidates"
            if finalist_count == len(final_rows)
            else "baseline_top_n_only"
        ),
        "finalist_profiles": [list(profile) for profile in FINALIST_PROFILES],
        "ranking": final_rows,
    }
    (output / "ranking.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if final_rows and final_rows[0]["optimized"] is not None:
        best_row = final_rows[0]
        fourth = best_row["code"]
        languages = (*FIXED, fourth)
        texts = {**fixed_texts, fourth: load_text(args.corpus, fourth)}
        best_tokenizer = train_tokenizer(
            texts, languages, best_row["optimized"]["weights"]
        )
        verified = evaluate(best_tokenizer, texts, languages)
        if any(
            verified[key] != best_row["optimized"][key]
            for key in ("vocab_size", "rows", "spread", "adjusted_score")
        ):
            raise AssertionError("winner retraining did not reproduce optimized metrics")
        meta = {
            "variant": "faithful_shared_bpe_language_search_winner",
            "languages": [*FIXED, fourth],
            "fourth_language": {"code": fourth, "name": best_row["name"]},
            "weights": best_row["optimized"]["weights"],
            "metrics": best_row["optimized"],
            "candidate_count": len(final_rows),
            "optimization_scope": report["optimization_scope"],
            "vocab_size": best_tokenizer.get_vocab_size(),
        }
        best_tokenizer.save(str(output / "winner.tokenizer.json"))
        (output / "winner.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"WINNER {fourth} ({best_row['name']}): "
            f"score={best_row['optimized']['adjusted_score']:.2f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
