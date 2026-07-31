#!/usr/bin/env python3
# Copyright 2026 Avnish Midha. All rights reserved.
# Author: Avnish Midha
# GitHub: avnishbm
# Purpose: Build the Netlify-ready interactive tokenizer review site.
"""Build the dependency-free Netlify review site."""
from __future__ import annotations

import json
import re
import shutil
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "site"
DIST = ROOT / "dist"
TOKENIZERS = ROOT / "tokenizers"
PIECE_ROOT = TOKENIZERS / "piecevocab"
BPE_ROOT = TOKENIZERS / "bpe"

ARTIFACTS = {
    PIECE_ROOT / "search-visible-5000" / "winner.tokenizer.json": "piecevocab.tokenizer.json",
    PIECE_ROOT / "piecevocab.py": "piecevocab.py",
    ROOT / "artifacts" / "piecevocab.SUBMISSION.md": "piecevocab.SUBMISSION.md",
    ROOT / "artifacts" / "piecevocab.summary.md": "piecevocab.summary.md",
    BPE_ROOT / "search-visible-5000" / "winner.tokenizer.json": "bpe.tokenizer.json",
    BPE_ROOT / "bpe_common.py": "bpe_common.py",
    ROOT / "artifacts" / "bpe.summary.md": "bpe.summary.md",
}

RANKINGS = {
    "piece": PIECE_ROOT / "search-results" / "ranking.json",
    "bpe": BPE_ROOT / "search-results" / "ranking.json",
}
CANDIDATE_CORPUS = PIECE_ROOT / "candidate-corpus"
URL_RE = re.compile(r"https?://[^\s)]+")
MIN_VISIBLE_WORD_RUNS = 5_000


def browser_ranking(path: Path, bpe: bool) -> list[dict]:
    ranking = json.loads(path.read_text(encoding="utf-8"))["ranking"]
    rows = []
    for rank, candidate in enumerate(ranking, 1):
        metrics = candidate["optimized"] if bpe else candidate
        code = candidate["code"]
        ratios = [row["ratio"] for row in metrics["rows"].values()]
        metadata = json.loads(
            (CANDIDATE_CORPUS / f"{code}.meta.json").read_text(encoding="utf-8")
        )
        text = (CANDIDATE_CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
        raw_word_runs = 0
        inside_word = False
        for char in text:
            is_word = unicodedata.category(char)[0] in {"L", "M", "N"}
            if is_word and not inside_word:
                raw_word_runs += 1
            inside_word = is_word
        visible_word_runs = 0
        inside_word = False
        for char in URL_RE.sub("", text):
            is_word = unicodedata.category(char)[0] in {"L", "M", "N"}
            if is_word and not inside_word:
                visible_word_runs += 1
            inside_word = is_word
        row = {
            "rank": rank,
            "code": code,
            "name": candidate["name"],
            "score": metrics["adjusted_score"],
            "spread": metrics["spread"],
            "fourth_fertility": metrics["rows"][code]["ratio"],
            "highest_fertility": max(ratios),
            "raw_word_runs": raw_word_runs,
            "visible_word_runs": visible_word_runs,
            "tokens": metrics["rows"][code]["tokens"],
            "faithful_units": metrics["rows"][code]["faithful_units"],
            "article_url": metadata["article_url"],
        }
        rows.append(row)
    return rows


def main() -> int:
    if DIST.exists():
        shutil.rmtree(DIST)
    shutil.copytree(SOURCE, DIST)
    downloads = DIST / "downloads"
    downloads.mkdir()
    for source, name in ARTIFACTS.items():
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, downloads / name)
    ranking_data = {
        "policy": {
            "source_candidates": 306,
            "min_visible_word_runs": MIN_VISIBLE_WORD_RUNS,
            "eligible_candidates": 86,
            "writing_direction_filter": None,
            "right_to_left_candidates_included": 19,
            "url_destinations_removed_before_counting": True,
            "eligibility_measure": "pre-tokenizer visible Unicode word runs",
        },
        **{
            name: browser_ranking(path, bpe=name == "bpe")
            for name, path in RANKINGS.items()
        },
    }
    for name in RANKINGS:
        if len(ranking_data[name]) != ranking_data["policy"]["source_candidates"]:
            raise AssertionError(
                f"{name}: expected 306 candidates, "
                f"found {len(ranking_data[name])}"
            )
    (DIST / "language-rankings.json").write_text(
        json.dumps(ranking_data, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Built {DIST} with {len(ARTIFACTS)} review artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
