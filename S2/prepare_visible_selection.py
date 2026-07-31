#!/usr/bin/env python3
"""Package the 5,000-visible-word winners without a directionality filter."""
from __future__ import annotations

import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TOKENIZERS = ROOT / "tokenizers"
PIECE_ROOT = TOKENIZERS / "piecevocab"
BPE_ROOT = TOKENIZERS / "bpe"
CORPUS = PIECE_ROOT / "candidate-corpus"
MIN_VISIBLE_WORD_RUNS = 5_000
URL_RE = re.compile(r"https?://[^\s)]+")


def word_runs(text: str) -> int:
    count = 0
    inside = False
    for char in text:
        is_word = unicodedata.category(char)[0] in {"L", "M", "N"}
        if is_word and not inside:
            count += 1
        inside = is_word
    return count


def visible_word_runs(code: str) -> int:
    text = (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
    return word_runs(URL_RE.sub("", text))


def eligible(rows: list[dict], bpe: bool) -> list[dict]:
    selected = [
        row for row in rows
        if visible_word_runs(row["code"]) >= MIN_VISIBLE_WORD_RUNS
    ]
    selected.sort(
        key=lambda row: (
            row["optimized"]["adjusted_score"] if bpe
            else row["adjusted_score"]
        ),
        reverse=True,
    )
    return selected


def package_piece() -> None:
    sys.path.insert(0, str(PIECE_ROOT))
    from rank_fourth_languages import prepare, train  # noqa: PLC0415

    source = json.loads(
        (PIECE_ROOT / "search-results" / "ranking.json").read_text(encoding="utf-8")
    )
    rows = eligible(source["ranking"], bpe=False)
    winner = rows[0]
    languages = ("en", "hi", "te", winner["code"])
    texts = {
        code: (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
        for code in languages
    }
    tokenizer, metrics = train(
        prepare(texts, languages), winner["weights"], verify_roundtrip=True
    )
    for code in languages:
        if metrics["rows"][code] != winner["rows"][code]:
            raise AssertionError(f"PieceVocab {code}: rebuilt metrics differ")

    output = PIECE_ROOT / "search-visible-5000"
    output.mkdir(exist_ok=True)
    report = {
        "policy": {
            "min_visible_word_runs": MIN_VISIBLE_WORD_RUNS,
            "writing_direction_filter": None,
        },
        "candidate_count": len(rows),
        "ranking": rows,
    }
    (output / "ranking.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    bundle = tokenizer.to_dict({
        "description": "5,000-visible-word PieceVocab winner; all writing directions included",
        "languages": list(languages),
        "vocab_size": len(tokenizer.vocab),
        "metrics": winner,
    })
    (output / "winner.tokenizer.json").write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"PieceVocab: {winner['name']} ({winner['code']}), "
        f"{len(rows)} eligible, score={winner['adjusted_score']:.2f}"
    )


def package_bpe() -> None:
    source_dir = BPE_ROOT / "search-results"
    source = json.loads((source_dir / "ranking.json").read_text(encoding="utf-8"))
    rows = eligible(source["ranking"], bpe=True)
    winner = rows[0]
    source_meta = json.loads(
        (source_dir / "winner.meta.json").read_text(encoding="utf-8")
    )
    if source_meta["fourth_language"]["code"] != winner["code"]:
        raise AssertionError("BPE global artifact is not the visible-word winner")

    output = BPE_ROOT / "search-visible-5000"
    output.mkdir(exist_ok=True)
    report = {
        "policy": {
            "min_visible_word_runs": MIN_VISIBLE_WORD_RUNS,
            "writing_direction_filter": None,
        },
        "candidate_count": len(rows),
        "ranking": rows,
    }
    (output / "ranking.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(source_dir / "winner.tokenizer.json", output / "winner.tokenizer.json")
    shutil.copy2(source_dir / "winner.meta.json", output / "winner.meta.json")
    print(
        f"BPE: {winner['name']} ({winner['code']}), "
        f"{len(rows)} eligible, score={winner['optimized']['adjusted_score']:.2f}"
    )


def main() -> int:
    package_piece()
    package_bpe()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
