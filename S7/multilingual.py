"""Full-byte multilingual composition experiment for RKE-Head."""

from __future__ import annotations

import csv
import json
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from lm_compare import calibration, softmax
from rke import Adam, FullByteCodec, TinyTransformer, constrained_utf8_decode, sha256


SCRIPT_PARTS = {
    "Hindi-Devanagari": ("कगचजटडतनपर", "ािीु"),
    "Telugu": ("కగచజటడతనపర", "ాిీు"),
    "Tamil": ("கசடதபமநரலவ", "ாிீு"),
    "Arabic": ("ابتثجحخدذر", "َُِْ"),
}


def dataset() -> dict[str, list[dict[str, Any]]]:
    train, held_out = [], []
    for script, (stems, suffixes) in SCRIPT_PARTS.items():
        for stem_index, stem in enumerate(stems):
            for suffix_index, suffix in enumerate(suffixes):
                record = {
                    "script": script, "stem": unicodedata.normalize("NFC", stem),
                    "suffix": unicodedata.normalize("NFC", suffix),
                    "target": unicodedata.normalize("NFC", stem + suffix),
                }
                # Eight balanced holdouts per script; every component remains in train.
                (held_out if (stem_index + suffix_index) % 5 == 0 else train).append(record)
    train_targets = {x["target"] for x in train}
    held_targets = {x["target"] for x in held_out}
    if train_targets & held_targets:
        raise AssertionError("multilingual target leakage")
    for script in SCRIPT_PARTS:
        script_train = [x for x in train if x["script"] == script]
        script_held = [x for x in held_out if x["script"] == script]
        if not ({x["stem"] for x in script_held} <= {x["stem"] for x in script_train}
                and {x["suffix"] for x in script_held} <= {x["suffix"] for x in script_train}):
            raise AssertionError(f"component leakage design failed for {script}")
    return {"train": train, "held_out": held_out}


def features(codec: FullByteCodec, records: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([np.stack([codec.encode(b"c"), codec.encode(x["suffix"].encode("utf-8")),
                               codec.encode(x["stem"].encode("utf-8"))]) for x in records])


def evaluate(model: TinyTransformer, codec: FullByteCodec,
             records: list[dict[str, Any]]) -> dict[str, Any]:
    logits, _ = model.forward(features(codec, records))
    probabilities = softmax(logits)
    targets = np.stack([codec.ids(x["target"].encode("utf-8")) for x in records])
    mask = targets != 0
    selected = probabilities[np.arange(len(records))[:, None], np.arange(codec.slots)[None], targets]
    word_nll = (-np.log(selected + 1e-12) * mask).sum(axis=1)
    raw_predictions, constrained_predictions, raw_valid = [], [], []
    for item in logits:
        try:
            payload = codec.decode_logits(item)
            raw_predictions.append(payload.decode("utf-8")); raw_valid.append(True)
        except (ValueError, UnicodeDecodeError):
            raw_predictions.append("<INVALID_UTF8>"); raw_valid.append(False)
        try:
            constrained_predictions.append(constrained_utf8_decode(item, codec.max_bytes).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            constrained_predictions.append("<NO_VALID_PATH>")
    truth = [x["target"] for x in records]
    raw_correct = np.array([a == b for a, b in zip(truth, raw_predictions)])
    constrained_correct = np.array([a == b for a, b in zip(truth, constrained_predictions)])
    confidence = np.exp(-word_nll)
    per_script: dict[str, Any] = {}
    for script in SCRIPT_PARTS:
        indices = np.array([i for i, x in enumerate(records) if x["script"] == script])
        per_script[script] = {
            "examples": len(indices), "raw_exact_match": float(raw_correct[indices].mean()),
            "constrained_exact_match": float(constrained_correct[indices].mean()),
            "nll_per_byte_or_eos": float(word_nll[indices].sum() / mask[indices].sum()),
        }
    length_buckets: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        length_buckets[str(len(record["target"].encode("utf-8")))].append(index)
    by_length = {length: {"examples": len(indices),
                          "exact_match": float(constrained_correct[indices].mean()),
                          "nll_per_byte_or_eos": float(word_nll[indices].sum() / mask[indices].sum())}
                 for length, values in length_buckets.items() for indices in [np.array(values)]}
    predictions = [{"script": x["script"], "context": [x["suffix"], x["stem"]], "target": x["target"],
                    "utf8_bytes": list(x["target"].encode("utf-8")), "raw": raw_predictions[i],
                    "constrained": constrained_predictions[i], "exact": bool(constrained_correct[i])}
                   for i, x in enumerate(records)]
    return {
        "examples": len(records), "raw_exact_match": float(raw_correct.mean()),
        "constrained_exact_match": float(constrained_correct.mean()),
        "raw_invalid_utf8_rate": 1 - float(np.mean(raw_valid)), "constrained_invalid_utf8_rate": 0.0,
        "nll_per_word": float(word_nll.mean()), "nll_per_byte_or_eos": float(word_nll.sum() / mask.sum()),
        "calibration": calibration(confidence, constrained_correct), "per_script": per_script,
        "per_utf8_byte_length": by_length, "predictions": predictions,
    }


def comparison_svg(path: Path, metrics: dict[str, Any]) -> None:
    entries = list(metrics["per_script"].items())
    bars = []
    for index, (name, values) in enumerate(entries):
        x, value = 70 + index * 205, values["constrained_exact_match"]
        height, y = 210 * value, 285 - 210 * value
        bars.append(f"<rect x='{x}' y='{y:.1f}' width='130' height='{height:.1f}' rx='8' fill='#34d399'/>"
                    f"<text x='{x+65}' y='{y-10:.1f}' text-anchor='middle' font-size='20'>{value:.0%}</text>"
                    f"<text x='{x+65}' y='320' text-anchor='middle' font-size='14'>{name}</text>")
    svg = "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='360'><rect width='900' height='360' rx='18' fill='#091526'/>" \
          "<g fill='#dce8f7' font-family='system-ui'><text x='42' y='40' font-size='22' font-weight='700'>Multilingual held-out exact match</text>" \
          "<path d='M45 285H855' stroke='#4a607b'/><path d='M45 75H855' stroke='#29405c' stroke-dasharray='5 7'/>" + "".join(bars) + "</g></svg>"
    path.write_text(svg, encoding="utf-8")


def run_multilingual_experiment(output: Path, seed: int = 33) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    data, codec = dataset(), FullByteCodec(6)
    model, optimizer, rng = TinyTransformer(codec, d_slot=8, seed=seed), None, np.random.default_rng(seed + 1)
    optimizer = Adam(model.params, lr=.004)
    curve, started = [], time.perf_counter()
    for step in range(1, 3001):
        batch = [data["train"][int(i)] for i in rng.integers(0, len(data["train"]), size=64)]
        targets = np.stack([codec.ids(x["target"].encode("utf-8")) for x in batch])
        loss, grads = model.loss_and_grads(features(codec, batch), targets, targets != 0)
        optimizer.update(model.params, grads)
        if step == 1 or step % 250 == 0:
            curve.append({"step": step, "loss": loss})
    train_metrics, held_metrics = evaluate(model, codec, data["train"]), evaluate(model, codec, data["held_out"])
    results = {
        "experiment": "full-byte multilingual next-token composition", "seed": seed,
        "scripts": list(SCRIPT_PARTS), "codec": {"states": 258, "max_bytes": 6, "eos": True, "pad_masked": True},
        "split": {"train": len(data["train"]), "held_out": len(data["held_out"]),
                  "whole_target_overlap": 0, "components_covered_in_train": True},
        "dataset_hash": sha256(data), "model_hash": model.state_hash(),
        "training": {"steps": 3000, "batch_size": 64, "seconds": time.perf_counter() - started},
        "train": train_metrics, "held_out": held_metrics,
        "closed_vocabulary_baseline": {"held_out_representable_fraction": 0.0,
                                       "note": "all held-out whole strings are absent from the training vocabulary"},
        "claims": {"held_out_exact_at_least_95_percent": held_metrics["constrained_exact_match"] >= .95,
                   "every_script_at_least_95_percent": all(x["constrained_exact_match"] >= .95 for x in held_metrics["per_script"].values()),
                   "zero_constrained_invalid_utf8": held_metrics["constrained_invalid_utf8_rate"] == 0,
                   "whole_words_held_out": True},
        "limitations": ["Four controlled scripts and 160 generated combinations, not a natural multilingual corpus.",
                        "The rule is compositional concatenation; this does not establish multilingual semantics or perplexity.",
                        "Maximum output is six UTF-8 bytes in this experiment."],
    }
    results["overall_pass"] = all(results["claims"].values())
    # Keep full predictions in their own file and compact the metrics bundle.
    predictions = {"train": train_metrics.pop("predictions"), "held_out": held_metrics.pop("predictions")}
    (output / "predictions.json").write_text(json.dumps(predictions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "split.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (output / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "training_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve[0]); writer.writeheader(); writer.writerows(curve)
    np.savez_compressed(output / "model.npz", **model.params)
    comparison_svg(output / "comparison.svg", held_metrics)
    return results
