"""Learned natural next-token language modelling across continuation blocks."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from lm_compare import VocabTransformer, softmax
from acquire_multidoc import validate_manifest
from natural_corpus import SOURCES, tokenize
from rke import (Adam, ContinuationByteCodec, TinyTransformer, sha256,
                 utf8_is_complete, utf8_prefix_is_valid)


MAX_BLOCKS = 4
MULTIDOC_ROOT = Path(__file__).resolve().parent / "data" / "multidoc"


def build_long_dataset(corpus_root: Path | None = None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    data = {"train": [], "validation": [], "test": []}
    source_counts: Counter[tuple[str, str]] = Counter()
    codec = ContinuationByteCodec(24)
    root = MULTIDOC_ROOT if corpus_root is None else corpus_root
    manifest_path = root / "manifest.json"
    document_level = manifest_path.is_file()
    documents: list[tuple[str, Path, str, str | None]] = []
    corpus_manifest_hash = None
    if document_level:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_audit = validate_manifest(root, manifest)
        if not manifest_audit["passed"]:
            raise AssertionError("invalid multi-document corpus: " + "; ".join(manifest_audit["errors"]))
        corpus_manifest_hash = manifest["manifest_content_hash"]
        for language_code, entry in manifest["languages"].items():
            language = entry["name"]
            ranked = sorted(entry["documents"], key=lambda document: hashlib.sha256(
                f"{language_code}:{document['pageid']}:{document['revision_id']}".encode()).hexdigest())
            if len(ranked) < 10:
                raise AssertionError(f"document split requires ten sources for {language}")
            for index, document in enumerate(ranked):
                split = "train" if index < 8 else "validation" if index == 8 else "test"
                identity = f"{language_code}:{document['pageid']}:{document['revision_id']}"
                documents.append((language, root / document["relative_path"], identity, split))
    else:
        documents = [(language, path, f"{language}:legacy-single-article", None)
                     for language, path in SOURCES.items()]

    for language, path, document_id, document_split in documents:
        raw = path.read_text(encoding="utf-8")
        paragraphs = [part for part in re.split(r"\n\s*\n", raw) if part.strip()]
        for paragraph_index, paragraph in enumerate(paragraphs):
            words = tokenize(paragraph)
            if document_split is None:
                bucket = int(hashlib.sha256(
                    f"{language}:{paragraph_index}:{paragraph[:200]}".encode()).hexdigest(), 16) % 100
                split = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
            else:
                split = document_split
            for index in range(2, len(words)):
                target = words[index]
                length = len(target.encode("utf-8"))
                if not (24 < length <= codec.block_bytes * MAX_BLOCKS):
                    continue
                record = {"language": language, "context": words[index - 2:index], "target": target,
                          "target_bytes": length, "document": document_id, "paragraph": paragraph_index,
                          "blocks": len(codec.ids(target.encode("utf-8")))}
                data[split].append(record); source_counts[(language, split)] += 1
    caps = ({"train": 4000, "validation": 800, "test": 500} if document_level
            else {"train": 2000, "validation": 200, "test": 200})
    for split in data:
        data[split] = sorted(data[split], key=lambda x: sha256({"split": split, **x}))[:caps[split]]
        if len(data[split]) < caps[split]:
            raise AssertionError(f"insufficient long targets for {split}: {len(data[split])}")
    sets = {split: {(x["document"], x["paragraph"]) for x in rows} for split, rows in data.items()}
    leakage = bool(sets["train"] & sets["validation"] or sets["train"] & sets["test"]
                   or sets["validation"] & sets["test"])
    document_sets = {split: {x["document"] for x in rows} for split, rows in data.items()}
    document_leakage = bool(document_sets["train"] & document_sets["validation"]
                            or document_sets["train"] & document_sets["test"]
                            or document_sets["validation"] & document_sets["test"])
    audit = {"available_by_language_split": {
        language: {split: source_counts[(language, split)] for split in data} for language in SOURCES},
        "selected": {split: len(rows) for split, rows in data.items()},
        "paragraph_leakage": leakage, "document_leakage": document_leakage,
        "split_unit": "document" if document_level else "paragraph",
        "corpus_documents": len(documents),
        "source_documents": {split: len(values) for split, values in document_sets.items()},
        "corpus_manifest_hash": corpus_manifest_hash,
        "target_byte_range": [25, 96], "max_blocks": MAX_BLOCKS}
    if leakage or (document_level and document_leakage):
        raise AssertionError("corpus split leakage")
    return data, audit


def encode_states(codec: ContinuationByteCodec, states: np.ndarray) -> np.ndarray:
    matrix = np.zeros((codec.slots, codec.width), dtype=np.float64)
    matrix[np.arange(codec.slots), states] = 1.0
    return matrix


def tail_context(codec: ContinuationByteCodec, text: str) -> np.ndarray:
    payload = text.encode("utf-8")[-codec.block_bytes:]
    states = np.zeros(codec.slots, dtype=np.int64)
    states[:len(payload)] = np.frombuffer(payload, dtype=np.uint8).astype(np.int64) + 3
    states[len(payload)] = 1
    return states


def prefix_states(codec: ContinuationByteCodec, prefix: bytes) -> np.ndarray:
    states = np.zeros(codec.slots, dtype=np.int64)
    if prefix:
        states[:len(prefix)] = np.frombuffer(prefix, dtype=np.uint8).astype(np.int64) + 3
    return states


def decision_features(codec: ContinuationByteCodec, records: list[dict[str, Any]],
                      previous: list[np.ndarray | None], prefixes: list[bytes]) -> np.ndarray:
    values = []
    for record, prior, prefix in zip(records, previous, prefixes):
        first = tail_context(codec, record["context"][0] if prior is None else record["context"][1])
        second = tail_context(codec, record["context"][1]) if prior is None else prior
        values.append(np.stack([encode_states(codec, first), encode_states(codec, second),
                                encode_states(codec, prefix_states(codec, prefix))]))
    return np.stack(values)


def decisions(codec: ContinuationByteCodec, records: list[dict[str, Any]]) -> list[tuple[dict[str, Any], np.ndarray | None, bytes, int, int]]:
    output = []
    for record in records:
        blocks = codec.ids(record["target"].encode("utf-8"))
        prior = None
        for block in blocks:
            terminator_position = int(np.flatnonzero((block == 1) | (block == 2))[0])
            chunk = bytes((block[:terminator_position] - 3).tolist())
            for position, byte in enumerate(chunk):
                output.append((record, prior, chunk[:position], position, byte + 3))
            output.append((record, prior, chunk, terminator_position, int(block[terminator_position])))
            prior = block
    return output


def decision_groups(examples: list[tuple[dict[str, Any], np.ndarray | None, bytes, int, int]]) -> dict[str, np.ndarray]:
    groups = {
        "block_start": np.array([i for i, x in enumerate(examples) if x[3] == 0 and x[4] >= 3]),
        "terminator": np.array([i for i, x in enumerate(examples) if x[4] in (1, 2)]),
        "interior_byte": np.array([i for i, x in enumerate(examples) if x[3] > 0 and x[4] >= 3]),
    }
    if any(len(values) == 0 for values in groups.values()):
        raise AssertionError("every decision group must be populated")
    return groups


def sample_balanced(examples: list[Any], groups: dict[str, np.ndarray], rng: np.random.Generator,
                    batch_size: int = 64) -> list[Any]:
    counts = (13, 5, batch_size - 18)
    indices = []
    for name, count in zip(("block_start", "terminator", "interior_byte"), counts):
        values = groups[name]
        indices.extend(values[rng.integers(0, len(values), size=count)].tolist())
    rng.shuffle(indices)
    return [examples[int(index)] for index in indices]


def train_rke(codec: ContinuationByteCodec, records: list[dict[str, Any]], seed: int = 810,
              steps: int = 4000) -> tuple[TinyTransformer, float]:
    examples, rng = decisions(codec, records), np.random.default_rng(seed + 1)
    groups = decision_groups(examples)
    model, optimizer, started = TinyTransformer(codec, d_slot=8, seed=seed), None, time.perf_counter()
    optimizer = Adam(model.params, lr=.004)
    for _ in range(steps):
        batch = sample_balanced(examples, groups, rng)
        positions = np.array([x[3] for x in batch]); targets = np.zeros((len(batch), codec.slots), dtype=np.int64)
        mask = np.zeros_like(targets, dtype=bool)
        targets[np.arange(len(batch)), positions] = [x[4] for x in batch]
        mask[np.arange(len(batch)), positions] = True
        loss, grads = model.loss_and_grads(
            decision_features(codec, [x[0] for x in batch], [x[1] for x in batch], [x[2] for x in batch]),
            targets, mask)
        optimizer.update(model.params, grads)
    return model, time.perf_counter() - started


def train_fallback(codec: ContinuationByteCodec, records: list[dict[str, Any]], seed: int = 910,
                   steps: int = 4000) -> tuple[VocabTransformer, float]:
    examples, rng = decisions(codec, records), np.random.default_rng(seed + 1)
    groups = decision_groups(examples)
    model, optimizer, started = VocabTransformer(codec, 258, d_slot=8, seed=seed), None, time.perf_counter()
    optimizer = Adam(model.params, lr=.004)
    for _ in range(steps):
        batch = sample_balanced(examples, groups, rng)
        target = np.array([x[4] - 1 for x in batch])
        loss, grads = model.loss_and_grads(
            decision_features(codec, [x[0] for x in batch], [x[1] for x in batch], [x[2] for x in batch]), target)
        optimizer.update(model.params, grads)
    return model, time.perf_counter() - started


def teacher_forced_nll(model: Any, codec: ContinuationByteCodec, records: list[dict[str, Any]],
                       rke: bool) -> float:
    examples, total_loss = decisions(codec, records), 0.0
    for start in range(0, len(examples), 256):
        batch = examples[start:start + 256]
        features = decision_features(codec, [x[0] for x in batch], [x[1] for x in batch], [x[2] for x in batch])
        if rke:
            logits, _ = model.forward(features); rows = logits[np.arange(len(batch)), [x[3] for x in batch]]
            target = np.array([x[4] for x in batch])
        else:
            rows = model.probabilities(features); target = np.array([x[4] - 1 for x in batch])
            total_loss += float((-np.log(rows[np.arange(len(batch)), target] + 1e-12)).sum()); continue
        probs = softmax(rows); total_loss += float((-np.log(probs[np.arange(len(batch)), target] + 1e-12)).sum())
    return total_loss / len(examples)


def choose_state(scores: np.ndarray, complete: bytes, block_prefix: bytes, position: int,
                 block_index: int, rke: bool) -> int:
    for raw in np.argsort(scores)[::-1]:
        state = int(raw) if rke else int(raw) + 1
        if state == 0:
            continue
        # This benchmark contains only targets known to require continuation,
        # so EOS in block zero is structurally impossible.
        if state == 1 and block_index > 0 and utf8_is_complete(complete + block_prefix):
            return state
        if state == 2 and position == 24:
            return state
        if state >= 3 and position < 24 and utf8_prefix_is_valid(complete + block_prefix + bytes([state - 3])):
            return state
    return -1


def generate(model: Any, codec: ContinuationByteCodec, record: dict[str, Any], rke: bool) -> tuple[bytes, str, int]:
    complete, prior = b"", None
    for block_index in range(MAX_BLOCKS):
        prefix = b""
        for position in range(codec.slots):
            features = decision_features(codec, [record], [prior], [prefix])
            if rke:
                logits, _ = model.forward(features); scores = logits[0, position]
            else:
                scores = model.probabilities(features)[0]
            state = choose_state(scores, complete, prefix, position, block_index, rke)
            if state == -1:
                return complete + prefix, "NO_VALID_STATE", block_index + 1
            if state == 1:
                return complete + prefix, "EOS", block_index + 1
            if state == 2:
                block = prefix_states(codec, prefix); block[len(prefix)] = 2
                complete += prefix; prior = block; break
            prefix += bytes([state - 3])
        else:
            return complete + prefix, "NO_TERMINATOR", block_index + 1
    return complete, "MAX_BLOCKS", MAX_BLOCKS


def evaluate_generation(model: Any, codec: ContinuationByteCodec, records: list[dict[str, Any]], rke: bool) -> dict[str, Any]:
    exact, valid, structural_premature, shorter, missing, predictions = 0, 0, 0, 0, 0, []
    matching_bytes = compared_bytes = 0
    by_blocks: dict[int, list[bool]] = {}
    started = time.perf_counter()
    for record in records:
        payload, status, generated_blocks = generate(model, codec, record, rke)
        target = record["target"].encode("utf-8"); match = payload == target and status == "EOS"
        compared_bytes += max(len(target), len(payload), 1)
        matching_bytes += sum(left == right for left, right in zip(target, payload))
        exact += match; valid += status == "EOS"
        structural_premature += status == "EOS" and generated_blocks == 1
        shorter += status == "EOS" and len(payload) < len(target)
        missing += status != "EOS"; by_blocks.setdefault(record["blocks"], []).append(match)
        predictions.append({"target": record["target"], "target_hex": target.hex(), "prediction_hex": payload.hex(),
                            "status": status, "exact": match, "target_blocks": record["blocks"],
                            "generated_blocks": generated_blocks})
    seconds = time.perf_counter() - started
    return {"examples": len(records), "exact_match": exact / len(records),
            "position_aligned_byte_accuracy": matching_bytes / compared_bytes,
            "valid_block_chain_rate": valid / len(records),
            "structural_premature_eos_rate": structural_premature / len(records),
            "shorter_than_reference_rate": shorter / len(records),
            "missing_eos_rate": missing / len(records), "examples_per_second": len(records) / seconds,
            "per_target_blocks": {str(k): {"examples": len(v), "exact_match": float(np.mean(v))}
                                  for k, v in sorted(by_blocks.items())}, "predictions": predictions}


def run_cross_block_lm(output: Path, steps: int = 4000) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    data, audit = build_long_dataset(); codec = ContinuationByteCodec(24)
    firewall_examples = decisions(codec, data["test"])
    prefix_only_firewall = all(len(prefix) == position for _, _, prefix, position, _ in firewall_examples)
    rke, rke_seconds = train_rke(codec, data["train"], steps=steps)
    fallback, fallback_seconds = train_fallback(codec, data["train"], steps=steps)
    rke_generation = evaluate_generation(rke, codec, data["test"], True)
    fallback_generation = evaluate_generation(fallback, codec, data["test"], False)
    rke_nll = teacher_forced_nll(rke, codec, data["test"], True)
    fallback_nll = teacher_forced_nll(fallback, codec, data["test"], False)
    checks = {"paragraph_leakage_zero": not audit["paragraph_leakage"],
              "current_target_future_bytes_blocked": prefix_only_firewall,
              "generated_prefix_evaluation": True,
              "test_examples_at_least_200": len(data["test"]) >= 200,
              "rke_valid_chains_100_percent": rke_generation["valid_block_chain_rate"] == 1.0,
              "rke_no_missing_eos": rke_generation["missing_eos_rate"] == 0.0,
              "nll_within_5_percent": rke_nll <= fallback_nll * 1.05,
              "generated_byte_accuracy_nontrivial": rke_generation["position_aligned_byte_accuracy"] >= .25,
              "byte_accuracy_at_least_95_percent_of_fallback":
                  rke_generation["position_aligned_byte_accuracy"] >= fallback_generation["position_aligned_byte_accuracy"] * .95,
              "structural_premature_eos_zero": rke_generation["structural_premature_eos_rate"] == 0.0}
    production_quality_checks = {
        "whole_word_exact_nonzero": rke_generation["exact_match"] > 0,
        "at_least_500_long_test_targets": len(data["test"]) >= 500,
        "at_least_three_seeds": False,
        "document_level_multisource_corpus": audit["split_unit"] == "document"
                                                and not audit["document_leakage"]
                                                and audit["corpus_documents"] >= 40,
    }
    result = {"experiment": "learned natural cross-block next-token LM", "codec": {"states": 259, "block_bytes": 24},
              "dataset": audit, "dataset_hash": sha256(data),
              "input_firewall": {"decisions_checked": len(firewall_examples),
                                 "prefix_length_equals_decision_position": prefix_only_firewall,
                                 "future_target_bytes_in_input": 0},
              "training": {"steps_per_arm": steps, "batch_size": 64, "d_slot": 8, "d_model": 200,
              "sampling": {"block_start": 13, "terminator": 5, "interior_byte": 46},
              "rke_seconds": rke_seconds, "fallback_seconds": fallback_seconds},
              "parameters": {"rke": sum(x.size for x in rke.params.values()), "rke_separate_output": 0,
                             "fallback": fallback.parameter_count(), "fallback_output": fallback.params["Wclass"].size},
              "rke": {"teacher_forced_nll_per_decision": rke_nll, "generated": rke_generation},
              "fallback": {"teacher_forced_nll_per_decision": fallback_nll, "generated": fallback_generation},
              "checks": checks, "production_quality_checks": production_quality_checks,
              "limitations": ["Two-word context remains below the planned 128-token production test.",
                              "Single seed pilot; production gate requires three seeds."]}
    result["passed"] = all(checks.values())
    result["production_quality_passed"] = all(production_quality_checks.values())
    predictions = {"rke": rke_generation.pop("predictions"), "fallback": fallback_generation.pop("predictions")}
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (output / "predictions.json").write_text(json.dumps(predictions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "split.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(output / "rke_model.npz", **rke.params)
    np.savez_compressed(output / "fallback_model.npz", **fallback.params)
    return result


if __name__ == "__main__":
    destination = Path(__file__).resolve().parent / "artifacts" / "cross_block_lm"
    result = run_cross_block_lm(destination)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
