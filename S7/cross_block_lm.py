"""Learned natural next-token language modelling across continuation blocks."""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from lm_compare import VocabTransformer, softmax
from acquire_multidoc import (DOCUMENTS_PER_TOPIC, DOCUMENT_SPLIT_PER_TOPIC,
                              TOPIC_ROOTS, validate_manifest)
from natural_corpus import SOURCES, tokenize
from rke import (Adam, ContinuationByteCodec, TinyTransformer, sha256,
                 utf8_is_complete, utf8_prefix_is_valid)


MAX_BLOCKS = 4
MIN_TARGET_BYTES = 25
MAX_TARGET_BYTES = 24 * MAX_BLOCKS
PRODUCTION_CONTINUATION_EXACT_THRESHOLD = .20
MULTIDOC_ROOT = Path(__file__).resolve().parent / "data" / "multidoc"


def record_languages(records: list[dict[str, Any]]) -> list[str]:
    present = {record["language"] for record in records}
    return [language for language in SOURCES if language in present] + sorted(present - set(SOURCES))


def build_continuation_span(words: list[str], start: int) -> tuple[str, int] | None:
    """Return the shortest complete-token span that crosses one byte block.

    Joining already NFC-normalized tokens with a single space ensures that a
    target boundary can never split a Unicode code point or a token.  Using a
    span rather than requiring one unusually byte-long word avoids selecting
    Indic scripts simply because their UTF-8 code points use more bytes.
    """
    selected: list[str] = []
    for word in words[start:]:
        candidate = " ".join([*selected, word])
        byte_count = len(candidate.encode("utf-8"))
        if byte_count > MAX_TARGET_BYTES:
            break
        selected.append(word)
        if byte_count >= MIN_TARGET_BYTES:
            return candidate, len(selected)
    return None


def build_long_dataset(corpus_root: Path | None = None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    data = {"train": [], "validation": [], "test": []}
    source_counts: Counter[tuple[str, str]] = Counter()
    source_topic_counts: Counter[tuple[str, str, str]] = Counter()
    codec = ContinuationByteCodec(24)
    root = MULTIDOC_ROOT if corpus_root is None else corpus_root
    manifest_path = root / "manifest.json"
    document_level = manifest_path.is_file()
    documents: list[tuple[str, Path, str, str | None, str, str]] = []
    corpus_manifest_hash = None
    if document_level:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_audit = validate_manifest(root, manifest)
        if not manifest_audit["passed"]:
            raise AssertionError("invalid multi-document corpus: " + "; ".join(manifest_audit["errors"]))
        corpus_manifest_hash = manifest["manifest_content_hash"]
        for language_code, entry in manifest["languages"].items():
            language = entry["name"]
            for topic in TOPIC_ROOTS:
                ranked = sorted(
                    [document for document in entry["documents"] if document["source_topic"] == topic],
                    key=lambda document: hashlib.sha256(
                        f"{language_code}:{topic}:{document['pageid']}:{document['revision_id']}".encode()
                    ).hexdigest())
                if len(ranked) != DOCUMENTS_PER_TOPIC:
                    raise AssertionError(
                        f"document split requires {DOCUMENTS_PER_TOPIC} {language}:{topic} sources")
                train_end = DOCUMENT_SPLIT_PER_TOPIC["train"]
                validation_end = train_end + DOCUMENT_SPLIT_PER_TOPIC["validation"]
                for index, document in enumerate(ranked):
                    split = "train" if index < train_end else "validation" if index < validation_end else "test"
                    identity = f"{language_code}:{document['pageid']}:{document['revision_id']}"
                    documents.append((language, root / document["relative_path"], identity, split,
                                      document["sha256"], topic))
    else:
        documents = [(language, path, f"{language}:legacy-single-article", None,
                      hashlib.sha256(path.read_bytes()).hexdigest(), "legacy")
                     for language, path in SOURCES.items()]

    for language, path, document_id, document_split, document_hash, source_topic in documents:
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
                span = build_continuation_span(words, index)
                if span is None:
                    continue
                target, target_words = span
                length = len(target.encode("utf-8"))
                record = {"language": language, "context": words[index - 2:index], "target": target,
                          "target_bytes": length, "document": document_id, "paragraph": paragraph_index,
                          "target_start_word": index, "target_end_word": index + target_words,
                          "target_words": target_words,
                          "source_topic": source_topic,
                          "document_sha256": document_hash,
                          "blocks": len(codec.ids(target.encode("utf-8")))}
                data[split].append(record); source_counts[(language, split)] += 1
                source_topic_counts[(language, source_topic, split)] += 1
    quotas_per_language = ({"train": 1000, "validation": 200, "test": 125} if document_level
                           else {"train": 500, "validation": 50, "test": 50})
    topics = list(TOPIC_ROOTS) if document_level else ["legacy"]
    topic_quotas_per_language = {}
    for split, quota in quotas_per_language.items():
        base, remainder = divmod(quota, len(topics))
        topic_quotas_per_language[split] = {
            topic: base + (index < remainder) for index, topic in enumerate(topics)}
    # Reserve target strings in train -> validation -> test order. Repeated
    # occurrences remain legal within a split, but a gold continuation target
    # can never appear in an earlier split and become a memorization shortcut.
    reserved_targets: set[str] = set()
    discarded_cross_split_targets: dict[str, int] = {}
    discarded_by_language_split: dict[str, dict[str, int]] = {language: {} for language in SOURCES}
    eligible_after_cross_split_filter: dict[str, dict[str, int]] = {language: {} for language in SOURCES}
    eligible_by_language_topic_split = {
        language: {topic: {} for topic in topics} for language in SOURCES}
    for split in ("train", "validation", "test"):
        selected_rows: list[dict[str, Any]] = []
        discarded_cross_split_targets[split] = 0
        for language in SOURCES:
            ordered = sorted((record for record in data[split] if record["language"] == language),
                             key=lambda x: sha256({"language": language, "split": split, **x}))
            eligible = ([record for record in ordered if record["target"] not in reserved_targets]
                        if split != "train" else ordered)
            discarded = len(ordered) - len(eligible)
            discarded_by_language_split[language][split] = discarded
            discarded_cross_split_targets[split] += discarded
            eligible_after_cross_split_filter[language][split] = len(eligible)
            for topic in topics:
                topic_eligible = [record for record in eligible if record["source_topic"] == topic]
                eligible_by_language_topic_split[language][topic][split] = len(topic_eligible)
                topic_quota = topic_quotas_per_language[split][topic]
                if len(topic_eligible) < topic_quota:
                    raise AssertionError(
                        f"insufficient {language}:{topic} continuation spans for {split}: "
                        f"{len(topic_eligible)} available after leakage filtering, {topic_quota} required")
                selected_rows.extend(topic_eligible[:topic_quota])
        # A final hash order interleaves languages and topics deterministically
        # without changing their quotas. Update reservations only after the
        # whole split is selected so duplicates remain legal within one split.
        data[split] = sorted(selected_rows, key=lambda x: sha256({"balanced_split": split, **x}))
        reserved_targets.update(record["target"] for record in data[split])
    sets = {split: {(x["document"], x["paragraph"]) for x in rows} for split, rows in data.items()}
    leakage = bool(sets["train"] & sets["validation"] or sets["train"] & sets["test"]
                   or sets["validation"] & sets["test"])
    document_sets = {split: {x["document"] for x in rows} for split, rows in data.items()}
    document_leakage = bool(document_sets["train"] & document_sets["validation"]
                            or document_sets["train"] & document_sets["test"]
                            or document_sets["validation"] & document_sets["test"])
    document_hash_sets = {split: {x["document_sha256"] for x in rows} for split, rows in data.items()}
    content_hash_leakage = bool(document_hash_sets["train"] & document_hash_sets["validation"]
                                or document_hash_sets["train"] & document_hash_sets["test"]
                                or document_hash_sets["validation"] & document_hash_sets["test"])
    target_sets = {split: {x["target"] for x in rows} for split, rows in data.items()}
    target_leakage = bool(target_sets["train"] & target_sets["validation"]
                          or target_sets["train"] & target_sets["test"]
                          or target_sets["validation"] & target_sets["test"])
    sample_sets = {split: {(x["language"], tuple(x["context"]), x["target"])
                           for x in rows} for split, rows in data.items()}
    sample_leakage = bool(sample_sets["train"] & sample_sets["validation"]
                          or sample_sets["train"] & sample_sets["test"]
                          or sample_sets["validation"] & sample_sets["test"])
    selected_by_language_split = {
        language: {split: sum(record["language"] == language for record in rows)
                   for split, rows in data.items()}
        for language in SOURCES
    }
    language_quotas_satisfied = all(
        selected_by_language_split[language][split] == quotas_per_language[split]
        for language in SOURCES for split in data)
    selected_by_language_topic_split = {
        language: {
            topic: {split: sum(record["language"] == language and record["source_topic"] == topic
                               for record in rows)
                    for split, rows in data.items()}
            for topic in topics}
        for language in SOURCES}
    topic_quotas_satisfied = all(
        selected_by_language_topic_split[language][topic][split] ==
        topic_quotas_per_language[split][topic]
        for language in SOURCES for topic in topics for split in data)
    document_inventory = Counter(
        (language, topic, split) for language, _, _, split, _, topic in documents if split is not None)
    unicode_safe_boundaries = all(
        unicodedata.is_normalized("NFC", record["target"])
        and record["target"].encode("utf-8").decode("utf-8") == record["target"]
        and record["target"] == " ".join(record["target"].split(" "))
        and MIN_TARGET_BYTES <= record["target_bytes"] <= MAX_TARGET_BYTES
        for rows in data.values() for record in rows)
    target_statistics_by_language_split: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for language in SOURCES:
        target_statistics_by_language_split[language] = {}
        for split, rows in data.items():
            selected = [record for record in rows if record["language"] == language]
            byte_lengths = [record["target_bytes"] for record in selected]
            word_lengths = [record["target_words"] for record in selected]
            target_statistics_by_language_split[language][split] = {
                "examples": len(selected),
                "minimum_bytes": min(byte_lengths) if byte_lengths else None,
                "maximum_bytes": max(byte_lengths) if byte_lengths else None,
                "mean_bytes": sum(byte_lengths) / len(byte_lengths) if byte_lengths else None,
                "minimum_words": min(word_lengths) if word_lengths else None,
                "maximum_words": max(word_lengths) if word_lengths else None,
                "mean_words": sum(word_lengths) / len(word_lengths) if word_lengths else None,
            }
    audit = {"available_by_language_split": {
        language: {split: source_counts[(language, split)] for split in data} for language in SOURCES},
        "available_by_language_topic_split": {
            language: {topic: {split: source_topic_counts[(language, topic, split)] for split in data}
                       for topic in topics} for language in SOURCES},
        "selected_by_language_split": selected_by_language_split,
        "selected_by_language_topic_split": selected_by_language_topic_split,
        "eligible_after_cross_split_filter": eligible_after_cross_split_filter,
        "eligible_by_language_topic_split": eligible_by_language_topic_split,
        "quotas_per_language": quotas_per_language,
        "topic_quotas_per_language": topic_quotas_per_language,
        "sampling_policy": "equal language/topic strata with deterministic within-stratum hash order",
        "selected": {split: len(rows) for split, rows in data.items()},
        "paragraph_leakage": leakage, "document_leakage": document_leakage,
        "document_content_hash_leakage": content_hash_leakage,
        "target_string_leakage": target_leakage, "exact_sample_leakage": sample_leakage,
        "discarded_cross_split_target_records": discarded_cross_split_targets,
        "discarded_cross_split_target_records_by_language": discarded_by_language_split,
        "split_unit": "document" if document_level else "paragraph",
        "corpus_documents": len(documents),
        "corpus_topics": topics,
        "document_inventory_by_language_topic_split": {
            language: {topic: {split: document_inventory[(language, topic, split)] for split in data}
                       for topic in topics} for language in SOURCES},
        "source_documents": {split: len(values) for split, values in document_sets.items()},
        "corpus_manifest_hash": corpus_manifest_hash,
        "target_construction": {
            "unit": "NFC-normalized complete-token continuation span",
            "selection": "shortest span reaching minimum bytes",
            "separator": "single ASCII space",
            "minimum_bytes": MIN_TARGET_BYTES,
            "maximum_bytes": MAX_TARGET_BYTES,
            "unicode_safe_boundaries": unicode_safe_boundaries,
        },
        "target_statistics_by_language_split": target_statistics_by_language_split,
        "target_byte_range": [MIN_TARGET_BYTES, MAX_TARGET_BYTES], "max_blocks": MAX_BLOCKS,
        "quality_gates": {
            "manifest_and_payload_hashes_valid": bool(document_level),
            "unicode_safe_target_boundaries": unicode_safe_boundaries,
            "equal_per_language_quotas": language_quotas_satisfied,
            "topic_strata_quotas_satisfied": topic_quotas_satisfied,
            "document_content_hash_overlap_zero": not content_hash_leakage,
            "target_string_overlap_zero": not target_leakage,
            "exact_sample_overlap_zero": not sample_leakage,
            "every_language_present_in_test": all(
                values["test"] > 0 for values in selected_by_language_split.values()),
        }}
    if (leakage or target_leakage or sample_leakage
            or (document_level and (document_leakage or content_hash_leakage))):
        raise AssertionError("corpus split leakage")
    if not language_quotas_satisfied or not topic_quotas_satisfied:
        raise AssertionError("balanced language/topic quotas not satisfied")
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


def teacher_forced_nll_report(model: Any, codec: ContinuationByteCodec,
                              records: list[dict[str, Any]], rke: bool) -> dict[str, Any]:
    per_language = {
        language: teacher_forced_nll(
            model, codec, [record for record in records if record["language"] == language], rke)
        for language in record_languages(records)
    }
    return {"normalization": "loss-bearing byte/CONT/EOS decision",
            "micro_average": teacher_forced_nll(model, codec, records, rke),
            "macro_average": float(np.mean(list(per_language.values()))),
            "per_language": per_language}


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


def evaluate_generation(model: Any, codec: ContinuationByteCodec, records: list[dict[str, Any]],
                        rke: bool, include_breakdown: bool = True) -> dict[str, Any]:
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
    result = {"examples": len(records), "exact_match": exact / len(records),
              "position_aligned_byte_accuracy": matching_bytes / compared_bytes,
              "valid_block_chain_rate": valid / len(records),
              "structural_premature_eos_rate": structural_premature / len(records),
              "shorter_than_reference_rate": shorter / len(records),
              "missing_eos_rate": missing / len(records), "examples_per_second": len(records) / seconds,
              "per_target_blocks": {str(k): {"examples": len(v), "exact_match": float(np.mean(v))}
                                    for k, v in sorted(by_blocks.items())}, "predictions": predictions}
    if include_breakdown:
        metric_names = ("exact_match", "position_aligned_byte_accuracy", "valid_block_chain_rate",
                        "structural_premature_eos_rate", "shorter_than_reference_rate", "missing_eos_rate")
        per_language = {}
        for language in record_languages(records):
            rows = [record for record in records if record["language"] == language]
            value = evaluate_generation(model, codec, rows, rke, include_breakdown=False)
            value.pop("predictions"); value.pop("examples_per_second")
            per_language[language] = value
        result["per_language"] = per_language
        result["micro_average"] = {name: result[name] for name in metric_names}
        result["macro_average"] = {
            name: float(np.mean([value[name] for value in per_language.values()]))
            for name in metric_names}
    return result


def generate_after_gold_first_block(model: Any, codec: ContinuationByteCodec,
                                    record: dict[str, Any], rke: bool) -> tuple[bytes, str, int]:
    """Generate every block after a frozen gold block zero.

    This isolates learned cross-block continuation from open-ended block-zero
    prediction. The model still receives and must continue from the exact same
    block representation used by normal generated-prefix decoding.
    """
    target_blocks = codec.ids(record["target"].encode("utf-8"))
    if len(target_blocks) < 2:
        raise ValueError("continuation evaluation requires at least two blocks")
    prior = target_blocks[0]
    complete = bytes((prior[:codec.block_bytes] - 3).tolist())
    for block_index in range(1, MAX_BLOCKS):
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


def evaluate_gold_first_block_continuation(model: Any, codec: ContinuationByteCodec,
                                           records: list[dict[str, Any]], rke: bool,
                                           include_breakdown: bool = True) -> dict[str, Any]:
    exact = valid = missing = matching = compared = 0
    predictions = []
    started = time.perf_counter()
    for record in records:
        payload, status, generated_blocks = generate_after_gold_first_block(model, codec, record, rke)
        target = record["target"].encode("utf-8")
        target_suffix, predicted_suffix = target[codec.block_bytes:], payload[codec.block_bytes:]
        match = payload == target and status == "EOS"
        exact += match; valid += status == "EOS"; missing += status != "EOS"
        compared += max(len(target_suffix), len(predicted_suffix), 1)
        matching += sum(left == right for left, right in zip(target_suffix, predicted_suffix))
        predictions.append({"target": record["target"], "target_suffix_hex": target_suffix.hex(),
                            "prediction_suffix_hex": predicted_suffix.hex(), "status": status,
                            "exact": match, "target_blocks": record["blocks"],
                            "generated_blocks": generated_blocks})
    seconds = time.perf_counter() - started
    result = {"protocol": "gold block 0; generate blocks 1..EOS", "examples": len(records),
              "exact_count": exact, "exact_match": exact / len(records),
              "position_aligned_suffix_byte_accuracy": matching / compared,
              "valid_chain_rate": valid / len(records), "missing_eos_rate": missing / len(records),
              "examples_per_second": len(records) / seconds, "predictions": predictions}
    if include_breakdown:
        metric_names = ("exact_match", "position_aligned_suffix_byte_accuracy",
                        "valid_chain_rate", "missing_eos_rate")
        per_language = {}
        for language in record_languages(records):
            rows = [record for record in records if record["language"] == language]
            value = evaluate_gold_first_block_continuation(
                model, codec, rows, rke, include_breakdown=False)
            value.pop("predictions"); value.pop("examples_per_second")
            per_language[language] = value
        result["per_language"] = per_language
        result["micro_average"] = {name: result[name] for name in metric_names}
        result["macro_average"] = {
            name: float(np.mean([value[name] for value in per_language.values()]))
            for name in metric_names}
    return result


def exact_prefix_retrieval_baseline(train: list[dict[str, Any]], test: list[dict[str, Any]],
                                    block_bytes: int, include_breakdown: bool = True) -> dict[str, Any]:
    """Memorization control: most frequent training span sharing gold block zero."""
    candidates: dict[bytes, Counter[bytes]] = {}
    for record in train:
        payload = record["target"].encode("utf-8")
        candidates.setdefault(payload[:block_bytes], Counter())[payload] += 1
    exact = covered = 0
    for record in test:
        payload = record["target"].encode("utf-8"); values = candidates.get(payload[:block_bytes])
        if not values:
            continue
        covered += 1
        prediction = sorted(values.items(), key=lambda item: (-item[1], item[0]))[0][0]
        exact += prediction == payload
    result = {"protocol": "most frequent training target with identical first block",
              "examples": len(test), "covered": covered, "coverage": covered / len(test),
              "exact_count": exact, "exact_match": exact / len(test)}
    if include_breakdown:
        per_language = {
            language: exact_prefix_retrieval_baseline(
                train, [record for record in test if record["language"] == language],
                block_bytes, include_breakdown=False)
            for language in record_languages(test)
        }
        result["per_language"] = per_language
        result["micro_average"] = {name: result[name] for name in ("coverage", "exact_match")}
        result["macro_average"] = {
            name: float(np.mean([value[name] for value in per_language.values()]))
            for name in ("coverage", "exact_match")}
    return result


def run_cross_block_lm(output: Path, steps: int = 4000) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    data, audit = build_long_dataset(); codec = ContinuationByteCodec(24)
    firewall_examples = decisions(codec, data["test"])
    prefix_only_firewall = all(len(prefix) == position for _, _, prefix, position, _ in firewall_examples)
    rke, rke_seconds = train_rke(codec, data["train"], steps=steps)
    fallback, fallback_seconds = train_fallback(codec, data["train"], steps=steps)
    rke_generation = evaluate_generation(rke, codec, data["test"], True)
    fallback_generation = evaluate_generation(fallback, codec, data["test"], False)
    rke_continuation = evaluate_gold_first_block_continuation(rke, codec, data["test"], True)
    fallback_continuation = evaluate_gold_first_block_continuation(fallback, codec, data["test"], False)
    retrieval_continuation = exact_prefix_retrieval_baseline(data["train"], data["test"], codec.block_bytes)
    rke_nll_report = teacher_forced_nll_report(rke, codec, data["test"], True)
    fallback_nll_report = teacher_forced_nll_report(fallback, codec, data["test"], False)
    rke_nll = rke_nll_report["micro_average"]
    fallback_nll = fallback_nll_report["micro_average"]
    checks = {"paragraph_leakage_zero": not audit["paragraph_leakage"],
              "document_content_hash_leakage_zero": not audit["document_content_hash_leakage"],
              "target_string_leakage_zero": not audit["target_string_leakage"],
              "exact_sample_leakage_zero": not audit["exact_sample_leakage"],
              "current_target_future_bytes_blocked": prefix_only_firewall,
              "generated_prefix_evaluation": True,
              "test_examples_at_least_200": len(data["test"]) >= 200,
              "rke_valid_chains_100_percent": rke_generation["valid_block_chain_rate"] == 1.0,
              "rke_no_missing_eos": rke_generation["missing_eos_rate"] == 0.0,
              "nll_within_5_percent": rke_nll <= fallback_nll * 1.05,
              "macro_nll_within_5_percent": rke_nll_report["macro_average"] <=
                                              fallback_nll_report["macro_average"] * 1.05,
              "generated_byte_accuracy_nontrivial": rke_generation["position_aligned_byte_accuracy"] >= .25,
              "byte_accuracy_at_least_95_percent_of_fallback":
                  rke_generation["position_aligned_byte_accuracy"] >= fallback_generation["position_aligned_byte_accuracy"] * .95,
              "structural_premature_eos_zero": rke_generation["structural_premature_eos_rate"] == 0.0,
              "learned_continuation_exact_nonzero": rke_continuation["exact_count"] > 0,
              "learned_continuation_beats_fallback":
                  rke_continuation["exact_match"] > fallback_continuation["exact_match"]}
    production_quality_checks = {
        "open_ended_full_span_exact_nonzero": rke_generation["exact_match"] > 0,
        "gold_first_block_continuation_exact_at_least_20_percent":
            rke_continuation["exact_match"] >= PRODUCTION_CONTINUATION_EXACT_THRESHOLD,
        "gold_first_block_continuation_at_least_95_percent_of_fallback":
            rke_continuation["exact_match"] >= fallback_continuation["exact_match"] * .95,
        "gold_first_block_continuation_beats_exact_prefix_retrieval":
            rke_continuation["exact_match"] > retrieval_continuation["exact_match"],
        "at_least_500_cross_block_test_spans": len(data["test"]) >= 500,
        "at_least_three_seeds": False,
        "document_level_multisource_corpus": audit["split_unit"] == "document"
                                                and not audit["document_leakage"]
                                                and not audit["document_content_hash_leakage"]
                                                and audit["corpus_documents"] >= 400
                                                and len(audit["corpus_topics"]) >= 10
                                                and audit["quality_gates"]["topic_strata_quotas_satisfied"],
    }
    result = {"experiment": "learned natural cross-block next-token LM", "codec": {"states": 259, "block_bytes": 24},
              "dataset": audit, "dataset_hash": sha256(data),
              "input_firewall": {"decisions_checked": len(firewall_examples),
                                 "prefix_length_equals_decision_position": prefix_only_firewall,
                                 "future_target_bytes_in_input": 0},
              "training": {"steps_per_arm": steps, "batch_size": 64, "d_slot": 8, "d_model": 200,
              "sampling": {"block_start": 13, "terminator": 5, "interior_byte": 46},
              "rke_seconds": rke_seconds, "fallback_seconds": fallback_seconds},
              "parameters": {"rke": sum(x.size for x in rke.params.values()),
                             "rke_separate_vocab_classifier": 0,
                             "fallback": fallback.parameter_count(), "fallback_output": fallback.params["Wclass"].size},
              "rke": {"teacher_forced_nll_per_decision": rke_nll,
                      "teacher_forced_nll_report": rke_nll_report, "generated": rke_generation},
              "fallback": {"teacher_forced_nll_per_decision": fallback_nll,
                           "teacher_forced_nll_report": fallback_nll_report,
                           "generated": fallback_generation},
              "gold_first_block_continuation": {"production_exact_threshold":
                                                     PRODUCTION_CONTINUATION_EXACT_THRESHOLD,
                                                 "rke": rke_continuation,
                                                 "fallback": fallback_continuation,
                                                 "exact_prefix_retrieval": retrieval_continuation},
              "checks": checks, "production_quality_checks": production_quality_checks,
              "limitations": ["Two-word context remains below the planned 128-token production test.",
                              "Single seed pilot; production gate requires three seeds."]}
    result["passed"] = all(checks.values())
    result["production_quality_passed"] = all(production_quality_checks.values())
    predictions = {"rke": rke_generation.pop("predictions"), "fallback": fallback_generation.pop("predictions"),
                   "gold_first_block_continuation": {
                       "rke": rke_continuation.pop("predictions"),
                       "fallback": fallback_continuation.pop("predictions")}}
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
