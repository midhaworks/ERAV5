"""Natural multilingual next-word pilot using local Wikipedia-derived corpora."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from lm_compare import VocabTransformer, calibration, softmax
from rke import (Adam, FullByteCodec, MaskedSlotRKE, RefinedSlotRKE, TinyTransformer, sha256,
                 utf8_is_complete, utf8_prefix_is_valid)


ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    "English": ROOT / "S2/tokenizers/bpe/corpus/en.faithful.txt",
    "Hindi": ROOT / "S2/tokenizers/bpe/corpus/hi.faithful.txt",
    "Telugu": ROOT / "S2/tokenizers/bpe/corpus/te.faithful.txt",
    "Sindhi": ROOT / "S2/tokenizers/bpe/corpus/sd.faithful.txt",
}
MAX_BYTES = 24
VOCAB_SIZE = 512


def tokenize(text: str) -> list[str]:
    words, current = [], []
    for char in unicodedata.normalize("NFC", text):
        category = unicodedata.category(char)
        if category[0] in "LMN":
            current.append(char.casefold())
        elif current:
            token = "".join(current)
            if any(unicodedata.category(x).startswith("L") for x in token):
                words.append(token)
            current = []
    if current:
        token = "".join(current)
        if any(unicodedata.category(x).startswith("L") for x in token):
            words.append(token)
    return words


def build_dataset(cap_per_language: dict[str, int] | None = None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    caps = cap_per_language or {"train": 1000, "validation": 200, "test": 200}
    collected = {split: [] for split in caps}
    audit: dict[str, Any] = {}
    for language, path in SOURCES.items():
        raw = path.read_text(encoding="utf-8")
        paragraphs = [part for part in re.split(r"\n\s*\n", raw) if part.strip()]
        candidates = {split: [] for split in caps}
        total_words = eligible_words = total_transitions = eligible_transitions = 0
        for paragraph_index, paragraph in enumerate(paragraphs):
            words = tokenize(paragraph)
            total_words += len(words)
            eligible_words += sum(len(word.encode("utf-8")) <= MAX_BYTES for word in words)
            paragraph_key = int(hashlib.sha256(f"{language}:{paragraph_index}:{paragraph[:200]}".encode()).hexdigest(), 16) % 100
            split = "train" if paragraph_key < 80 else "validation" if paragraph_key < 90 else "test"
            for index in range(2, len(words)):
                total_transitions += 1
                context, target = words[index - 2:index], words[index]
                if any(len(word.encode("utf-8")) > MAX_BYTES for word in [*context, target]):
                    continue
                eligible_transitions += 1
                record = {"language": language, "context": context, "target": target,
                          "target_bytes": len(target.encode("utf-8")), "paragraph": paragraph_index}
                candidates[split].append(record)
        for split, values in candidates.items():
            ordered = sorted(values, key=lambda x: sha256({"language": language, "split": split, **x}))
            chosen = ordered[:caps[split]]
            if len(chosen) < caps[split]:
                raise AssertionError(f"insufficient {language} {split} examples: {len(chosen)}")
            collected[split].extend(chosen)
        audit[language] = {
            "source": str(path.relative_to(ROOT)), "source_hash": sha256(path.read_bytes()),
            "paragraphs": len(paragraphs), "total_words": total_words, "eligible_words": eligible_words,
            "word_byte_cap_coverage": eligible_words / max(1, total_words),
            "total_transitions": total_transitions, "eligible_transitions": eligible_transitions,
            "transition_coverage": eligible_transitions / max(1, total_transitions),
            "selected": {split: caps[split] for split in caps},
        }
    # A paragraph may belong to only one split by construction.
    paragraphs_by_split = {split: {(x["language"], x["paragraph"]) for x in values}
                           for split, values in collected.items()}
    leakage = any(paragraphs_by_split[a] & paragraphs_by_split[b]
                  for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
    audit["split_policy"] = {"unit": "paragraph", "hash_buckets": "80/10/10", "paragraph_leakage": leakage,
                             "max_target_bytes": MAX_BYTES}
    if leakage:
        raise AssertionError("paragraph leakage")
    return collected, audit


def features(codec: FullByteCodec, records: list[dict[str, Any]], prefixes: list[bytes] | None = None) -> np.ndarray:
    if prefixes is None:
        return np.stack([np.stack([codec.encode(b"c"), codec.encode(x["context"][0].encode("utf-8")),
                                   codec.encode(x["context"][1].encode("utf-8"))]) for x in records])
    return np.stack([np.stack([codec.encode(x["context"][0].encode("utf-8")),
                               codec.encode(x["context"][1].encode("utf-8")), codec.encode(prefix)])
                     for x, prefix in zip(records, prefixes)])


def train_rke(codec: FullByteCodec, records: list[dict[str, Any]], seed: int = 501,
              steps: int = 800) -> tuple[TinyTransformer, list[dict[str, float]], float]:
    model, rng = TinyTransformer(codec, d_slot=4, seed=seed), np.random.default_rng(seed + 1)
    optimizer, curve, started = Adam(model.params, lr=.004), [], time.perf_counter()
    for step in range(1, steps + 1):
        batch = [records[int(i)] for i in rng.integers(0, len(records), size=64)]
        targets = np.stack([codec.ids(x["target"].encode("utf-8")) for x in batch])
        loss, grads = model.loss_and_grads(features(codec, batch), targets, targets != 0)
        optimizer.update(model.params, grads)
        if step == 1 or step % 100 == 0:
            curve.append({"step": step, "loss": loss})
    return model, curve, time.perf_counter() - started


def train_causal_rke(codec: FullByteCodec, records: list[dict[str, Any]], seed: int = 551,
                     steps: int = 5000) -> tuple[TinyTransformer, list[dict[str, float]], float]:
    """Teacher-force byte prefixes while retaining the tied RKE codebook.

    Only the next position bears loss. Unlike the parallel arm, this lets the
    prediction at slot p depend on the true slots <p without adding a 258-way
    output matrix or any vocabulary-sized parameters.
    """
    examples, rng = fallback_training_examples(records), np.random.default_rng(seed + 1)
    model = TinyTransformer(codec, d_slot=4, seed=seed)
    optimizer, curve, started = Adam(model.params, lr=.004), [], time.perf_counter()
    for step in range(1, steps + 1):
        batch = [examples[int(i)] for i in rng.integers(0, len(examples), size=64)]
        prefixes = [x[1] for x in batch]
        positions = np.array([len(prefix) for prefix in prefixes])
        targets = np.zeros((len(batch), codec.slots), dtype=np.int64)
        mask = np.zeros_like(targets, dtype=bool)
        # fallback target 0..255 is a byte; 256 is EOS. RKE states are
        # PAD=0, EOS=1, byte=byte+2.
        states = np.array([1 if x[2] == 256 else x[2] + 2 for x in batch])
        targets[np.arange(len(batch)), positions] = states
        mask[np.arange(len(batch)), positions] = True
        loss, grads = model.loss_and_grads(features(codec, [x[0] for x in batch], prefixes), targets, mask)
        optimizer.update(model.params, grads)
        if step == 1 or step % 250 == 0:
            curve.append({"step": step, "loss": loss})
    return model, curve, time.perf_counter() - started


def train_masked_rke(codec: FullByteCodec, records: list[dict[str, Any]], seed: int = 576,
                     steps: int = 2500) -> tuple[MaskedSlotRKE, list[dict[str, float]], float]:
    model = MaskedSlotRKE(codec, d_slot=4, kernel_size=8, seed=seed)
    rng = np.random.default_rng(seed + 1)
    optimizer, curve, started = Adam(model.params, lr=.004), [], time.perf_counter()
    for step in range(1, steps + 1):
        batch = [records[int(i)] for i in rng.integers(0, len(records), size=64)]
        targets = np.stack([codec.ids(x["target"].encode("utf-8")) for x in batch])
        loss, grads = model.loss_and_grads(features(codec, batch), targets, targets != 0)
        optimizer.update(model.params, grads)
        if step == 1 or step % 100 == 0:
            curve.append({"step": step, "loss": loss})
    return model, curve, time.perf_counter() - started


def train_refined_rke(codec: FullByteCodec, records: list[dict[str, Any]], seed: int = 501,
                      steps: int = 2500) -> tuple[RefinedSlotRKE, list[dict[str, float]], float]:
    model = RefinedSlotRKE(codec, d_slot=4, kernel_size=8, auxiliary_weight=.25, seed=seed)
    rng = np.random.default_rng(seed + 1)
    optimizer, curve, started = Adam(model.params, lr=.004), [], time.perf_counter()
    for step in range(1, steps + 1):
        batch = [records[int(i)] for i in rng.integers(0, len(records), size=64)]
        targets = np.stack([codec.ids(x["target"].encode("utf-8")) for x in batch])
        loss, grads = model.loss_and_grads(features(codec, batch), targets, targets != 0)
        optimizer.update(model.params, grads)
        if step == 1 or step % 250 == 0:
            curve.append({"step": step, "loss": loss})
    return model, curve, time.perf_counter() - started


def train_vocab(codec: FullByteCodec, records: list[dict[str, Any]], vocabulary: list[str], seed: int = 601,
                steps: int = 800) -> tuple[VocabTransformer, list[dict[str, float]], float]:
    target_to_id = {word: index for index, word in enumerate(vocabulary)}
    model, rng = VocabTransformer(codec, len(vocabulary) + 1, d_slot=4, seed=seed), np.random.default_rng(seed + 1)
    optimizer, curve, started = Adam(model.params, lr=.004), [], time.perf_counter()
    for step in range(1, steps + 1):
        batch = [records[int(i)] for i in rng.integers(0, len(records), size=64)]
        targets = np.array([target_to_id.get(x["target"], len(vocabulary)) for x in batch])
        loss, grads = model.loss_and_grads(features(codec, batch), targets)
        optimizer.update(model.params, grads)
        if step == 1 or step % 100 == 0:
            curve.append({"step": step, "loss": loss})
    return model, curve, time.perf_counter() - started


def fallback_training_examples(records: list[dict[str, Any]]) -> list[tuple[dict[str, Any], bytes, int]]:
    values = []
    for record in records:
        payload = record["target"].encode("utf-8")
        for index, byte in enumerate(payload):
            values.append((record, payload[:index], byte))
        values.append((record, payload, 256))  # EOS class
    return values


def train_fallback(codec: FullByteCodec, records: list[dict[str, Any]], seed: int = 551,
                   steps: int = 5000) -> tuple[VocabTransformer, list[dict[str, float]], float]:
    examples, rng = fallback_training_examples(records), np.random.default_rng(seed + 1)
    model = VocabTransformer(codec, 257, d_slot=4, seed=seed)
    optimizer, curve, started = Adam(model.params, lr=.004), [], time.perf_counter()
    for step in range(1, steps + 1):
        batch = [examples[int(i)] for i in rng.integers(0, len(examples), size=64)]
        loss, grads = model.loss_and_grads(features(codec, [x[0] for x in batch], [x[1] for x in batch]),
                                           np.array([x[2] for x in batch]))
        optimizer.update(model.params, grads)
        if step == 1 or step % 250 == 0:
            curve.append({"step": step, "loss": loss})
    return model, curve, time.perf_counter() - started


def _choose_fallback(row: np.ndarray, prefix: bytes) -> int | None:
    for value in np.argsort(row)[::-1]:
        value = int(value)
        if value == 256:
            if utf8_is_complete(prefix):
                return None
        elif len(prefix) < MAX_BYTES and utf8_prefix_is_valid(prefix + bytes([value])):
            return value
    return -1  # no valid continuation/EOS within the configured byte budget


def _choose_rke_state(row: np.ndarray, prefix: bytes) -> int | None:
    for state in np.argsort(row)[::-1]:
        state = int(state)
        if state == 0:
            continue
        if state == 1:
            if utf8_is_complete(prefix):
                return None
        elif len(prefix) < MAX_BYTES and utf8_prefix_is_valid(prefix + bytes([state - 2])):
            return state - 2
    return -1


def decode_fallback(model: VocabTransformer, codec: FullByteCodec,
                    records: list[dict[str, Any]]) -> list[str]:
    prefixes, finished = [b""] * len(records), np.zeros(len(records), dtype=bool)
    failed = np.zeros(len(records), dtype=bool)
    for _ in range(MAX_BYTES + 1):
        probs = model.probabilities(features(codec, records, prefixes))
        for index, row in enumerate(probs):
            if finished[index]:
                continue
            chosen = _choose_fallback(row, prefixes[index])
            if chosen == -1:
                failed[index] = True; finished[index] = True
            elif chosen is None:
                finished[index] = True
            else:
                prefixes[index] += bytes([chosen])
        if finished.all():
            break
    output = []
    for index, (prefix, done) in enumerate(zip(prefixes, finished)):
        try: output.append("<NO_VALID_PATH>" if failed[index] else prefix.decode("utf-8") if done else "<NO_EOS>")
        except UnicodeDecodeError: output.append("<INVALID_UTF8>")
    return output


def decode_causal_rke(model: TinyTransformer, codec: FullByteCodec,
                      records: list[dict[str, Any]]) -> list[str]:
    prefixes, finished = [b""] * len(records), np.zeros(len(records), dtype=bool)
    failed = np.zeros(len(records), dtype=bool)
    for _ in range(MAX_BYTES + 1):
        logits, _ = model.forward(features(codec, records, prefixes))
        for index, rows in enumerate(logits):
            if finished[index]:
                continue
            chosen = _choose_rke_state(rows[len(prefixes[index])], prefixes[index])
            if chosen == -1:
                failed[index] = True; finished[index] = True
            elif chosen is None:
                finished[index] = True
            else:
                prefixes[index] += bytes([chosen])
        if finished.all():
            break
    output = []
    for index, (prefix, done) in enumerate(zip(prefixes, finished)):
        try: output.append("<NO_VALID_PATH>" if failed[index] else prefix.decode("utf-8") if done else "<NO_EOS>")
        except UnicodeDecodeError: output.append("<INVALID_UTF8>")
    return output


def evaluate_rke(model: TinyTransformer, codec: FullByteCodec, records: list[dict[str, Any]]) -> dict[str, Any]:
    logits, _ = model.forward(features(codec, records)); probabilities = softmax(logits)
    targets = np.stack([codec.ids(x["target"].encode("utf-8")) for x in records]); mask = targets != 0
    selected = probabilities[np.arange(len(records))[:, None], np.arange(codec.slots)[None], targets]
    word_nll = (-np.log(selected + 1e-12) * mask).sum(axis=1)
    predictions, valid = [], []
    from rke import constrained_utf8_decode
    for item in logits:
        try: predictions.append(constrained_utf8_decode(item, MAX_BYTES).decode("utf-8")); valid.append(True)
        except (ValueError, UnicodeDecodeError): predictions.append("<INVALID>"); valid.append(False)
    return aggregate(records, predictions, word_nll, mask.sum(axis=1), np.exp(-word_nll), np.array(valid))


def evaluate_fallback(model: VocabTransformer, codec: FullByteCodec, records: list[dict[str, Any]]) -> dict[str, Any]:
    examples = fallback_training_examples(records)
    probs = model.probabilities(features(codec, [x[0] for x in examples], [x[1] for x in examples]))
    targets = np.array([x[2] for x in examples]); losses = -np.log(probs[np.arange(len(examples)), targets] + 1e-12)
    word_losses, lengths, cursor = [], [], 0
    for record in records:
        size = len(record["target"].encode("utf-8")) + 1
        word_losses.append(float(losses[cursor:cursor + size].sum())); lengths.append(size); cursor += size
    predictions = decode_fallback(model, codec, records)
    valid = np.array([x not in ("<NO_EOS>", "<INVALID_UTF8>", "<NO_VALID_PATH>") for x in predictions])
    return aggregate(records, predictions, np.array(word_losses), np.array(lengths), np.exp(-np.array(word_losses)), valid)


def evaluate_causal_rke(model: TinyTransformer, codec: FullByteCodec,
                        records: list[dict[str, Any]]) -> dict[str, Any]:
    examples = fallback_training_examples(records)
    prefixes = [x[1] for x in examples]
    logits, _ = model.forward(features(codec, [x[0] for x in examples], prefixes))
    positions = np.array([len(prefix) for prefix in prefixes])
    states = np.array([1 if x[2] == 256 else x[2] + 2 for x in examples])
    probabilities = softmax(logits[np.arange(len(examples)), positions])
    losses = -np.log(probabilities[np.arange(len(examples)), states] + 1e-12)
    word_losses, lengths, cursor = [], [], 0
    for record in records:
        size = len(record["target"].encode("utf-8")) + 1
        word_losses.append(float(losses[cursor:cursor + size].sum())); lengths.append(size); cursor += size
    predictions = decode_causal_rke(model, codec, records)
    valid = np.array([x not in ("<NO_EOS>", "<INVALID_UTF8>", "<NO_VALID_PATH>") for x in predictions])
    return aggregate(records, predictions, np.array(word_losses), np.array(lengths),
                     np.exp(-np.array(word_losses)), valid)


def evaluate_vocab(model: VocabTransformer, codec: FullByteCodec, records: list[dict[str, Any]],
                   vocabulary: list[str]) -> dict[str, Any]:
    lookup = {word: index for index, word in enumerate(vocabulary)}
    probs = model.probabilities(features(codec, records)); predicted = probs.argmax(axis=1)
    predictions = [vocabulary[index] if index < len(vocabulary) else "<UNK>" for index in predicted]
    representable = np.array([x["target"] in lookup for x in records])
    nll = np.full(len(records), np.nan)
    for index, record in enumerate(records):
        if representable[index]: nll[index] = -np.log(probs[index, lookup[record["target"]]] + 1e-12)
    exact = np.array([x["target"] == prediction for x, prediction in zip(records, predictions)])
    confidence = probs.max(axis=1)
    per_language = {}
    for language in SOURCES:
        indices = np.array([i for i, x in enumerate(records) if x["language"] == language])
        available = indices[representable[indices]]
        per_language[language] = {"examples": len(indices), "representable_fraction": float(representable[indices].mean()),
                                  "exact_match": float(exact[indices].mean()),
                                  "nll_per_word_on_representable": float(np.nanmean(nll[available])) if len(available) else None}
    return {"examples": len(records), "representable_fraction": float(representable.mean()),
            "exact_match": float(exact.mean()), "nll_per_word_on_representable": float(np.nanmean(nll)) if representable.any() else None,
            "oov_true_nll": None, "oov_note": "undefined because absent words have no output row",
            "calibration": calibration(confidence, exact), "per_language": per_language, "predictions": predictions}


def aggregate(records: list[dict[str, Any]], predictions: list[str], word_nll: np.ndarray,
              decisions: np.ndarray, confidence: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    exact = np.array([x["target"] == prediction for x, prediction in zip(records, predictions)])
    per_language, by_length = {}, {}
    for language in SOURCES:
        indices = np.array([i for i, x in enumerate(records) if x["language"] == language])
        per_language[language] = {"examples": len(indices), "exact_match": float(exact[indices].mean()),
                                  "nll_per_byte_or_eos": float(word_nll[indices].sum() / decisions[indices].sum()),
                                  "valid_utf8_rate": float(valid[indices].mean())}
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        length = record["target_bytes"]
        label = "1-6" if length <= 6 else "7-12" if length <= 12 else "13-18" if length <= 18 else "19-24"
        buckets[label].append(index)
    for label, values in buckets.items():
        indices = np.array(values)
        by_length[label] = {"examples": len(indices), "exact_match": float(exact[indices].mean()),
                            "nll_per_byte_or_eos": float(word_nll[indices].sum() / decisions[indices].sum())}
    return {"examples": len(records), "exact_match": float(exact.mean()),
            "nll_per_word": float(word_nll.mean()), "nll_per_byte_or_eos": float(word_nll.sum() / decisions.sum()),
            "valid_utf8_rate": float(valid.mean()), "calibration": calibration(confidence, exact),
            "per_language": per_language, "per_byte_length": by_length, "predictions": predictions}


def benchmark(fn: Callable[[], Any], examples: int, repeats: int = 5) -> dict[str, float]:
    fn(); timings = []
    for _ in range(repeats):
        start = time.perf_counter(); fn(); timings.append(time.perf_counter() - start)
    median = statistics.median(timings)
    return {"median_seconds": median, "examples_per_second": examples / median, "repeats": repeats}


def parameter_hash(params: dict[str, np.ndarray]) -> str:
    return sha256(b"".join(key.encode() + params[key].tobytes() for key in sorted(params)))


def sample_stream_hash(example_count: int, seed: int, steps: int, batch_size: int = 64) -> str:
    rng, digest = np.random.default_rng(seed + 1), hashlib.sha256()
    for _ in range(steps):
        digest.update(rng.integers(0, example_count, size=batch_size).tobytes())
    return digest.hexdigest()


def all_finite_where_defined(value: Any) -> bool:
    """Recursively reject NaN/Inf while allowing strings, booleans and None."""
    if isinstance(value, dict):
        return all(all_finite_where_defined(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite_where_defined(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def summarize_seed_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"runs": runs}
    # Student's t is required when the variance is estimated from only three
    # runs. A normal 1.96 multiplier materially understates uncertainty here.
    t95 = 4.302652729911275 if len(runs) == 3 else 1.96
    for metric in ("nll_per_byte_or_eos", "exact_match"):
        values = np.array([run[metric] for run in runs], dtype=np.float64)
        std = float(values.std(ddof=1))
        summary[metric] = {"mean": float(values.mean()), "sample_std": std,
                           "ci95_half_width": float(t95 * std / np.sqrt(len(values))),
                           "min": float(values.min()), "max": float(values.max())}
    summary["ci95_method"] = ("two-sided Student t, df=2" if len(runs) == 3
                              else "normal approximation")
    return summary


def run_natural_corpus_experiment(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    data, corpus_audit = build_dataset()
    codec = FullByteCodec(MAX_BYTES)
    counts = Counter(x["target"] for x in data["train"])
    vocabulary = [word for word, _ in counts.most_common(VOCAB_SIZE)]
    rke, rke_curve, rke_seconds = train_rke(codec, data["train"])
    masked_rke, masked_curve, masked_seconds = train_masked_rke(codec, data["train"])
    refined_rke, refined_curve, refined_seconds = train_refined_rke(codec, data["train"])
    causal_rke, causal_curve, causal_seconds = train_causal_rke(codec, data["train"])
    vocab, vocab_curve, vocab_seconds = train_vocab(codec, data["train"], vocabulary)
    fallback, fallback_curve, fallback_seconds = train_fallback(codec, data["train"], seed=551)
    evaluations = {}
    for split in ("validation", "test"):
        evaluations[split] = {
            "rke": evaluate_rke(rke, codec, data[split]),
            "masked_rke": evaluate_rke(masked_rke, codec, data[split]),
            "refined_rke": evaluate_rke(refined_rke, codec, data[split]),
            "causal_rke": evaluate_causal_rke(causal_rke, codec, data[split]),
            "vocabulary": evaluate_vocab(vocab, codec, data[split], vocabulary),
            "byte_fallback": evaluate_fallback(fallback, codec, data[split]),
        }
    # Production-test preflight uses three fixed seeds at a convergence budget.
    causal_seed_models, fallback_seed_models = {551: causal_rke}, {551: fallback}
    seed_training_seconds = {"causal_rke": {551: causal_seconds},
                             "byte_fallback": {551: fallback_seconds}}
    for seed in (552, 553):
        model, _, seconds = train_causal_rke(codec, data["train"], seed=seed)
        causal_seed_models[seed] = model; seed_training_seconds["causal_rke"][seed] = seconds
    for seed in (552, 553):
        model, _, seconds = train_fallback(codec, data["train"], seed=seed)
        fallback_seed_models[seed] = model; seed_training_seconds["byte_fallback"][seed] = seconds
    causal_runs = []
    for seed, model in causal_seed_models.items():
        metric = evaluate_causal_rke(model, codec, data["test"])
        causal_runs.append({"seed": seed, "training_seconds": seed_training_seconds["causal_rke"][seed],
                            "nll_per_byte_or_eos": metric["nll_per_byte_or_eos"],
                            "exact_match": metric["exact_match"], "model_hash": parameter_hash(model.params)})
    fallback_runs = []
    for seed, model in fallback_seed_models.items():
        metric = evaluate_fallback(model, codec, data["test"])
        fallback_runs.append({"seed": seed, "training_seconds": seed_training_seconds["byte_fallback"][seed],
                              "nll_per_byte_or_eos": metric["nll_per_byte_or_eos"],
                              "exact_match": metric["exact_match"], "model_hash": parameter_hash(model.params)})
    seed_stability = {"causal_rke": summarize_seed_runs(causal_runs),
                      "byte_fallback": summarize_seed_runs(fallback_runs), "seeds_per_arm": 3}
    initial_rke = TinyTransformer(codec, d_slot=4, seed=551)
    initial_fallback = VocabTransformer(codec, 257, d_slot=4, seed=551)
    shared_names = sorted(set(initial_rke.params) & set(initial_fallback.params))
    shared_rke_hash = parameter_hash({name: initial_rke.params[name] for name in shared_names})
    shared_fallback_hash = parameter_hash({name: initial_fallback.params[name] for name in shared_names})
    decision_count = len(fallback_training_examples(data["train"]))
    stream_hashes = {
        str(seed): {arm: sample_stream_hash(decision_count, seed, 5000)
                    for arm in ("causal_rke", "byte_fallback")}
        for seed in (551, 552, 553)
    }
    seed_stability["matched_controls"] = {
        "paired_seeds": [551, 552, 553], "optimizer_steps_per_arm": 5000,
        "initial_shared_parameter_names": shared_names,
        "rke_initial_shared_hash": shared_rke_hash,
        "fallback_initial_shared_hash": shared_fallback_hash,
        "identical_initial_shared_body": shared_rke_hash == shared_fallback_hash,
        "batch_stream_hash_by_seed_and_arm": stream_hashes,
        "identical_batch_stream_per_seed": all(
            value["causal_rke"] == value["byte_fallback"] for value in stream_hashes.values()),
    }
    causal_mean, fallback_mean = seed_stability["causal_rke"], seed_stability["byte_fallback"]
    seed_stability["within_5_percent_mean_parity"] = bool(
        causal_mean["nll_per_byte_or_eos"]["mean"] <= fallback_mean["nll_per_byte_or_eos"]["mean"] * 1.05
        and causal_mean["exact_match"]["mean"] >= fallback_mean["exact_match"]["mean"] * .95)
    test = data["test"]
    test_features = features(codec, test)
    speeds = {
        "rke": benchmark(lambda: rke.forward(test_features)[0].argmax(axis=-1), len(test)),
        "masked_rke": benchmark(lambda: masked_rke.forward(test_features)[0].argmax(axis=-1), len(test)),
        "refined_rke": benchmark(lambda: refined_rke.forward(test_features)[0].argmax(axis=-1), len(test)),
        "causal_rke": benchmark(lambda: decode_causal_rke(causal_rke, codec, test), len(test)),
        "vocabulary": benchmark(lambda: vocab.probabilities(test_features).argmax(axis=-1), len(test)),
        "byte_fallback": benchmark(lambda: decode_fallback(fallback, codec, test), len(test)),
        "environment": f"NumPy {np.__version__}; CPU; batch={len(test)}",
    }
    results = {
        "experiment": "natural multilingual next-word pilot", "corpus": corpus_audit,
        "split_sizes": {split: len(values) for split, values in data.items()},
        "dataset_hash": sha256(data), "codec": {"states": 258, "max_bytes": MAX_BYTES},
        "vocabulary": {"size": VOCAB_SIZE, "hash": sha256(vocabulary)},
        "training": {"rke": {"steps": 800, "seconds": rke_seconds},
                     "masked_rke": {"steps": 2500, "seconds": masked_seconds, "kernel_size": 8},
                     "refined_rke": {"steps": 2500, "seconds": refined_seconds, "kernel_size": 8,
                                     "passes": 2, "auxiliary_weight": .25,
                                     "proposal_temperature": .25},
                     "causal_rke": {"steps": 5000, "seconds": causal_seconds},
                     "vocabulary": {"steps": 800, "seconds": vocab_seconds},
                     "byte_fallback": {"steps": 5000, "seconds": fallback_seconds}, "batch_size": 64},
        "model_hashes": {"rke": parameter_hash(rke.params), "masked_rke": parameter_hash(masked_rke.params),
                         "refined_rke": parameter_hash(refined_rke.params),
                         "causal_rke": parameter_hash(causal_rke.params),
                         "vocabulary": parameter_hash(vocab.params),
                         "byte_fallback": parameter_hash(fallback.params)},
        "parameters": {"rke": sum(x.size for x in rke.params.values()),
                       "masked_rke": sum(x.size for x in masked_rke.params.values()),
                       "refined_rke": sum(x.size for x in refined_rke.params.values()),
                       "causal_rke": sum(x.size for x in causal_rke.params.values()),
                       "vocabulary": vocab.parameter_count(), "byte_fallback": fallback.parameter_count(),
                       "rke_separate_vocab_classifier": 0,
                       "masked_rke_separate_vocab_classifier": 0,
                       "refined_rke_separate_vocab_classifier": 0,
                       "causal_rke_separate_vocab_classifier": 0,
                       "vocabulary_output": vocab.params["Wclass"].size,
                       "byte_fallback_output": fallback.params["Wclass"].size},
        "evaluation": evaluations, "seed_stability": seed_stability, "decode_speed": speeds,
        "status": "research pilot; results are descriptive and not a production gate",
        "completion_checks": {"paragraph_split_no_leakage": not corpus_audit["split_policy"]["paragraph_leakage"],
                              "all_metrics_finite_where_defined": all_finite_where_defined(
                                  {"evaluation": evaluations, "seed_stability": seed_stability,
                                   "decode_speed": speeds}),
                              "all_models_saved": False,
                              "natural_corpora_used": all(path.is_file() for path in SOURCES.values())},
        "limitations": ["Small Wikipedia-derived samples from four files, not web-scale pretraining.",
                        "Paragraph hashing reduces but cannot eliminate semantic overlap within one article per language.",
                        "The 24-byte cap excludes a measured fraction of long words, especially Indic text.",
                        "Training budgets are documented but not FLOP-matched across parallel and autoregressive targets.",
                        "CPU NumPy throughput is not production GPU throughput."],
    }
    predictions = {}
    for split in ("validation", "test"):
        predictions[split] = [{"language": record["language"], "context": record["context"], "target": record["target"],
                               "target_bytes": record["target_bytes"],
                               "rke": evaluations[split]["rke"]["predictions"][i],
                               "masked_rke": evaluations[split]["masked_rke"]["predictions"][i],
                               "refined_rke": evaluations[split]["refined_rke"]["predictions"][i],
                               "causal_rke": evaluations[split]["causal_rke"]["predictions"][i],
                               "vocabulary": evaluations[split]["vocabulary"]["predictions"][i],
                               "byte_fallback": evaluations[split]["byte_fallback"]["predictions"][i]}
                              for i, record in enumerate(data[split])]
        for arm in evaluations[split].values(): arm.pop("predictions", None)
    (output / "split.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (output / "predictions.json").write_text(json.dumps(predictions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "vocabulary.json").write_text(json.dumps(vocabulary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, curve in (("rke", rke_curve), ("masked_rke", masked_curve),
                        ("refined_rke", refined_curve), ("causal_rke", causal_curve),
                        ("vocabulary", vocab_curve), ("byte_fallback", fallback_curve)):
        with (output / f"{name}_curve.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=curve[0], lineterminator="\n")
            writer.writeheader(); writer.writerows(curve)
    np.savez_compressed(output / "rke_model.npz", **rke.params)
    np.savez_compressed(output / "masked_rke_model.npz", **masked_rke.params)
    np.savez_compressed(output / "refined_rke_model.npz", **refined_rke.params)
    np.savez_compressed(output / "causal_rke_model.npz", **causal_rke.params)
    np.savez_compressed(output / "vocabulary_model.npz", **vocab.params)
    np.savez_compressed(output / "byte_fallback_model.npz", **fallback.params)
    for seed, model in causal_seed_models.items():
        np.savez_compressed(output / f"causal_rke_seed_{seed}_model.npz", **model.params)
    for seed, model in fallback_seed_models.items():
        np.savez_compressed(output / f"byte_fallback_seed_{seed}_model.npz", **model.params)
    expected_models = [output / f"{name}_model.npz" for name in
                       ("rke", "masked_rke", "refined_rke", "causal_rke", "vocabulary", "byte_fallback")]
    expected_models += [output / f"causal_rke_seed_{seed}_model.npz" for seed in causal_seed_models]
    expected_models += [output / f"byte_fallback_seed_{seed}_model.npz" for seed in fallback_seed_models]
    results["completion_checks"]["all_models_saved"] = all(path.is_file() for path in expected_models)
    results["completed"] = all(results["completion_checks"].values())
    (output / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return results
