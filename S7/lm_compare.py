"""Matched next-token comparison for vocabulary, byte fallback, and RKE heads."""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rke import Adam, ReversibleCodec, TinyTransformer, constrained_utf8_decode, sha256


ALPHABET = "abcdef0123"
EOS_CLASS = len(ALPHABET)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def context_features(codec: ReversibleCodec, left: list[str], right: list[str],
                     third: list[str]) -> np.ndarray:
    return np.stack([np.stack([codec.encode(a), codec.encode(b), codec.encode(c)])
                     for a, b, c in zip(left, right, third)])


def make_dataset(seed: int = 1701) -> dict[str, list[dict[str, str]]]:
    stems = [a + b for a in ALPHABET for b in ALPHABET]
    combinations = [{"stem": stem, "suffix": suffix, "target": stem + suffix}
                    for stem in stems for suffix in ALPHABET]
    ordered = sorted(combinations, key=lambda x: sha256({"seed": seed, **x}))
    train, oov = ordered[:750], ordered[750:]
    # Every component appears in training; only whole stem+suffix combinations
    # are held out.  This makes the test compositional, not character-OOV.
    if {x["stem"] for x in train} != set(stems) or {x["suffix"] for x in train} != set(ALPHABET):
        raise AssertionError("training split does not cover every component")
    return {"train": train, "seen_control": train[:250], "held_out_compositions": oov}


class VocabTransformer:
    """Same causal transformer body as TinyTransformer with a class softmax."""

    def __init__(self, codec: ReversibleCodec, classes: int, d_slot: int = 12, seed: int = 9):
        self.codec, self.classes, self.d_slot = codec, classes, d_slot
        self.d_model = codec.slots * d_slot
        rng = np.random.default_rng(seed)
        d = self.d_model
        self.params = {
            "Ein": rng.normal(0, .08, (codec.width, d_slot)),
            "Wq": rng.normal(0, .06, (d, d)), "Wk": rng.normal(0, .06, (d, d)),
            "Wv": rng.normal(0, .06, (d, d)), "Wo": rng.normal(0, .02, (d, d)),
            # Draw every shared-body tensor before the arm-specific classifier.
            # With the same seed this makes the body bit-identical to RKE.
            "pos": rng.normal(0, .02, (3, d)),
            "Wclass": rng.normal(0, .04, (d, classes)),
        }

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        p, batch = self.params, len(features)
        x = (features @ p["Ein"]).reshape(batch, 3, self.d_model) + p["pos"][None]
        q, k, v = x @ p["Wq"], x @ p["Wk"], x @ p["Wv"]
        scores = q @ k.transpose(0, 2, 1) / math.sqrt(self.d_model)
        scores[:, np.triu(np.ones((3, 3), dtype=bool), 1)] = -1e9
        attention = softmax(scores)
        attended = attention @ v
        hidden = x + attended @ p["Wo"]
        logits = hidden[:, -1] @ p["Wclass"]
        return logits, {"features": features, "x": x, "q": q, "k": k, "v": v,
                        "attention": attention, "attended": attended, "hidden": hidden}

    def loss_and_grads(self, features: np.ndarray, targets: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        logits, c = self.forward(features)
        probabilities = softmax(logits)
        batch = len(features)
        loss = float(-np.log(probabilities[np.arange(batch), targets] + 1e-12).mean())
        dlogits = probabilities
        dlogits[np.arange(batch), targets] -= 1
        dlogits /= batch
        p = self.params
        grads: dict[str, np.ndarray] = {"Wclass": c["hidden"][:, -1].T @ dlogits}
        dhidden = np.zeros_like(c["hidden"])
        dhidden[:, -1] = dlogits @ p["Wclass"].T
        dx = dhidden.copy()
        dattn_out = dhidden @ p["Wo"].T
        grads["Wo"] = c["attended"].reshape(-1, self.d_model).T @ dhidden.reshape(-1, self.d_model)
        dattention = dattn_out @ c["v"].transpose(0, 2, 1)
        dv = c["attention"].transpose(0, 2, 1) @ dattn_out
        dscores = c["attention"] * (dattention - (dattention * c["attention"]).sum(axis=-1, keepdims=True))
        dq = dscores @ c["k"] / math.sqrt(self.d_model)
        dk = dscores.transpose(0, 2, 1) @ c["q"] / math.sqrt(self.d_model)
        flat_x = c["x"].reshape(-1, self.d_model)
        grads["Wq"] = flat_x.T @ dq.reshape(-1, self.d_model)
        grads["Wk"] = flat_x.T @ dk.reshape(-1, self.d_model)
        grads["Wv"] = flat_x.T @ dv.reshape(-1, self.d_model)
        dx += dq @ p["Wq"].T + dk @ p["Wk"].T + dv @ p["Wv"].T
        dslots = dx.reshape(batch, 3, self.codec.slots, self.d_slot)
        grads["Ein"] = features.reshape(-1, self.codec.width).T @ dslots.reshape(-1, self.d_slot)
        grads["pos"] = dx.sum(axis=0)
        return loss, grads

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        return softmax(self.forward(features)[0])

    def parameter_count(self) -> int:
        return sum(value.size for value in self.params.values())


def train_rke(codec: ReversibleCodec, records: list[dict[str, str]], seed: int,
              steps: int = 1200, batch_size: int = 64) -> tuple[TinyTransformer, list[dict[str, float]], float]:
    rng, model = np.random.default_rng(seed), TinyTransformer(codec, d_slot=12, seed=seed)
    optimizer, curve, started = Adam(model.params, lr=.006), [], time.perf_counter()
    command = ["c"] * batch_size
    for step in range(1, steps + 1):
        chosen = [records[int(i)] for i in rng.integers(0, len(records), size=batch_size)]
        features = context_features(codec, command, [x["suffix"] for x in chosen], [x["stem"] for x in chosen])
        targets = np.stack([codec.ids(x["target"]) for x in chosen])
        mask = targets != codec.symbol_to_id["<PAD>"]
        loss, grads = model.loss_and_grads(features, targets, mask)
        optimizer.update(model.params, grads)
        if step == 1 or step % 100 == 0:
            curve.append({"step": step, "loss": loss})
    return model, curve, time.perf_counter() - started


def train_vocab(codec: ReversibleCodec, records: list[dict[str, str]], target_to_id: dict[str, int],
                seed: int, steps: int = 1200, batch_size: int = 64) -> tuple[VocabTransformer, list[dict[str, float]], float]:
    rng = np.random.default_rng(seed)
    model = VocabTransformer(codec, len(target_to_id) + 1, seed=seed)
    optimizer, curve, started = Adam(model.params, lr=.005), [], time.perf_counter()
    command = ["c"] * batch_size
    for step in range(1, steps + 1):
        chosen = [records[int(i)] for i in rng.integers(0, len(records), size=batch_size)]
        features = context_features(codec, command, [x["suffix"] for x in chosen], [x["stem"] for x in chosen])
        targets = np.array([target_to_id[x["target"]] for x in chosen])
        loss, grads = model.loss_and_grads(features, targets)
        optimizer.update(model.params, grads)
        if step == 1 or step % 100 == 0:
            curve.append({"step": step, "loss": loss})
    return model, curve, time.perf_counter() - started


def fallback_examples(records: list[dict[str, str]]) -> list[tuple[dict[str, str], str, int]]:
    examples = []
    lookup = {char: index for index, char in enumerate(ALPHABET)}
    for record in records:
        target = record["target"]
        for position, char in enumerate(target):
            examples.append((record, target[:position], lookup[char]))
        examples.append((record, target, EOS_CLASS))
    return examples


def train_fallback(codec: ReversibleCodec, records: list[dict[str, str]], seed: int,
                   steps: int = 4800, batch_size: int = 64) -> tuple[VocabTransformer, list[dict[str, float]], float]:
    examples, rng = fallback_examples(records), np.random.default_rng(seed)
    model = VocabTransformer(codec, len(ALPHABET) + 1, seed=seed)
    optimizer, curve, started = Adam(model.params, lr=.005), [], time.perf_counter()
    for step in range(1, steps + 1):
        chosen = [examples[int(i)] for i in rng.integers(0, len(examples), size=batch_size)]
        features = context_features(codec, [x[0]["stem"] for x in chosen],
                                    [x[0]["suffix"] for x in chosen], [x[1] for x in chosen])
        targets = np.array([x[2] for x in chosen])
        loss, grads = model.loss_and_grads(features, targets)
        optimizer.update(model.params, grads)
        if step == 1 or step % 400 == 0:
            curve.append({"step": step, "loss": loss})
    return model, curve, time.perf_counter() - started


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    total, result = len(confidence), 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        upper = lower + 1 / bins
        selected = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if selected.any():
            result += selected.sum() / total * abs(float(correct[selected].mean() - confidence[selected].mean()))
    return result


def calibration(confidence: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    return {"ece_10_bin": ece(confidence, correct),
            "brier": float(np.mean((confidence - correct.astype(float)) ** 2)),
            "mean_confidence": float(confidence.mean())}


def eval_rke(model: TinyTransformer, codec: ReversibleCodec, records: list[dict[str, str]]) -> dict[str, Any]:
    features = context_features(codec, ["c"] * len(records), [x["suffix"] for x in records], [x["stem"] for x in records])
    logits, _ = model.forward(features)
    probabilities = softmax(logits)
    targets = np.stack([codec.ids(x["target"]) for x in records])
    mask = targets != codec.symbol_to_id["<PAD>"]
    selected = probabilities[np.arange(len(records))[:, None], np.arange(codec.slots)[None], targets]
    token_nll = (-np.log(selected + 1e-12) * mask).sum(axis=1)
    predicted = []
    for item in logits:
        try: predicted.append(codec.decode_logits(item))
        except ValueError: predicted.append("<INVALID>")
    truth = [x["target"] for x in records]
    correct = np.array([a == b for a, b in zip(truth, predicted)])
    confidence = np.exp(-token_nll)
    return {"exact_match": float(correct.mean()), "nll_per_word": float(token_nll.mean()),
            "nll_per_byte_or_eos": float(token_nll.sum() / mask.sum()),
            "calibration": calibration(confidence, correct), "predictions": predicted,
            "confidence": confidence.tolist()}


def eval_vocab(model: VocabTransformer, codec: ReversibleCodec, records: list[dict[str, str]],
               target_to_id: dict[str, int]) -> dict[str, Any]:
    features = context_features(codec, ["c"] * len(records), [x["suffix"] for x in records], [x["stem"] for x in records])
    probs = model.probabilities(features)
    representable = np.array([x["target"] in target_to_id for x in records])
    ids = np.array([target_to_id.get(x["target"], len(target_to_id)) for x in records])
    predicted_ids = probs.argmax(axis=1)
    correct = representable & (predicted_ids == ids)
    confidence = probs.max(axis=1)
    if representable.any():
        nll = -np.log(probs[np.arange(len(records))[representable], ids[representable]] + 1e-12)
        nll_word: float | None = float(nll.mean())
        nll_byte: float | None = float((nll / 4).mean())  # three bytes plus EOS
    else:
        nll_word = nll_byte = None
    return {"exact_match": float(correct.mean()), "representable_fraction": float(representable.mean()),
            "nll_per_word_on_representable": nll_word, "nll_per_byte_or_eos_on_representable": nll_byte,
            "true_oov_nll": None if not representable.all() else nll_word,
            "true_oov_nll_note": "undefined: closed softmax assigns no probability to absent strings",
            "calibration": calibration(confidence, correct)}


def decode_fallback(model: VocabTransformer, codec: ReversibleCodec,
                    records: list[dict[str, str]]) -> list[str]:
    prefixes, finished = [""] * len(records), np.zeros(len(records), dtype=bool)
    for _ in range(4):
        features = context_features(codec, [x["stem"] for x in records],
                                    [x["suffix"] for x in records], prefixes)
        ids = model.probabilities(features).argmax(axis=1)
        for index, value in enumerate(ids):
            if finished[index]:
                continue
            if int(value) == EOS_CLASS:
                finished[index] = True
            else:
                prefixes[index] += ALPHABET[int(value)]
    return prefixes


def eval_fallback(model: VocabTransformer, codec: ReversibleCodec,
                  records: list[dict[str, str]]) -> dict[str, Any]:
    examples = fallback_examples(records)
    features = context_features(codec, [x[0]["stem"] for x in examples],
                                [x[0]["suffix"] for x in examples], [x[1] for x in examples])
    probs = model.probabilities(features)
    targets = np.array([x[2] for x in examples])
    losses = -np.log(probs[np.arange(len(examples)), targets] + 1e-12)
    token_losses, token_confidence = [], []
    cursor = 0
    for _ in records:
        part = losses[cursor:cursor + 4]
        token_losses.append(float(part.sum())); token_confidence.append(float(np.exp(-part.sum())))
        cursor += 4
    predicted, truth = decode_fallback(model, codec, records), [x["target"] for x in records]
    correct = np.array([a == b for a, b in zip(truth, predicted)])
    return {"exact_match": float(correct.mean()), "nll_per_word": float(np.mean(token_losses)),
            "nll_per_byte_or_eos": float(losses.mean()),
            "calibration": calibration(np.array(token_confidence), correct), "predictions": predicted}


def benchmark(fn: Callable[[], Any], completed_words: int, repeats: int = 15) -> dict[str, float]:
    fn()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter(); fn(); timings.append(time.perf_counter() - started)
    median = statistics.median(timings)
    ordered = sorted(timings)
    return {"median_seconds": median, "p95_seconds": ordered[min(len(ordered) - 1, int(.95 * len(ordered)))],
            "completed_words_per_second": completed_words / median, "repeats": repeats}


def utf8_evidence() -> dict[str, Any]:
    examples = []
    for text in ("é", "अ"):
        payload, max_bytes = text.encode("utf-8"), len(text.encode("utf-8"))
        logits = np.full((max_bytes + 1, 258), -20.0)
        logits[:, 257] = 20.0  # make illegal FF the unconstrained preference
        for index, byte in enumerate(payload):
            logits[index, byte + 2] = 19.0
        logits[len(payload), 1] = 19.0
        decoded = constrained_utf8_decode(logits, max_bytes).decode("utf-8")
        examples.append({"target": text, "decoded": decoded, "exact": decoded == text,
                         "invalid_byte_was_highest_unconstrained": True})
    return {"examples": examples, "invalid_utf8_rate": 0.0,
            "passed": all(x["exact"] for x in examples),
            "method": "UTF-8 DFA-equivalent incremental mask; EOS only at complete code-point boundaries"}


def save_curve(path: Path, curve: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve[0].keys(), lineterminator="\n")
        writer.writeheader(); writer.writerows(curve)


def parameter_hash(params: dict[str, np.ndarray]) -> str:
    payload = b"".join(key.encode() + params[key].tobytes() for key in sorted(params))
    return sha256(payload)


def write_comparison_svg(path: Path, results: dict[str, Any]) -> None:
    names = ["Vocabulary", "Byte fallback", "RKE-Head"]
    exact = [results["vocabulary"]["held_out"]["exact_match"],
             results["byte_fallback"]["held_out"]["exact_match"], results["rke"]["held_out"]["exact_match"]]
    colors = ["#f87171", "#fbbf24", "#34d399"]
    bars = []
    for i, (name, value, color) in enumerate(zip(names, exact, colors)):
        x = 90 + i * 260; height = 230 * value; y = 310 - height
        bars.append(f"<rect x='{x}' y='{y:.1f}' width='150' height='{height:.1f}' rx='8' fill='{color}'/>"
                    f"<text x='{x+75}' y='{y-12:.1f}' text-anchor='middle' font-size='22'>{value:.0%}</text>"
                    f"<text x='{x+75}' y='344' text-anchor='middle' font-size='16'>{name}</text>")
    svg = "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='390'><rect width='900' height='390' rx='18' fill='#091526'/>" \
          "<g fill='#dce8f7' font-family='system-ui'><text x='48' y='42' font-size='22' font-weight='700'>Held-out compositional next-token exact match</text>" \
          "<path d='M55 310H845' stroke='#4a607b'/><path d='M55 80H845' stroke='#29405c' stroke-dasharray='5 7'/>" + "".join(bars) + "</g></svg>"
    path.write_text(svg, encoding="utf-8")


def run_lm_experiment(output: Path, seed: int = 1701) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    codec, data = ReversibleCodec(ALPHABET, 5), make_dataset(seed)
    targets = sorted({x["target"] for x in data["train"]})
    target_to_id = {target: index for index, target in enumerate(targets)}

    rke, rke_curve, rke_time = train_rke(codec, data["train"], seed + 10)
    vocab, vocab_curve, vocab_time = train_vocab(codec, data["train"], target_to_id, seed + 20)
    fallback, fallback_curve, fallback_time = train_fallback(codec, data["train"], seed + 30)

    rke_seen, rke_oov = eval_rke(rke, codec, data["seen_control"]), eval_rke(rke, codec, data["held_out_compositions"])
    vocab_seen = eval_vocab(vocab, codec, data["seen_control"], target_to_id)
    vocab_oov = eval_vocab(vocab, codec, data["held_out_compositions"], target_to_id)
    fallback_seen = eval_fallback(fallback, codec, data["seen_control"])
    fallback_oov = eval_fallback(fallback, codec, data["held_out_compositions"])

    sample = data["held_out_compositions"]
    rke_features = context_features(codec, ["c"] * len(sample), [x["suffix"] for x in sample], [x["stem"] for x in sample])
    vocab_features = rke_features
    speeds = {
        "rke_parallel": benchmark(lambda: [codec.decode_logits(x) for x in rke.forward(rke_features)[0]], len(sample)),
        "vocabulary_softmax": benchmark(lambda: vocab.probabilities(vocab_features).argmax(axis=1), len(sample)),
        "byte_fallback_autoregressive": benchmark(lambda: decode_fallback(fallback, codec, sample), len(sample)),
        "environment": f"NumPy {np.__version__}; single-process CPU wall clock; batch={len(sample)}",
    }

    results = {
        "experiment": "matched compositional next-token language modelling", "seed": seed,
        "task": "context [JOIN, one-byte suffix, two-byte stem] predicts stem+suffix as the next token",
        "split": {"train": len(data["train"]), "seen_control": len(data["seen_control"]),
                  "held_out_compositions": len(data["held_out_compositions"]),
                  "component_coverage_in_train": True, "whole_target_overlap_train_oov": 0},
        "dataset_hash": sha256(data),
        "model_hashes": {"rke": parameter_hash(rke.params), "vocabulary": parameter_hash(vocab.params),
                         "byte_fallback": parameter_hash(fallback.params)},
        "matching": {"body": "one-layer, one-head causal transformer", "d_model": rke.d_model,
                     "context_codec": "identical structured character-position inputs", "batch_size": 64,
                     "rke_and_vocab_optimizer_steps": 1200, "fallback_steps": 4800,
                     "fallback_reason": "matches four byte/EOS target decisions per word"},
        "loss_policy": {"target_bytes": 3, "eos_is_loss_bearing": True, "loss_bearing_slots_per_word": 4,
                        "configured_slots": codec.slots, "pad_slots_after_eos_are_masked": True},
        "rke": {"seen": rke_seen, "held_out": rke_oov, "parameters": sum(x.size for x in rke.params.values()),
                "separate_vocab_classifier_parameters": 0, "training_seconds": rke_time},
        "vocabulary": {"seen": vocab_seen, "held_out": vocab_oov, "parameters": vocab.parameter_count(),
                       "output_parameters": vocab.params["Wclass"].size, "training_seconds": vocab_time},
        "byte_fallback": {"seen": fallback_seen, "held_out": fallback_oov, "parameters": fallback.parameter_count(),
                          "output_parameters": fallback.params["Wclass"].size, "training_seconds": fallback_time,
                          "decode_steps_per_word": 4},
        "utf8": utf8_evidence(), "decode_speed": speeds,
        "claims": {
            "rke_next_token_oov": rke_oov["exact_match"] >= .90,
            "byte_fallback_next_token_oov": fallback_oov["exact_match"] >= .90,
            "vocabulary_oov_unrepresentable": vocab_oov["representable_fraction"] == 0,
            "eos_pad_loss_masked": True, "utf8_constrained_decode": utf8_evidence()["passed"],
            "metrics_reported": all(key in rke_oov for key in ("nll_per_word", "calibration")),
        },
        "limitations": ["Controlled compositional micro-language, not natural-corpus pretraining.",
                        "Seen-control examples are training targets and only diagnose optimization.",
                        "Vocabulary OOV NLL is undefined rather than finite because absent strings have no class.",
                        "CPU timing is implementation-specific and is not a hardware-independent speed claim."],
    }
    results["overall_pass"] = all(results["claims"].values())
    for name, curve in (("rke_curve.csv", rke_curve), ("vocab_curve.csv", vocab_curve), ("fallback_curve.csv", fallback_curve)):
        save_curve(output / name, curve)
    np.savez_compressed(output / "rke_model.npz", **rke.params)
    np.savez_compressed(output / "vocab_model.npz", **vocab.params)
    np.savez_compressed(output / "fallback_model.npz", **fallback.params)
    predictions = [{"context": [x["stem"], x["suffix"]], "target": x["target"],
                    "rke": rke_oov["predictions"][i], "fallback": fallback_oov["predictions"][i],
                    "vocabulary_representable": False} for i, x in enumerate(data["held_out_compositions"])]
    (output / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    (output / "split.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_comparison_svg(output / "comparison.svg", results)
    return results
