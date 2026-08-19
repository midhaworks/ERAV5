"""Neural batching proof for dynamic RKE continuation blocks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rke import ContinuationByteCodec, TinyTransformer, sha256


def make_payloads(seed: int = 4401, train_size: int = 500,
                  test_size: int = 150) -> tuple[list[bytes], list[bytes]]:
    rng = np.random.default_rng(seed)
    values: dict[str, bytes] = {}
    # Explicit boundaries guarantee that EOS and CONT behavior is exercised.
    lengths = [0, 1, 23, 24, 25, 47, 48, 49, 72, 95, 96]
    attempt = 0
    while len(values) < train_size + test_size:
        length = lengths[attempt % len(lengths)]
        attempt += 1
        payload = rng.integers(0, 256, size=length, dtype=np.uint8).tobytes()
        values[sha256(payload)] = payload
    ordered = [values[key] for key in sorted(values)]
    return ordered[:train_size], ordered[train_size:train_size + test_size]


def flatten_blocks(codec: ContinuationByteCodec, payloads: list[bytes]) -> list[tuple[int, int, np.ndarray]]:
    output = []
    for payload_index, payload in enumerate(payloads):
        for block_index, states in enumerate(codec.ids(payload)):
            output.append((payload_index, block_index, states))
    return output


def one_hot(codec: ContinuationByteCodec, states: np.ndarray) -> np.ndarray:
    matrix = np.zeros((codec.slots, codec.width), dtype=np.float64)
    matrix[np.arange(codec.slots), states] = 1.0
    return matrix


def block_features(codec: ContinuationByteCodec, states: list[np.ndarray]) -> np.ndarray:
    command = one_hot(codec, codec.ids(b"")[0])
    pad = np.zeros(codec.slots, dtype=np.int64); pad[0] = 1
    distractor = one_hot(codec, pad)
    return np.stack([np.stack([command, distractor, one_hot(codec, item)]) for item in states])


def evaluate(model: TinyTransformer, codec: ContinuationByteCodec,
             payloads: list[bytes]) -> dict[str, Any]:
    flattened = flatten_blocks(codec, payloads)
    states = [item[2] for item in flattened]
    logits, _ = model.forward(block_features(codec, states))
    predictions = logits.argmax(axis=-1)
    by_payload: list[list[np.ndarray]] = [[] for _ in payloads]
    for (payload_index, _, _), predicted in zip(flattened, predictions):
        by_payload[payload_index].append(predicted)
    recovered, valid = [], []
    for blocks in by_payload:
        try:
            recovered.append(codec.decode_ids(blocks)); valid.append(True)
        except ValueError:
            recovered.append(b""); valid.append(False)
    exact = [ok and target == result for target, result, ok in zip(payloads, recovered, valid)]
    targets = np.stack(states); mask = targets != 0
    probabilities = model._softmax(logits)
    selected = probabilities[np.arange(len(targets))[:, None], np.arange(codec.slots)[None, :], targets]
    nll = float((-np.log(selected + 1e-12) * mask).sum() / mask.sum())
    return {"payloads": len(payloads), "blocks": len(flattened), "bytes": sum(map(len, payloads)),
            "max_payload_bytes": max(map(len, payloads)), "exact_match": float(np.mean(exact)),
            "valid_block_chain_rate": float(np.mean(valid)), "nll_per_byte_or_terminator": nll,
            "loss_bearing_cont_states": int(sum(np.count_nonzero(x == 2) for x in states)),
            "loss_bearing_eos_states": int(sum(np.count_nonzero(x == 1) for x in states)),
            "loss_bearing_byte_states": int(sum(np.count_nonzero(x >= 3) for x in states))}


def run_continuation_neural(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    codec = ContinuationByteCodec(24)
    train_payloads, test_payloads = make_payloads()
    train_blocks = flatten_blocks(codec, train_payloads)
    model = TinyTransformer(codec, d_slot=2, seed=4401)
    # Construct 259 distinct equal-norm prototypes on the unit circle. Their
    # self-dot product is strictly greater than every cross-dot product, so the
    # tied Ein/Ein.T path exactly recovers every state. Zero attention leaves
    # the final input block as the output latent without a vocabulary head.
    angles = 2 * np.pi * np.arange(codec.width) / codec.width
    model.params["Ein"][:] = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 20.0
    for name in ("Wq", "Wk", "Wv", "Wo", "pos"):
        model.params[name].fill(0.0)
    train_metrics, test_metrics = evaluate(model, codec, train_payloads), evaluate(model, codec, test_payloads)

    # Prove post-terminator PAD is excluded from both loss and gradient.
    sample = np.stack([train_blocks[0][2]])
    sample_mask = sample != 0
    changed = sample.copy(); changed[~sample_mask] = 3
    sample_features = block_features(codec, [sample[0]])
    loss_a, grads_a = model.loss_and_grads(sample_features, sample, sample_mask)
    loss_b, grads_b = model.loss_and_grads(sample_features, changed, sample_mask)
    mask_invariant = abs(loss_a - loss_b) < 1e-12 and all(
        np.allclose(grads_a[key], grads_b[key], atol=1e-12) for key in grads_a)
    result = {
        "experiment": "neural continuation-block integration proof", "seed": 4401,
        "codec": {"block_bytes": 24, "states": 259, "terminators": ["EOS", "CONT"]},
        "split": {"train_payloads": len(train_payloads), "test_payloads": len(test_payloads),
                  "overlap": len(set(train_payloads) & set(test_payloads)),
                  "hash": sha256({"train": [x.hex() for x in train_payloads],
                                  "test": [x.hex() for x in test_payloads]})},
        "construction": {"kind": "equal-norm circular tied prototypes", "optimizer_steps": 0,
                         "purpose": "mechanical neural batching/decoding oracle"},
        "train": train_metrics, "test": test_metrics,
        "checks": {"split_disjoint": not (set(train_payloads) & set(test_payloads)),
                   "multi_block_tested": test_metrics["max_payload_bytes"] > 24,
                   "cont_is_loss_bearing": test_metrics["loss_bearing_cont_states"] > 0,
                   "eos_is_loss_bearing": test_metrics["loss_bearing_eos_states"] == len(test_payloads),
                   "post_terminator_pad_masked": mask_invariant,
                   "held_out_exact_at_least_95_percent": test_metrics["exact_match"] >= .95,
                   "valid_chain_rate_100_percent": test_metrics["valid_block_chain_rate"] == 1.0},
        "limitations": ["This is a constructed neural block-copy oracle, not learned long-word language modelling.",
                        "Blocks are independently reconstructed; learned cross-block conditioning is future work."],
    }
    result["passed"] = all(result["checks"].values())
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(output / "model.npz", **model.params)
    return result


if __name__ == "__main__":
    destination = Path(__file__).resolve().parent / "artifacts" / "continuation_neural"
    value = run_continuation_neural(destination)
    print(json.dumps(value, indent=2))
    if not value["passed"]:
        raise SystemExit(1)
