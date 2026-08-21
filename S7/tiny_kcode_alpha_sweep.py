"""Single-pass validation sweep for fixed-code decoder length calibration."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from qwen_kcode_input import MODEL
from tiny_exact_kcode_model import TinyKModel


def run(limit=500, alphas=(-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)):
    torch.set_num_threads(1)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "artifacts/qwen_data_manifest/results.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = TinyKModel(tokenizer, max_len=64).to(device)
    checkpoint = root / "artifacts/tiny_exact_kcode_model/model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    candidates = model.k.state_table[: tokenizer.vocab_size].to(device)
    lengths = (candidates != 0).sum(-1).clamp_min(1).float()
    rows = manifest["rows"]["validation"][:limit]
    correct = {alpha: 0 for alpha in alphas}
    total = 0
    started = time.perf_counter()

    with torch.no_grad():
        for start in range(0, len(rows), 8):
            encoded = tokenizer(
                [row["target"] for row in rows[start : start + 8]],
                add_special_tokens=True, truncation=True, max_length=64,
                padding=True, return_tensors="pt"
            )
            ids = encoded["input_ids"].to(device)
            attention = encoded["attention_mask"].to(device)
            hidden = model(ids, attention)[:, :-1]
            logits = torch.matmul(hidden, model.k.projection.t()).view(
                *hidden.shape[:2], model.k.positions, model.k.states
            )
            flat = F.log_softmax(logits.float(), -1)[attention[:, 1:].bool()]
            targets = ids[:, 1:][attention[:, 1:].bool()]
            best_scores = {alpha: torch.full((flat.shape[0],), -torch.inf, device=device) for alpha in alphas}
            best_ids = {alpha: torch.zeros(flat.shape[0], dtype=torch.long, device=device) for alpha in alphas}
            for offset in range(0, candidates.shape[0], 2048):
                codes = candidates[offset : offset + 2048]
                scores = torch.zeros((flat.shape[0], codes.shape[0]), device=device)
                for position in range(model.k.positions):
                    state = codes[:, position]
                    scores += flat[:, position, :][:, state] * (state != 0)[None, :]
                chunk_lengths = lengths[offset : offset + codes.shape[0]][None, :]
                for alpha in alphas:
                    values, indices = (scores / (chunk_lengths ** alpha)).max(-1)
                    improve = values > best_scores[alpha]
                    best_scores[alpha] = torch.where(improve, values, best_scores[alpha])
                    best_ids[alpha] = torch.where(improve, indices + offset, best_ids[alpha])
            for alpha in alphas:
                correct[alpha] += int((best_ids[alpha] == targets).sum())
            total += int(targets.numel())

    metrics = [
        {"alpha": alpha, "correct": correct[alpha], "token_accuracy": correct[alpha] / total}
        for alpha in alphas
    ]
    selected = max(metrics, key=lambda row: (row["correct"], -abs(row["alpha"])))
    result = {
        "status": "validation_only_length_calibration",
        "device": str(device),
        "split": "validation",
        "examples": len(rows),
        "tokens": total,
        "metrics": metrics,
        "selected_alpha": selected["alpha"],
        "selected_token_accuracy": selected["token_accuracy"],
        "trainable_parameters_added": 0,
        "seconds": time.perf_counter() - started,
        "dataset_hash": manifest["dataset_hash"],
    }
    output = root / "artifacts/tiny_kcode_vocab_decode/alpha_sweep_validation.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
