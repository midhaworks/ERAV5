"""Candidate-aware fine-tuning for the small exact K-code transformer.

The auxiliary sampled-softmax objective trains the score used by the fixed-code
decoder. Candidate token codes are immutable and introduce no learned vocab rows.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from qwen_kcode_input import MODEL
from tiny_exact_kcode_model import TinyKModel


def run(steps=2000, batch_size=8, random_candidates=128, rank_weight=0.25, lr=1e-4):
    torch.manual_seed(3180)
    torch.set_num_threads(1)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "artifacts/qwen_data_manifest/results.json").read_text())
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = TinyKModel(tok, max_len=64).to(device)
    source = root / "artifacts/tiny_exact_kcode_model/model.pt"
    model.load_state_dict(torch.load(source, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    texts = [row["target"] for row in manifest["rows"]["train"]]
    batches = []
    for start in range(0, len(texts), batch_size):
        encoded = tok(
            texts[start : start + batch_size], add_special_tokens=True,
            truncation=True, max_length=64, padding=True, return_tensors="pt"
        )
        batches.append((encoded["input_ids"].to(device), encoded["attention_mask"].to(device)))

    losses = []
    structured_losses = []
    rank_losses = []
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        ids, attention = batches[step % len(batches)]
        hidden = model(ids, attention)[:, :-1]
        labels = ids[:, 1:]
        active_tokens = attention[:, 1:].bool()
        states = model.k.state_table[labels]
        logits = torch.matmul(hidden, model.k.projection.t()).view(
            *hidden.shape[:2], model.k.positions, model.k.states
        )

        active_states = (states != 0) & active_tokens[..., None]
        per_state = F.cross_entropy(
            logits.reshape(-1, model.k.states), states.reshape(-1), reduction="none"
        ).view_as(states)
        structured_loss = per_state.masked_select(active_states).mean()

        flat_lp = F.log_softmax(logits[active_tokens].float(), dim=-1)
        targets = labels[active_tokens]
        sampled = torch.randint(tok.vocab_size, (random_candidates,), device=device)
        candidate_ids = torch.unique(torch.cat((targets, sampled)), sorted=True)
        candidate_codes = model.k.state_table[candidate_ids]
        candidate_active = candidate_codes != 0
        scores = torch.zeros((targets.numel(), candidate_ids.numel()), device=device)
        for position in range(model.k.positions):
            state = candidate_codes[:, position]
            scores += flat_lp[:, position, :][:, state] * candidate_active[:, position][None, :]
        target_columns = torch.searchsorted(candidate_ids, targets)
        rank_loss = F.cross_entropy(scores, target_columns)

        loss = structured_loss + rank_weight * rank_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        structured_losses.append(float(structured_loss.detach()))
        rank_losses.append(float(rank_loss.detach()))

    output = root / "artifacts/tiny_kcode_candidate_finetune"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    result = {
        "status": "candidate_aware_kcode_finetune",
        "device": str(device),
        "steps": steps,
        "batch_size": batch_size,
        "random_candidates": random_candidates,
        "rank_weight": rank_weight,
        "learning_rate": lr,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "added_vocabulary_parameters": 0,
        "loss_first_last": [losses[0], losses[-1]],
        "structured_loss_first_last": [structured_losses[0], structured_losses[-1]],
        "rank_loss_first_last": [rank_losses[0], rank_losses[-1]],
        "seconds": time.perf_counter() - started,
        "dataset_hash": manifest["dataset_hash"],
        "source_checkpoint": str(source.relative_to(root)),
        "checkpoint": str(checkpoint.relative_to(root)),
    }
    (output / "training_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
