"""Exact token-distribution fine-tuning without learned vocabulary rows.

Each vocabulary logit is composed from its immutable byte/EOS K-code. The
normalizer covers the complete tokenizer vocabulary in chunks, so this is exact
cross-entropy rather than sampled-negative training.
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


def _code_scores(slot_log_probs, codes):
    scores = torch.zeros((slot_log_probs.shape[0], codes.shape[0]), device=slot_log_probs.device)
    for position in range(slot_log_probs.shape[1]):
        state = codes[:, position]
        scores += slot_log_probs[:, position, :][:, state] * (state != 0)[None, :]
    return scores


def run(steps=1000, batch_size=8, token_weight=0.25, lr=5e-5, vocab_chunk=8192):
    torch.manual_seed(3180)
    torch.set_num_threads(1)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "artifacts/qwen_data_manifest/results.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = TinyKModel(tokenizer, max_len=64).to(device)
    source = root / "artifacts/tiny_exact_kcode_model/model.pt"
    model.load_state_dict(torch.load(source, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    vocabulary_codes = model.k.state_table[: tokenizer.vocab_size]

    texts = [row["target"] for row in manifest["rows"]["train"]]
    batches = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size], add_special_tokens=True,
            truncation=True, max_length=64, padding=True, return_tensors="pt"
        )
        batches.append((encoded["input_ids"].to(device), encoded["attention_mask"].to(device)))

    total_history = []
    structured_history = []
    token_history = []
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        ids, attention = batches[step % len(batches)]
        hidden = model(ids, attention)[:, :-1]
        labels = ids[:, 1:]
        token_mask = attention[:, 1:].bool()
        states = model.k.state_table[labels]
        logits = torch.matmul(hidden, model.k.projection.t()).view(
            *hidden.shape[:2], model.k.positions, model.k.states
        )
        state_mask = (states != 0) & token_mask[..., None]
        state_losses = F.cross_entropy(
            logits.reshape(-1, model.k.states), states.reshape(-1), reduction="none"
        ).view_as(states)
        structured_loss = state_losses.masked_select(state_mask).mean()

        # Cover different causal positions deterministically while keeping the
        # exact full-vocabulary normalizer affordable on laptop hardware.
        selected_hidden = []
        selected_targets = []
        for batch_index in range(ids.shape[0]):
            count = int(token_mask[batch_index].sum())
            if count:
                position = (step + 7 * batch_index) % count
                selected_hidden.append(logits[batch_index, position])
                selected_targets.append(labels[batch_index, position])
        selected_logits = torch.stack(selected_hidden)
        selected_targets = torch.stack(selected_targets)
        slot_log_probs = F.log_softmax(selected_logits.float(), dim=-1)
        true_codes = model.k.state_table[selected_targets]
        true_scores = _code_scores(slot_log_probs, true_codes).diagonal()
        log_normalizer = None
        for offset in range(0, tokenizer.vocab_size, vocab_chunk):
            scores = _code_scores(slot_log_probs, vocabulary_codes[offset : offset + vocab_chunk])
            chunk_normalizer = torch.logsumexp(scores, dim=-1)
            log_normalizer = chunk_normalizer if log_normalizer is None else torch.logaddexp(log_normalizer, chunk_normalizer)
        token_loss = (log_normalizer - true_scores).mean()

        loss = structured_loss + token_weight * token_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_history.append(float(loss.detach()))
        structured_history.append(float(structured_loss.detach()))
        token_history.append(float(token_loss.detach()))

    output = root / "artifacts/tiny_kcode_exact_token_finetune"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    result = {
        "status": "exact_full_vocabulary_kcode_finetune",
        "device": str(device),
        "steps": steps,
        "batch_size": batch_size,
        "positions_per_sequence_per_step": 1,
        "vocabulary_candidates": tokenizer.vocab_size,
        "vocabulary_chunk": vocab_chunk,
        "token_weight": token_weight,
        "learning_rate": lr,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "added_vocabulary_parameters": 0,
        "loss_first_last": [total_history[0], total_history[-1]],
        "structured_loss_first_last": [structured_history[0], structured_history[-1]],
        "exact_token_nll_first_last": [token_history[0], token_history[-1]],
        "seconds": time.perf_counter() - started,
        "dataset_hash": manifest["dataset_hash"],
        "source_checkpoint": str(source.relative_to(root)),
        "checkpoint": str(checkpoint.relative_to(root)),
    }
    (output / "training_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
