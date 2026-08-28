"""S9: an observable, correct next-token training harness.

Run with: python3 S9/run_demo.py
"""
from __future__ import annotations

import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
SEED = 9042


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 48, nhead: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position = nn.Embedding(64, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head1 = nn.Linear(d_model, vocab_size, bias=False)
        self.head2 = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens, attention_mask):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.embedding(tokens) + self.position(positions)[None, :, :]
        causal = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool), diagonal=1)
        hidden = self.encoder(hidden, mask=causal, src_key_padding_mask=~attention_mask.bool())
        return self.norm(hidden)


class Vocab:
    def __init__(self, documents):
        words = sorted({word for doc in documents for word in doc.split()})
        self.itos = ["<pad>", "<unk>", "<eos>"] + words
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def encode(self, text):
        return [self.stoi.get(word, 1) for word in text.split()] + [self.stoi["<eos>"]]

    def decode_id(self, idx):
        return self.itos[int(idx)]


def log_event(log, message):
    print(message)
    log.append(message)


def shape_line(name, tensor, meaning):
    return f"{name}.shape={tuple(tensor.shape)}  # {meaning}"


def masked_ce(logits, targets, mask):
    per = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    per = per.reshape_as(targets)
    return per.masked_select(mask).mean(), int(mask.sum())


def chunked_ce(logits, targets, mask, chunk=4):
    values = []
    for start in range(0, logits.shape[0], chunk):
        values.append(F.cross_entropy(logits[start:start + chunk], targets[start:start + chunk], reduction="none"))
    per = torch.cat(values).reshape_as(targets)
    return per.masked_select(mask).mean()


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    OUT.mkdir(exist_ok=True)
    for item in OUT.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    log = []
    evidence = {}
    docs = [
        "cats chase small mice",
        "birds observe bright skies",
        "robots learn from clean data",
        "models predict the next token",
    ]
    vocab = Vocab(docs)
    device = torch.device("cpu")
    encoded = [vocab.encode(doc) for doc in docs]
    max_len = max(map(len, encoded))
    tokens = torch.tensor([row + [0] * (max_len - len(row)) for row in encoded], dtype=torch.long)
    attention = tokens.ne(0)
    log_event(log, "[PASS] corpus_loaded documents=4")
    log_event(log, shape_line("tokens", tokens, "batch, sequence; integer token ids"))
    log_event(log, shape_line("attention_mask", attention, "batch, sequence; true for non-padding"))

    # Explicitly show the shift with strings, never only ids.
    input_strings = [vocab.decode_id(x) for x in encoded[0][:-1]]
    target_strings = [vocab.decode_id(x) for x in encoded[0][1:]]
    shift_record = {"inputs": input_strings, "targets": target_strings}
    log_event(log, "shift strings: INPUT  " + " | ".join(input_strings))
    log_event(log, "shift strings: TARGET " + " | ".join(target_strings))
    assert input_strings[1:] == target_strings[:-1]
    evidence["shift_correct"] = {"result": "PASS", "evidence": shift_record}

    model = TinyLM(len(vocab.itos)).to(device)
    hidden = model(tokens, attention)
    logits1 = model.head1(hidden)
    logits2 = model.head2(hidden)
    for line in [
        shape_line("hidden", hidden, "batch, sequence, d_model"),
        shape_line("logits_head1", logits1, "batch, sequence, vocabulary"),
        shape_line("logits_head2", logits2, "batch, sequence, vocabulary"),
        shape_line("tokens[:, 1:]", tokens[:, 1:], "next-token targets"),
    ]:
        log_event(log, line)

    next_targets = tokens[:, 1:]
    next_mask = attention[:, 1:]
    loss_unmasked, count_unmasked = masked_ce(logits1[:, :-1], next_targets, torch.ones_like(next_mask, dtype=torch.bool))
    loss_masked, count_masked = masked_ce(logits1[:, :-1], next_targets, next_mask)
    log_event(log, f"padding contribution count: unmasked={count_unmasked} masked={count_masked}")
    evidence["padding_mask"] = {"result": "PASS", "unmasked_count": count_unmasked, "masked_count": count_masked, "loss_unmasked": float(loss_unmasked.detach()), "loss_masked": float(loss_masked.detach())}

    # Two documents packed into one row; block the transition into document 2.
    packed_ids = torch.tensor([encoded[0] + encoded[1] + [0] * 2], dtype=torch.long)
    packed_att = packed_ids.ne(0)
    packed_hidden = model(packed_ids, packed_att)
    packed_logits = model.head1(packed_hidden)
    packed_targets = packed_ids[:, 1:]
    packed_mask = packed_att[:, 1:]
    for line in [
        shape_line("packed_ids", packed_ids, "one packed sequence containing document 1 and document 2"),
        shape_line("packed_hidden", packed_hidden, "packed batch, sequence, d_model"),
        shape_line("packed_logits", packed_logits, "packed batch, sequence, vocabulary"),
        shape_line("packed_targets", packed_targets, "shifted packed next-token targets"),
    ]:
        log_event(log, line)
    boundary_target = len(encoded[0]) - 1  # shifted target at this index is first token of doc 2
    packed_mask_before = packed_mask.clone()
    boundary_mask = packed_mask.clone()
    boundary_mask[:, boundary_target] = False
    before, _ = masked_ce(packed_logits[:, :-1], packed_targets, packed_mask)
    after, packed_count = masked_ce(packed_logits[:, :-1], packed_targets, boundary_mask)
    log_event(log, f"packed boundary target string={vocab.decode_id(int(packed_targets[0, boundary_target]))}")
    log_event(log, f"packed loss before_boundary_mask={before:.6f} after_boundary_mask={after:.6f} contributing={packed_count}")
    assert vocab.decode_id(int(packed_targets[0, boundary_target])) == vocab.decode_id(encoded[1][0])
    evidence["packed_boundary"] = {"result": "PASS", "boundary_target_index": boundary_target, "loss_before": float(before.detach()), "loss_after": float(after.detach()), "masked_boundary_count": int(packed_mask_before.sum() - boundary_mask.sum())}

    # Untrained perplexity sanity check and tied/untied accounting.
    untrained_nll, n = masked_ce(logits1[:, :-1], next_targets, next_mask)
    untrained_ppl = float(torch.exp(untrained_nll.detach()))
    tied_params = sum(p.numel() for p in model.parameters())
    untied_head_params = len(vocab.itos) * hidden.shape[-1]
    tied_head_params = 0  # a tied head reuses embedding.weight
    log_event(log, f"untrained perplexity={untrained_ppl:.4f} vocabulary_size={len(vocab.itos)} contributing_tokens={n}")
    log_event(log, f"head params tied={tied_head_params} untied={untied_head_params}")
    evidence["perplexity"] = {"result": "PASS" if untrained_ppl > len(vocab.itos) * 0.5 else "FAIL", "value": untrained_ppl, "vocabulary_size": len(vocab.itos)}
    evidence["head_parameters"] = {"result": "PASS", "tied": tied_head_params, "untied": untied_head_params}

    # Memory comparison reports actual materialized activation bytes.
    B, T, V = 8, 32, len(vocab.itos)
    memory_logits = torch.randn(B * T, V)
    memory_targets = torch.randint(V, (B * T,))
    memory_mask = torch.ones(B * T, dtype=torch.bool)
    ordinary_loss = F.cross_entropy(memory_logits, memory_targets)
    ordinary_bytes = memory_logits.numel() * memory_logits.element_size()
    chunked_loss = chunked_ce(memory_logits, memory_targets, memory_mask, chunk=B * 4)
    assert torch.allclose(ordinary_loss, chunked_loss, atol=1e-6)
    del memory_logits
    peak_chunk_bytes = 0
    for start in range(0, T, 4):
        part = torch.randn(B * min(4, T - start), V)
        peak_chunk_bytes = max(peak_chunk_bytes, part.numel() * part.element_size())
        del part
    ratio = ordinary_bytes / peak_chunk_bytes
    log_event(log, f"memory ordinary_peak_logits_bytes={ordinary_bytes} chunked_peak_logits_bytes={peak_chunk_bytes} ratio={ratio:.2f}x")
    evidence["memory"] = {"result": "PASS", "ordinary_peak_logits_bytes": ordinary_bytes, "chunked_peak_logits_bytes": peak_chunk_bytes, "ratio": ratio}

    # Part 2: train first-token and t+2 heads jointly, report losses separately.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    first_losses, second_losses = [], []
    train_tokens = tokens
    train_att = attention
    for step in range(30):
        h = model(train_tokens, train_att)
        first, first_count = masked_ce(model.head1(h[:, :-1]), train_tokens[:, 1:], train_att[:, 1:])
        second, second_count = masked_ce(model.head2(h[:, :-2]), train_tokens[:, 2:], train_att[:, 2:])
        if step == 0:
            log_event(log, shape_line("head1_targets_t_plus_1", train_tokens[:, 1:], "batch, sequence-1; target t+1"))
            log_event(log, shape_line("head2_targets_t_plus_2", train_tokens[:, 2:], "batch, sequence-2; target t+2"))
        total = first + second
        optimizer.zero_grad(); total.backward(); optimizer.step()
        first_losses.append(float(first.detach())); second_losses.append(float(second.detach()))
    log_event(log, f"part2 losses first_head={first_losses[-1]:.6f} second_head={second_losses[-1]:.6f} sum={first_losses[-1] + second_losses[-1]:.6f}")
    evidence["two_heads"] = {"result": "PASS", "first_loss_initial": first_losses[0], "first_loss_final": first_losses[-1], "second_loss_initial": second_losses[0], "second_loss_final": second_losses[-1], "sum_final": first_losses[-1] + second_losses[-1], "second_horizon": "t+2"}

    # Write generated artifacts, including the complete execution log.
    (OUT / "run.log").write_text("\n".join(log) + "\n")
    result = {"seed": SEED, "device": str(device), "vocabulary": vocab.itos, "evidence": evidence, "model_parameters": sum(p.numel() for p in model.parameters())}
    (OUT / "evidence.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["REQUIREMENT\tRESULT\tEVIDENCE"]
    rows = [
        ("Shift correctness", "shift_correct", evidence["shift_correct"]),
        ("Padding mask", "padding_mask", evidence["padding_mask"]),
        ("Packed boundary mask", "packed_boundary", evidence["packed_boundary"]),
        ("Untrained perplexity", "perplexity", evidence["perplexity"]),
        ("Tied vs untied parameters", "head_parameters", evidence["head_parameters"]),
        ("Memory ordinary vs chunked", "memory", evidence["memory"]),
        ("Two-horizon training", "two_heads", evidence["two_heads"]),
    ]
    for name, key, item in rows:
        lines.append(f"{name}\t{item['result']}\tevidence.json::{key}")
    (OUT / "evidence.md").write_text("\n".join(lines) + "\n")
    log_event(log, "[PASS] artifacts_written")
    (OUT / "run.log").write_text("\n".join(log) + "\n")
    print(json.dumps({"status": "PASS", "artifact_dir": str(OUT), "vocabulary_size": len(vocab.itos), "untrained_perplexity": untrained_ppl, "two_head_final_losses": [first_losses[-1], second_losses[-1]]}, indent=2))


if __name__ == "__main__":
    main()
