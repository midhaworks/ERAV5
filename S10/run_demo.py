"""Session 10: an honest, observable training loop.

Run from the repository root with ``python3 S10/run_demo.py``.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/s10-mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
SEED = 1010


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position = nn.Embedding(32, d_model)
        self.body = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.head = nn.Linear(d_model, vocab_size)
        self.head2 = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, return_hidden=False):
        positions = torch.arange(tokens.shape[1])
        hidden = self.embedding(tokens) + self.position(positions)[None, :, :]
        hidden = self.body(hidden)
        return hidden if return_hidden else self.head(hidden)


def ce(logits, tokens):
    """Correct causal shift: output t predicts token t+1."""
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), ignore_index=0)


def shape_lines(tokens, hidden, logits):
    return [
        f"tokens.shape={tuple(tokens.shape)}  # batch, sequence length; integer token ids",
        f"hidden.shape={tuple(hidden.shape)}  # batch, sequence length, d_model; model states",
        f"logits.shape={tuple(logits.shape)}  # batch, sequence length, vocabulary classes",
        f"targets.shape={tuple(tokens[:, 1:].shape)}  # batch, sequence length minus one; next token",
        f"loss_input.shape={tuple(logits[:, :-1].reshape(-1, logits.shape[-1]).shape)}  # flattened causal positions, vocabulary",
    ]


def token_strings(ids, itos):
    return " ".join(itos[int(x)] for x in ids)


def grad_check(model, tokens):
    model.zero_grad(set_to_none=True)
    logits = model(tokens)
    loss = ce(logits, tokens)
    loss.backward()
    parameter = model.embedding.weight
    index = (2, 0)  # a real vocabulary entry, not the padding row
    analytical = float(parameter.grad[index])
    epsilon = 1e-3
    original = float(parameter[index].detach())
    with torch.no_grad():
        parameter[index] = original + epsilon
    plus = float(ce(model(tokens), tokens).detach())
    with torch.no_grad():
        parameter[index] = original - epsilon
    minus = float(ce(model(tokens), tokens).detach())
    with torch.no_grad():
        parameter[index] = original
    numerical = (plus - minus) / (2 * epsilon)
    return {"parameter": "embedding.weight[2,0]", "epsilon": epsilon, "backward": analytical, "finite_difference": numerical, "absolute_error": abs(analytical - numerical), "result": "PASS" if abs(analytical - numerical) < 1e-3 else "FAIL"}


def micro_batch(model, tokens, optimizer, correct):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    counts = []
    for micro in tokens:
        logits = model(micro)
        loss = ce(logits, micro)
        count = micro.numel() - micro.shape[0]
        (loss * count if correct else loss).backward()
        losses.append(float(loss.detach()))
        counts.append(count)
    if correct:
        reported = sum(loss * count for loss, count in zip(losses, counts)) / sum(counts)
        # gradients were accumulated as loss*count; put them on mean-loss scale.
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(sum(counts))
    else:
        reported = sum(losses) / len(losses)
        # average of averages: each micro gradient receives equal weight.
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(len(losses))
    optimizer.step()
    return reported, losses, counts


def float_bits():
    return {
        "0.1": {
            "fp32": {"hex": "0x3dcccccd", "bits": "0 01111011 10011001100110011001101", "value": 0.10000000149011612},
            "bf16": {"hex": "0x3dcd", "bits": "0 01111011 1001101", "value": 0.10009765625},
            "fp8_e4m3": {"hex": "0x1d", "bits": "0 0011 101", "value": 0.1015625, "assumption": "E4M3FN, bias 7"},
        },
        "training_choice": "bf16 for suitable accelerator training; fp32 master weights/accumulation for stability; fp8 only with calibrated kernels and loss scaling",
    }


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    ARTIFACTS.mkdir(exist_ok=True)
    for path in ARTIFACTS.iterdir():
        if path.is_file(): path.unlink()

    documents = [
        "the quick brown fox jumps over the dog",
        "the small model learns from real data",
        "careful measurements reveal hidden bugs",
        "training loops should report what happened",
    ]
    itos = ["<pad>", "<eos>"] + sorted(set(" ".join(documents).split()))
    stoi = {word: i for i, word in enumerate(itos)}
    encoded = [[stoi[word] for word in doc.split()] + [stoi["<eos>"]] for doc in documents]
    length = max(map(len, encoded))
    tokens = torch.tensor([row + [stoi["<pad>"]] * (length - len(row)) for row in encoded])
    model = TinyLanguageModel(len(itos))
    hidden = model(tokens, return_hidden=True)
    logits = model.head(hidden)
    log = ["[PASS] run_started", *shape_lines(tokens, hidden, logits)]
    log += ["strings INPUT: " + token_strings(tokens[0, :-1], itos), "strings TARGET: " + token_strings(tokens[0, 1:], itos)]
    untrained_loss = ce(logits, tokens)
    untrained_perplexity = float(torch.exp(untrained_loss.detach()))
    contributing_tokens = int((tokens[:, 1:] != 0).sum())
    log.append(f"untrained nll={float(untrained_loss.detach()):.6f} perplexity={untrained_perplexity:.6f} vocabulary_size={len(itos)} contributing_tokens={contributing_tokens}")
    gradient = grad_check(model, tokens)
    log.append(f"gradient check backward={gradient['backward']:.8f} finite_difference={gradient['finite_difference']:.8f} error={gradient['absolute_error']:.8f}")

    # Deliberately broken and correct accumulation on unequal micro-batch lengths.
    # Each micro-batch is unpadded; their sequence lengths (5 and 8) differ.
    micros = [torch.tensor([encoded[2][:-1]]), torch.tensor([encoded[0][:-1]])]
    broken_model = deepcopy(model)
    correct_model = deepcopy(model)
    broken_opt = torch.optim.SGD(broken_model.parameters(), lr=0.05)
    correct_opt = torch.optim.SGD(correct_model.parameters(), lr=0.05)
    broken_curve, correct_curve = [], []
    accumulation_rows = []
    for step in range(8):
        broken, broken_parts, counts = micro_batch(broken_model, micros, broken_opt, correct=False)
        correct, correct_parts, _ = micro_batch(correct_model, micros, correct_opt, correct=True)
        broken_curve.append(broken); correct_curve.append(correct)
        accumulation_rows.append({"step": step, "broken_average_of_averages": broken, "correct_token_weighted": correct, "micro_losses": broken_parts, "contributing_tokens": counts})
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(broken_curve, marker="o", label="broken average of averages")
    ax.plot(correct_curve, marker="o", label="correct token-weighted")
    ax.set(xlabel="macro step", ylabel="reported loss", title="Unequal-length micro-batch accumulation")
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS / "accumulation_curves.png", dpi=140); plt.close(fig)
    log.append(f"accumulation micro_counts={accumulation_rows[0]['contributing_tokens']} first_broken={broken_curve[0]:.6f} first_correct={correct_curve[0]:.6f}")

    # Real optimization loop and gradient-norm trace.
    train_model = TinyLanguageModel(len(itos))
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=0.03)
    trace = []
    started = time.perf_counter()
    for step in range(25):
        optimizer.zero_grad(set_to_none=True)
        step_hidden = train_model(tokens, return_hidden=True)
        step_logits = train_model.head(step_hidden)
        step_loss = ce(step_logits, tokens)
        step_loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(train_model.parameters(), float("inf")))
        optimizer.step()
        trace.append({"step": step, "loss": float(step_loss.detach()), "grad_norm": grad_norm})
        if step == 0:
            for line in shape_lines(tokens, step_hidden, step_logits): log.append("train " + line)
    candidate_steps = []
    for left, right in zip(trace, trace[1:]):
        candidate_steps.append((abs(right["grad_norm"] - left["grad_norm"]) / (abs(right["loss"] - left["loss"]) + 1e-9), left["step"], right["step"]))
    _, prior_step, chosen_step = max(candidate_steps)
    log.append(f"grad_norm_before_loss step={chosen_step} prior={prior_step} norm_delta={trace[chosen_step]['grad_norm'] - trace[prior_step]['grad_norm']:.6f} loss_delta={trace[chosen_step]['loss'] - trace[prior_step]['loss']:.6f}")

    # Part 2: train a second head for the t+2 target and retain separate traces.
    horizon_model = TinyLanguageModel(len(itos))
    horizon_optimizer = torch.optim.AdamW(horizon_model.parameters(), lr=0.03)
    first_losses, second_losses = [], []
    for step in range(30):
        horizon_hidden = horizon_model(tokens, return_hidden=True)
        first_loss = ce(horizon_model.head(horizon_hidden), tokens)
        second_loss = F.cross_entropy(horizon_model.head2(horizon_hidden[:, :-2]).reshape(-1, len(itos)), tokens[:, 2:].reshape(-1), ignore_index=0)
        (first_loss + second_loss).backward()
        horizon_optimizer.step()
        horizon_optimizer.zero_grad(set_to_none=True)
        first_losses.append(float(first_loss.detach()))
        second_losses.append(float(second_loss.detach()))
    log.append(f"head1_targets_t_plus_1.shape={tuple(tokens[:, 1:].shape)}  # batch, sequence length minus one")
    log.append(f"head2_targets_t_plus_2.shape={tuple(tokens[:, 2:].shape)}  # batch, sequence length minus two")
    log.append(f"head2_logits.shape={tuple(horizon_model.head2(horizon_hidden).shape)}  # batch, sequence length, vocabulary classes")
    log.append(f"part2 losses first_head_initial={first_losses[0]:.6f} first_head_final={first_losses[-1]:.6f} second_head_initial={second_losses[0]:.6f} second_head_final={second_losses[-1]:.6f} sum_final={first_losses[-1] + second_losses[-1]:.6f}")

    elapsed = time.perf_counter() - started
    # The first-head loop does not execute head2, so exclude that inactive head
    # from this rough dense-FLOP estimate.
    parameters = sum(p.numel() for name, p in train_model.named_parameters() if not name.startswith("head2."))
    contributing_tokens = int((tokens[:, 1:] != 0).sum())
    flops = 6 * parameters * contributing_tokens * len(trace)
    peak_flops_assumption = 1e9  # conservative one-core CPU reference, stated explicitly
    mfu = flops / elapsed / peak_flops_assumption
    mfu_record = {"parameters": parameters, "tokens_per_step": contributing_tokens, "steps": len(trace), "elapsed_seconds": elapsed, "estimated_training_flops": flops, "assumed_peak_flops": peak_flops_assumption, "mfu": mfu, "mfu_percent": 100 * mfu, "honest_note": "6*parameter*tokens is a rough dense-transformer estimate; the one-core CPU peak is an assumption, so this is a diagnostic lower-bound estimate, not a vendor benchmark."}
    log.append(f"MFU estimated={100*mfu:.6f}% assumed_peak={peak_flops_assumption:.1e} FLOP/s")

    result = {"seed": SEED, "vocabulary": itos, "shapes": shape_lines(tokens, hidden, logits), "untrained_perplexity": {"nll": float(untrained_loss.detach()), "value": untrained_perplexity, "vocabulary_size": len(itos), "contributing_tokens": contributing_tokens}, "gradient_check": gradient, "accumulation": {"rows": accumulation_rows, "plot": "accumulation_curves.png"}, "gradient_trace": trace, "grad_norm_before_loss_step": chosen_step, "mfu": mfu_record, "float_bits": float_bits(), "two_horizon": {"first_loss_initial": first_losses[0], "first_loss_final": first_losses[-1], "second_loss_initial": second_losses[0], "second_loss_final": second_losses[-1], "sum_final": first_losses[-1] + second_losses[-1]}}
    (ARTIFACTS / "evidence.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (ARTIFACTS / "run.log").write_text("\n".join(log) + "\n")
    (ARTIFACTS / "grad_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "artifact_dir": str(ARTIFACTS), "gradient_error": gradient["absolute_error"], "mfu_percent": mfu_record["mfu_percent"], "grad_norm_before_loss_step": chosen_step}, indent=2))


if __name__ == "__main__":
    main()
