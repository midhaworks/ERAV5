"""Controlled K-CRF quality benchmark.

This pilot isolates sequence-dependency modelling from the transformer body:
the corpus consists of alternating UTF-8 ASCII bytes with a random starting
byte.  Independent slot logits can learn marginals but cannot represent the
alternation; K-CRF can use transitions.  It is deliberately not presented as
a natural-language result.
"""

from __future__ import annotations

import json
import math
import time
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from kcrf import EOS, KCRFHead


OUT = Path(__file__).resolve().parent / "artifacts" / "kcrf_benchmark"
STATES = 259
LENGTH = 7  # six alternating bytes plus EOS
SEED = 2308


def make_data(n: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(2, (n,), generator=generator)
    rows = []
    for start in starts.tolist():
        # A=65 and B=66; state IDs are byte+3.
        values = [3 + (65 if (index + start) % 2 == 0 else 66) for index in range(LENGTH - 1)]
        rows.append(values + [EOS])
    return torch.tensor(rows, dtype=torch.long)


def independent_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    expanded = logits.unsqueeze(0).expand(len(targets), -1, -1)
    return F.cross_entropy(expanded.reshape(-1, STATES), targets.reshape(-1))


def train(seed: int = SEED, steps: int = 500, learning_rate: float = 0.08) -> dict:
    torch.manual_seed(seed)
    train_targets = make_data(256, seed)
    test_targets = make_data(512, seed + 1)
    # Independent slot model: one categorical distribution per position.
    independent = torch.nn.Parameter(torch.zeros(LENGTH, STATES))
    opt_i = torch.optim.Adam([independent], lr=0.15)
    # K-CRF: same unary degrees of freedom, plus rank-4 transitions.
    active = (EOS, 2, 3 + 65, 3 + 66)
    head = KCRFHead(hidden=8, codebook=torch.nn.Embedding(STATES, 8), rank=4,
                    active_states=active, positive_transitions=False)
    unary = torch.nn.Parameter(torch.zeros(LENGTH, STATES))
    opt_k = torch.optim.Adam(list(head.parameters()) + [unary], lr=learning_rate)
    for _ in range(steps):
        loss_i = independent_nll(independent, train_targets)
        opt_i.zero_grad(); loss_i.backward(); opt_i.step()
        # Exact reference NLL is evaluated per sequence because this pilot is
        # intentionally small; the implementation is differentiable.
        loss_k = torch.stack([head.nll(unary, target) for target in train_targets[:32]]).mean()
        opt_k.zero_grad(); loss_k.backward(); opt_k.step()

    with torch.no_grad():
        independent_test_nll = float(independent_nll(independent, test_targets))
        # Normalize the exact sequence NLL to the same per-state unit as the
        # independent cross-entropy baseline.
        kcrf_test_nll = float(torch.stack([head.nll(unary, target) for target in test_targets[:64]]).mean()) / LENGTH
        independent_pred = independent.argmax(-1).expand(len(test_targets), -1)
        kcrf_pred = head.viterbi(unary).ids
        independent_exact = float((independent_pred == test_targets).all(dim=1).float().mean())
        kcrf_exact = float(sum(tuple(row.tolist()) == kcrf_pred for row in test_targets[:64])) / 64
    result = {
        "protocol": "controlled alternating UTF-8 byte dependency pilot",
        "seed": seed, "train_examples": len(train_targets), "test_examples": len(test_targets),
        "sequence_length": LENGTH, "states": STATES, "steps": steps, "learning_rate": learning_rate,
        "independent": {"nll": independent_test_nll, "exact_match": independent_exact,
                         "parameters": int(independent.numel())},
        "kcrf": {"nll": kcrf_test_nll, "exact_match": kcrf_exact,
                 "parameters": sum(p.numel() for p in head.parameters()) + int(unary.numel()),
                 "head": head.parameter_report(), "viterbi": list(kcrf_pred)},
        "kcrf_improvement": (independent_test_nll - kcrf_test_nll) / independent_test_nll,
        "interpretation": "structural pilot only; not a natural-language quality claim",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    args = parser.parse_args()
    started = time.perf_counter()
    result = train(steps=args.steps, learning_rate=args.learning_rate)
    result["seconds"] = time.perf_counter() - started
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
