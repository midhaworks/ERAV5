"""Natural-text byte dependency benchmark for the exact K-CRF reference head.

This is an unconditional structural benchmark: it uses six-byte ASCII words
from the existing paragraph-disjoint train/validation/test corpus. It does not
claim full language-model quality because no transformer context is supplied.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from kcrf import EOS, KCRFHead
from natural_corpus import build_dataset


OUT = Path(__file__).resolve().parent / "artifacts" / "kcrf_natural_benchmark"
STATES = 259
LENGTH = 7


def collect(seed: int = 2308):
    data, audit = build_dataset({"train": 1000, "validation": 200, "test": 200})
    result = {}
    for split, records in data.items():
        words = sorted({r["target"] for r in records
                        if r["target"].isascii() and len(r["target"].encode("ascii")) == 6})
        # Deterministic split source already comes from paragraph-disjoint data;
        # deduplicating within each split prevents repeated word weighting.
        result[split] = torch.tensor(
            [[3 + b for b in word.encode("ascii")] + [EOS] for word in words], dtype=torch.long)
    return result, audit


def independent_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    expanded = logits.unsqueeze(0).expand(len(targets), -1, -1)
    return F.cross_entropy(expanded.reshape(-1, STATES), targets.reshape(-1))


def run(steps: int = 50, learning_rate: float = 0.04) -> dict:
    torch.manual_seed(2308)
    data, audit = collect()
    active = sorted({int(x) for rows in data.values() for x in rows.reshape(-1).tolist()})
    independent = torch.nn.Parameter(torch.zeros(LENGTH, STATES))
    opt_i = torch.optim.Adam([independent], lr=learning_rate)
    head = KCRFHead(8, torch.nn.Embedding(STATES, 8), rank=8,
                    active_states=tuple(active), positive_transitions=False)
    unary = torch.nn.Parameter(torch.zeros(LENGTH, STATES))
    opt_k = torch.optim.Adam(list(head.parameters()) + [unary], lr=learning_rate)
    train = data["train"]
    for _ in range(steps):
        loss_i = independent_nll(independent, train)
        opt_i.zero_grad(); loss_i.backward(); opt_i.step()
        # Small batches keep the exact reference chart tractable.
        indices = torch.randperm(len(train))[: min(8, len(train))]
        loss_k = torch.stack([head.nll(unary, train[i]) for i in indices]).mean()
        opt_k.zero_grad(); loss_k.backward(); opt_k.step()
    test = data["test"]
    with torch.no_grad():
        independent_test = float(independent_nll(independent, test))
        kcrf_test = float(torch.stack([head.nll(unary, row) for row in test]).mean()) / LENGTH
        independent_pred = independent.argmax(-1).expand(len(test), -1)
        kcrf_pred = head.viterbi(unary).ids
        independent_exact = float((independent_pred == test).all(dim=1).float().mean())
        kcrf_exact = float(sum(tuple(row.tolist()) == kcrf_pred for row in test)) / max(1, len(test))
    result = {
        "protocol": "natural six-byte ASCII word dependency pilot",
        "source_audit": audit,
        "split_counts": {split: len(rows) for split, rows in data.items()},
        "active_states": active,
        "sequence_length": LENGTH,
        "steps": steps,
        "learning_rate": learning_rate,
        "independent": {"per_state_nll": independent_test, "exact_match": independent_exact,
                         "parameters": int(independent.numel())},
        "kcrf": {"per_state_nll": kcrf_test, "exact_match": kcrf_exact,
                 "parameters": sum(p.numel() for p in head.parameters()) + int(unary.numel()),
                 "head": head.parameter_report(), "viterbi": list(kcrf_pred)},
        "nll_improvement": (independent_test - kcrf_test) / independent_test,
        "interpretation": "natural-text structural pilot only; no transformer context",
        "data_hash": hashlib.sha256(json.dumps(
            {split: rows.tolist() for split, rows in data.items()}, sort_keys=True).encode()).hexdigest(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    started = time.perf_counter()
    result = run()
    result["seconds"] = time.perf_counter() - started
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
