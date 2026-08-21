"""Matched conditional K-CRF versus independent RKE-slot benchmark."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from kcrf import EOS, KCRFHead
from natural_corpus import build_dataset


OUT = Path(__file__).resolve().parent / "artifacts" / "kcrf_conditional_benchmark"
STATES, LENGTH, HIDDEN, CODE = 259, 7, 64, 32


def load_data():
    data, audit = build_dataset({"train": 1000, "validation": 200, "test": 200})
    selected = {}
    for split, records in data.items():
        rows = []
        for record in records:
            target = record["target"].encode("utf-8")
            if len(target) != 6:
                continue
            context = (" ".join(record["context"])).encode("utf-8")[-32:]
            source = [b + 3 for b in context]
            target_states = [b + 3 for b in target] + [EOS]
            rows.append((source, target_states, record["language"]))
        selected[split] = rows
    return selected, audit


class Body(nn.Module):
    def __init__(self, codebook: nn.Embedding, output: nn.Module):
        super().__init__()
        self.codebook = codebook
        self.encoder = nn.GRU(CODE, HIDDEN, batch_first=True)
        self.context = nn.Linear(HIDDEN, CODE)
        self.position = nn.Parameter(torch.randn(LENGTH, CODE) * 0.02)
        self.output = output

    def unary(self, sources: list[list[int]]) -> torch.Tensor:
        padded = torch.zeros((len(sources), max(map(len, sources))), dtype=torch.long)
        for i, source in enumerate(sources): padded[i, :len(source)] = torch.tensor(source)
        _, hidden = self.encoder(self.codebook(padded))
        base = self.context(hidden[0]).unsqueeze(1) + self.position.unsqueeze(0)
        return self.output(base)


def batch_nll_kcrf(head, unary, targets):
    return torch.stack([head.nll(row, target) for row, target in zip(unary, targets)]).mean() / LENGTH


def run(steps: int = 120, batch_size: int = 8, seed: int = 2308) -> dict:
    torch.manual_seed(seed)
    data, audit = load_data()
    active = sorted({value for split in data.values() for source, target, _ in split for value in target})
    codebook = nn.Embedding(STATES, CODE)
    khead = KCRFHead(CODE, codebook, rank=8, active_states=tuple(active), positive_transitions=False)
    kbody = Body(codebook, khead.unary)
    # Independent arm has a matched body and the same shared codebook shape.
    ibody = Body(nn.Embedding(STATES, CODE), nn.Linear(CODE, STATES, bias=False))
    kparams = list(kbody.parameters()) + [khead.to_code.weight,
                                          khead.transition_left,
                                          khead.transition_right]
    kopt = torch.optim.Adam(kparams, lr=0.002)
    iopt = torch.optim.Adam(ibody.parameters(), lr=0.01)
    train = data["train"]
    generator = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        indices = torch.randperm(len(train), generator=generator)[:batch_size]
        rows = [train[int(i)] for i in indices]
        sources = [row[0] for row in rows]
        targets = [torch.tensor(row[1], dtype=torch.long) for row in rows]
        ku = kbody.unary(sources)
        kloss = batch_nll_kcrf(khead, ku, targets)
        kopt.zero_grad(); kloss.backward(); torch.nn.utils.clip_grad_norm_(kparams, 1.0); kopt.step()
        iu = ibody.unary(sources)
        iloss = F.cross_entropy(iu.reshape(-1, STATES), torch.stack(targets).reshape(-1))
        iopt.zero_grad(); iloss.backward(); torch.nn.utils.clip_grad_norm_(ibody.parameters(), 1.0); iopt.step()

    def evaluate_k(rows):
        nlls, exact, valid = [], 0, 0
        for source, target, _ in rows:
            unary = kbody.unary([source])[0]
            nlls.append(float(khead.nll(unary, torch.tensor(target)).detach()) / LENGTH)
            prediction = khead.viterbi(unary).ids
            exact += tuple(target) == prediction
            valid += prediction[-1] == EOS
        return {"per_state_nll": sum(nlls) / len(nlls), "exact_match": exact / len(nlls), "valid_eos": valid / len(nlls)}

    def evaluate_i(rows):
        nlls, exact = [], 0
        for source, target, _ in rows:
            unary = ibody.unary([source])[0]
            nlls.append(float(F.cross_entropy(unary, torch.tensor(target))))
            exact += tuple(target) == tuple(unary.argmax(-1).tolist())
        return {"per_state_nll": sum(nlls) / len(nlls), "exact_match": exact / len(nlls)}

    kresult, iresult = evaluate_k(data["test"]), evaluate_i(data["test"])
    start = time.perf_counter()
    for source, _, _ in data["test"]:
        khead.viterbi(kbody.unary([source])[0])
    k_decode_rate = len(data["test"]) / (time.perf_counter() - start)
    start = time.perf_counter()
    for source, _, _ in data["test"]:
        ibody.unary([source])[0].argmax(-1)
    i_decode_rate = len(data["test"]) / (time.perf_counter() - start)
    per_language = {}
    for language in sorted({row[2] for row in data["test"]}):
        rows = [row for row in data["test"] if row[2] == language]
        per_language[language] = {"independent": evaluate_i(rows), "kcrf": evaluate_k(rows)}
    result = {"protocol": "conditional six-byte target benchmark", "steps": steps,
              "batch_size": batch_size, "split_counts": {key: len(value) for key, value in data.items()},
              "languages": sorted({row[2] for row in data["test"]}), "active_states": active,
              "independent": iresult, "kcrf": kresult,
              "per_language": per_language,
              "decode_examples_per_second": {"kcrf_viterbi": k_decode_rate,
                                               "independent_argmax": i_decode_rate,
                                               "ratio": k_decode_rate / i_decode_rate},
              "nll_improvement": (iresult["per_state_nll"] - kresult["per_state_nll"]) / iresult["per_state_nll"],
              "source_audit": audit, "interpretation": "small conditional pilot; not production-scale evidence",
              "data_hash": hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    started = time.perf_counter(); result = run(); result["seconds"] = time.perf_counter() - started
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
