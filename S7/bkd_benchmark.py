"""Blockwise Kronecker Decoder (BKD) pilot.

Unlike K-CRF, BKD predicts small groups causally with ordinary cross-entropy.
The grouped decoder carries a learned boundary state and uses the tied RKE
codebook for every output score.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from kcrf import EOS, CONT, _valid_next
from rke import utf8_is_complete, utf8_prefix_is_valid
from kcrf_conditional_benchmark import load_data


OUT = Path(__file__).resolve().parent / "artifacts" / "bkd_benchmark"
STATES, CODE, HIDDEN, LENGTH = 259, 32, 64, 7


class BKD(nn.Module):
    def __init__(self, group: int, seed: int):
        super().__init__(); torch.manual_seed(seed)
        self.group = group
        self.codebook = nn.Embedding(STATES, CODE)
        self.encoder = nn.GRU(CODE, HIDDEN, batch_first=True)
        self.decoder = nn.GRU(CODE, HIDDEN, batch_first=True)
        self.to_code = nn.Linear(HIDDEN, CODE, bias=False)
        self.position = nn.Parameter(torch.randn(group, CODE) * .02)

    def encode(self, sources):
        padded = torch.zeros((len(sources), max(map(len, sources))), dtype=torch.long)
        for i, source in enumerate(sources): padded[i, :len(source)] = torch.tensor(source)
        _, hidden = self.encoder(self.codebook(padded)); return hidden

    def logits(self, hidden, count):
        base = self.to_code(hidden).unsqueeze(1) + self.position[:count].unsqueeze(0)
        return base @ self.codebook.weight.T / CODE ** .5

    def loss(self, sources, targets):
        hidden = self.encode(sources); total = []
        previous = torch.full((len(sources), 1), CONT, dtype=torch.long)
        for start in range(0, LENGTH, self.group):
            count = min(self.group, LENGTH - start)
            step, hidden = self.decoder(self.codebook(previous), hidden)
            logits = self.logits(step[:, 0], count)
            gold = torch.stack([target[start:start + count] for target in targets])
            total.append(F.cross_entropy(logits.reshape(-1, STATES), gold.reshape(-1)))
            previous = gold
        return torch.stack(total).mean()

    @torch.no_grad()
    def generate(self, source):
        hidden = self.encode([source]); previous = torch.tensor([[CONT]])
        output, pending = [], b""
        for start in range(0, LENGTH, self.group):
            count = min(self.group, LENGTH - start)
            step, hidden = self.decoder(self.codebook(previous), hidden)
            scores = self.logits(step[:, 0], count)[0]
            group = []
            for offset in range(count):
                position = start + offset
                chosen = -1
                chosen_pending = pending
                for state in scores[offset].argsort(descending=True).tolist():
                    new_pending, allowed = _valid_next(pending, state, position, LENGTH - 1)
                    if allowed:
                        chosen = state; chosen_pending = new_pending; break
                if chosen < 0: chosen = EOS if position == LENGTH - 1 else 3 + 32
                group.append(chosen); output.append(chosen); pending = chosen_pending
            previous = torch.tensor([group])
        return tuple(output)

    @torch.no_grad()
    def test_nll(self, rows):
        values = []
        for source, target, _ in rows:
            hidden = self.encode([source]); previous = torch.tensor([[CONT]])
            total = 0.0
            for start in range(0, LENGTH, self.group):
                count = min(self.group, LENGTH - start)
                step, hidden = self.decoder(self.codebook(previous), hidden)
                logits = self.logits(step[:, 0], count)[0]
                gold = torch.tensor(target[start:start + count])
                total += float(F.cross_entropy(logits, gold, reduction="sum"))
                previous = gold[None]
            values.append(total / LENGTH)
        return sum(values) / len(values)


def run(steps: int = 80):
    data, _ = load_data(); results = {}
    for group in (1, 2, 4):
        model = BKD(group, 2400 + group); opt = torch.optim.Adam(model.parameters(), lr=.003)
        train = data["train"]
        for step in range(steps):
            rows = train[(step * 8) % len(train): (step * 8) % len(train) + 8]
            sources = [r[0] for r in rows]; targets = [torch.tensor(r[1]) for r in rows]
            loss = model.loss(sources, targets); opt.zero_grad(); loss.backward(); opt.step()
        start = time.perf_counter(); exact = valid = utf8_valid = 0
        for source, target, _ in data["test"]:
            prediction = model.generate(source); exact += prediction == tuple(target); valid += prediction[-1] == EOS
            payload = bytes(state - 3 for state in prediction[:-1] if state >= 3)
            utf8_valid += int(utf8_is_complete(payload) and utf8_prefix_is_valid(payload))
        rate = len(data["test"]) / (time.perf_counter() - start)
        results[str(group)] = {"group_size": group, "exact_match": exact / len(data["test"]),
                               "valid_eos": valid / len(data["test"]), "utf8_valid": utf8_valid / len(data["test"]),
                               "per_state_nll": model.test_nll(data["test"]),
                               "decode_examples_per_second": rate,
                               "parameters": sum(p.numel() for p in model.parameters())}
    result = {"protocol": "conditional blockwise Kronecker decoder pilot", "steps": steps,
              "results": results, "interpretation": "simpler alternative to full K-CRF"}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    started = time.perf_counter(); result = run(); result["seconds"] = time.perf_counter() - started
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result))
