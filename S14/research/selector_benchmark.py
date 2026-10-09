"""Prompt-only group selection, evaluated against the same frozen parent."""
import copy
import hashlib
import json
import time
from pathlib import Path
from statistics import mean

import torch
from torch import nn
from torch.nn import functional as F
from benchmark import LM
from grouped_moe import parameters
from trajectory_benchmark import cases, training_records, evaluate

ROLES = ("coding", "agentic")


def features(prompts):
    # Byte frequency baseline deliberately discards word order.
    rows = []
    for text in prompts:
        tokens = torch.tensor(list(text.encode()), dtype=torch.long)
        counts = torch.bincount(tokens, minlength=256).float()
        rows.append(counts / max(len(tokens), 1))
    return torch.stack(rows)


class Selector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x)

    def choose(self, prompt):
        self.eval()
        with torch.no_grad():
            return ROLES[int(self(features([prompt]))[0].argmax())]


def shifted(text):
    for before, after in (
        ("Write Python solve(x) returning", "Return Python solve(x) for"),
        ("Repair solve.py to return", "Read and fix solve.py for"),
        ("Read solve.py:", "Read tool output:"),
        ("Edited solve.py:", "Edited file:"),
        ("Next JSON action?", "Next tool action in JSON?"),
    ):
        text = text.replace(before, after)
    return text


def classifier_evaluation(selector, records, transform=lambda x:x):
    traces = []
    for record in records:
        text = transform(record["prompt"])
        chosen = selector.choose(text)
        traces.append(dict(prompt=text, expected_role=record["role"], chosen_group=chosen,
                           correct=chosen == record["role"]))
    per_role = {role: sum(t["correct"] for t in traces if t["expected_role"] == role) /
                     sum(t["expected_role"] == role for t in traces) for role in ROLES}
    return dict(correct=sum(t["correct"] for t in traces), total=len(traces),
                accuracy_by_role=per_role, balanced_accuracy=mean(per_role.values()), traces=traces)


def fit_selector(seed, train, validation):
    torch.manual_seed(seed)
    model = Selector()
    x = features([r["prompt"] for r in train])
    y = torch.tensor([ROLES.index(r["role"]) for r in train])
    vx = features([r["prompt"] for r in validation])
    vy = torch.tensor([ROLES.index(r["role"]) for r in validation])
    # Inverse-frequency weighting avoids three action stages outvoting coding.
    weights = len(y) / (2 * torch.bincount(y, minlength=2).float())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    best, best_loss, best_step = None, float("inf"), 0
    trace = []
    started = time.perf_counter()
    for step in range(1, 301):
        model.train()
        loss = F.cross_entropy(model(x), y, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = float(F.cross_entropy(model(vx), vy, weight=weights))
            trace.append(dict(step=step, train_loss=float(loss.detach()), validation_loss=val_loss))
            if val_loss < best_loss:
                best, best_loss, best_step = copy.deepcopy(model), val_loss, step
    return best, dict(parameters=parameters(best), steps=300, selected_step=best_step,
                      validation_loss=best_loss, elapsed_seconds=time.perf_counter()-started, trace=trace)


def main():
    torch.set_num_threads(2)
    root = Path(__file__).parent / "artifacts"
    data = cases()
    records = training_records(data)
    train = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "validation"]
    test_records = [r for r in records if r["split"] == "test"]
    test_cases = [r for r in data if r["split"] == "test"]
    seen_bytes = set("".join(r["prompt"]+r["answer"] for r in train).encode())
    assert set("".join(shifted(r["prompt"]) for r in test_records).encode()) <= seen_bytes
    for seed in (14, 27, 42):
        checkpoint = root / "trajectories" / f"seed_{seed}" / "grouped_moe.pt"
        model = LM().upcycle()
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        selector, training = fit_selector(seed, train, validation)
        out = root / "selectors" / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        result = dict(seed=seed, status="synthetic_prompt_selector_pilot",
            parent_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            data_sha256=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
            selector_training=training, classifier_standard=classifier_evaluation(selector, test_records),
            classifier_shifted=classifier_evaluation(selector, test_records, shifted),
            oracle=evaluate(model, test_cases, role_mask=True),
            learned=evaluate(model, test_cases, selector=selector.choose),
            shifted_oracle=evaluate(model, test_cases, role_mask=True, transform_prompt=shifted),
            shifted_learned=evaluate(model, test_cases, selector=selector.choose, transform_prompt=shifted))
        (out / "results.json").write_text(json.dumps(result, indent=2)+"\n")
        torch.save(selector.state_dict(), out / "selector.pt")
        print(f'seed={seed} classifier standard={result["classifier_standard"]["correct"]}/48 '
              f'shifted={result["classifier_shifted"]["correct"]}/48', flush=True)
    # Report actual checkpoint results; never assume learned routing matches oracle.
    results = [json.loads((root / "selectors" / f"seed_{s}" / "results.json").read_text()) for s in (14,27,42)]
    lines = ["# Generated prompt-selector results", "",
        "Same frozen grouped parents as the trajectory pilot. The selector receives prompt text only.",
        "No final-test or shifted-template examples enter selector training or checkpoint selection.", "",
        "| Evaluation | Coding success | Three-action repair success |", "|---|---:|---:|"]
    for mode in ("oracle", "learned", "shifted_oracle", "shifted_learned"):
        scores = [r[mode]["scores"] for r in results]
        lines.append(f'| {mode} | {mean(s["coding"]["success_rate"] for s in scores):.1%} | '
                     f'{mean(s["agentic"]["success_rate"] for s in scores):.1%} |')
    lines.extend(["", "| Seed | Standard selector accuracy | Unseen wording accuracy | Shifted balanced accuracy |",
                  "|---:|---:|---:|---:|"])
    for r in results:
        a,b = r["classifier_standard"],r["classifier_shifted"]
        lines.append(f'| {r["seed"]} | {a["correct"]}/{a["total"]} | {b["correct"]}/{b["total"]} | '
                     f'{b["balanced_accuracy"]:.1%} |')
    lines.extend(["", "This small classifier uses normalized byte counts; it cannot represent token order.",
        "Group choice is fixed for each generation, then recomputed from observed state at the next action.",
        "Matched-template classification does not establish general routing capability or expert specialization.",
        "Overall accuracy is sensitive to the 3:1 action/coding class balance; balanced accuracy weights both roles equally.",
        "Shifted prompts introduce no byte values absent from parent training data; wording, not byte coverage, changes.",
        "Unseen wording changes the parent prompts too; shifted oracle separates parent brittleness from selector error.",
        "Only 12 test cases per role are evaluated. Reported success is not production readiness.", ""])
    summary = root / "selectors" / "summary.md"
    summary.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
