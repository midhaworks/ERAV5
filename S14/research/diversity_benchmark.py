"""Prompt diversification ablation; fixed model, compute proxy, held-out task pairs."""
import copy
import hashlib
import json
from pathlib import Path
from statistics import mean

import torch
from benchmark import LM, nll
from grouped_moe import parameters
from trajectory_benchmark import cases, training_records, fit, evaluate


def wording(text, family):
    replacements = {
        "train_a": (
            ("Write Python solve(x) returning", "Python solve(x) should return"),
            ("Repair solve.py to return", "Fix solve.py to return"),
            ("Read solve.py:", "File solve.py read:"),
            ("Edited solve.py:", "File solve.py edited:"),
            ("Next JSON action?", "Return next action as JSON.")),
        "train_b": (
            ("Write Python solve(x) returning", "Return code for solve(x) to return"),
            ("Repair solve.py to return", "File solve.py should return"),
            ("Read solve.py:", "Read file solve.py:"),
            ("Edited solve.py:", "Edited file solve.py:"),
            ("Next JSON action?", "Next action as JSON?")),
        "heldout": (
            ("Write Python solve(x) returning", "Write solve(x) in Python for"),
            ("Repair solve.py to return", "Repair file solve.py for"),
            ("Read solve.py:", "Read output for solve.py:"),
            ("Edited solve.py:", "Edited output for solve.py:"),
            ("Next JSON action?", "JSON for the next tool action?")),
    }
    for before, after in replacements[family]:
        text = text.replace(before, after)
    return text


def diversified(records):
    return [{**r, "prompt": r["prompt"] if family == "original" else wording(r["prompt"], family),
             "wording_family": family}
            for r in records for family in ("original", "train_a", "train_b")]


def main():
    torch.set_num_threads(2)
    root = Path(__file__).parent / "artifacts"
    data = cases()
    original_train = training_records([c for c in data if c["split"] == "train"])
    train = diversified(original_train)
    # Same original validation tasks/wording as the existing control's tuning.
    validation = training_records([c for c in data if c["split"] == "validation"])
    test = [c for c in data if c["split"] == "test"]
    test_records = training_records(test)
    seen = set("".join(r["prompt"]+r["answer"] for r in train).encode())
    assert set("".join(wording(r["prompt"], "heldout") for r in test_records).encode()) <= seen
    for seed in (14, 27, 42):
        torch.manual_seed(seed)
        dense = LM()
        old_dir = root / "trajectories" / f"seed_{seed}"
        old = json.loads((old_dir / "results.json").read_text())
        out = root / "diversity" / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(train+validation, sort_keys=True).encode()
        (out / "training_validation.json").write_bytes(payload)
        result = dict(seed=seed, torch=torch.__version__, status="synthetic_wording_diversity_ablation",
            device="cpu", data_sha256=hashlib.sha256(payload).hexdigest(),
            training_families=["original", "train_a", "train_b"], test_family="heldout",
            role_selector="oracle", controls="previous trajectory checkpoints, re-evaluated on new wording", runs={})
        result["pretraining"] = fit(dense, train, old["pretraining"]["budget"], seed, 0.002)
        for name in ("dense_continuation", "unrestricted_moe", "grouped_moe"):
            print(f"seed={seed} {name}", flush=True)
            grouped = name == "grouped_moe"
            limit = old["runs"][name]["selected_training"]["budget"]
            torch.manual_seed(seed+1)
            initial = copy.deepcopy(dense) if name == "dense_continuation" else dense.upcycle()
            trials, best, best_loss = [], None, float("inf")
            for lr in (0.001, 0.002):
                model = copy.deepcopy(initial)
                trained = fit(model, train, limit, seed+1, lr, grouped)
                loss = nll(model, validation, grouped)
                trial = dict(lr=lr, validation_nll=loss, training=trained)
                trials.append(trial)
                if loss < best_loss:
                    best, best_loss, selected = model, loss, trial
            baseline = LM() if name == "dense_continuation" else LM().upcycle()
            path = old_dir / f"{name}.pt"
            baseline.load_state_dict(torch.load(path, weights_only=True))
            baseline.eval()
            transform = lambda p: wording(p, "heldout")
            run = dict(parameters=parameters(best), trials=trials,
                selected_lr=selected["lr"], selected_training=selected["training"],
                total_search_flops_proxy=sum(t["training"]["flops_proxy"] for t in trials),
                train_nll=nll(best, train, grouped), validation_nll=best_loss,
                control_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                control_standard=evaluate(baseline, test, grouped),
                control_heldout_wording=evaluate(baseline, test, grouped, transform_prompt=transform),
                diversified_standard=evaluate(best, test, grouped),
                diversified_heldout_wording=evaluate(best, test, grouped, transform_prompt=transform))
            torch.save(best.state_dict(), out / f"{name}.pt")
            result["runs"][name] = run
            (out / "results.json").write_text(json.dumps(result, indent=2)+"\n")
            print(json.dumps({mode: run[mode]["scores"] for mode in
                  ("control_heldout_wording", "diversified_heldout_wording")}), flush=True)
    results = [json.loads((root / "diversity" / f"seed_{seed}" / "results.json").read_text())
               for seed in (14,27,42)]
    lines = ["# Generated instruction-diversity ablation", "",
        "Same architectures, task-pair splits, LR search and approximate FLOP budgets.",
        "The held-out wording family is absent from training and validation.", "",
        "| Model | Training wording | Standard coding | Standard repair | Held-out coding | Held-out repair |",
        "|---|---|---:|---:|---:|---:|"]
    for name in results[0]["runs"]:
        for prefix, label in (("control", "Original only"), ("diversified", "Three families")):
            vals = []
            for mode in (f"{prefix}_standard", f"{prefix}_heldout_wording"):
                for role in ("coding", "agentic"):
                    vals.append(mean(r["runs"][name][mode]["scores"][role]["success_rate"] for r in results))
            lines.append(f'| {name} | {label} | '+" | ".join(f'{v:.1%}' for v in vals)+" |")
    lines.extend(["", "All percentages are averages across seeds 14, 27 and 42 on 12 task pairs per role.",
        "New wording contains no unseen training bytes. Grouped routing still uses oracle roles.",
        "This is a synthetic fixed read/edit/test environment, with no retries or repository execution.",
        "Compute parity uses an analytical estimate; dispatch, optimizer and hardware effects are excluded.",
        "Instruction diversification triples prompt variants, not underlying task pairs or unique solutions.",
        "Previous control checkpoints are reused, with their hashes recorded, and tested on the same new prompts.",
        "Results do not establish statistical significance or a 30B scaling recipe.", ""])
    (root / "diversity" / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
