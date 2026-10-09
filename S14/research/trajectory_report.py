"""Reconstruct summary from trajectory run artifacts."""
import json
from pathlib import Path
from statistics import mean


def main():
    root = Path(__file__).parent / "artifacts" / "trajectories"
    results = [json.loads(p.read_text()) for p in sorted(root.glob("seed_*/results.json"))]
    rows = ["# Generated read/edit/test pilot report", "",
        "Synthetic arithmetic code tasks, oracle role labels, approximate FLOP budgets.",
        "Learning rates 0.001 and 0.002 selected by held-out validation NLL only.", "",
        "| Seed | Model | LR | Steps | FLOP budget used | Coding pass | Three-action repair pass |",
        "|---:|---|---:|---:|---:|---:|---:|"]
    aggregate = {}
    for result in results:
        assert len(result["runs"]) == 3, "Incomplete result"
        for name, run in result["runs"].items():
            training = run["selected_training"]
            assert 0.98 < training["budget_fraction"] <= 1.0
            scores = run["test"]["scores"]
            c, a = scores["coding"], scores["agentic"]
            rows.append(f'| {result["seed"]} | {name} | {run["selected_lr"]} | '
                f'{training["steps"]} | {training["budget_fraction"]:.2%} | '
                f'{c["passed"]}/{c["total"]} | {a["passed"]}/{a["total"]} |')
            aggregate.setdefault(name, []).append(scores)
            for exported in run.get("exports", {}).values():
                assert exported["masked_parent_max_logit_error"] < 1e-5
    rows.extend(["", "| Model | Mean coding success | Mean repair success |",
                 "|---|---:|---:|"])
    for name, scores in aggregate.items():
        rows.append(f'| {name} | {mean(s["coding"]["success_rate"] for s in scores):.1%} | '
                    f'{mean(s["agentic"]["success_rate"] for s in scores):.1%} |')
    rows.extend(["", "Evidence scope:", "",
        "* A repair succeeds only after generated read, edit and test actions and passing arithmetic tests.",
        "* Actual generated patches determine test results; no reference patch is substituted.",
        "* Shared pretraining cost is recorded separately. Both LR trials are charged in total_search_flops_proxy.",
        "* FLOP proxy covers attention, active FFNs, routers and vocabulary projection with backward estimated at twice forward.",
        "* It excludes optimizer, activation and dispatch costs; wall time is separate. This is not measured GPU compute parity.",
        "* Each role has only 12 correlated synthetic test cases; three seeds do not establish statistical significance.",
        "* Exports match role-masked parent logits; role selection is an oracle and generalization to repositories is untested.", ""])
    (root / "summary.md").write_text("\n".join(rows))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
