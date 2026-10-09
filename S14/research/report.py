"""Generate a readable report from actual pilot results; no expected scores."""
import json
from pathlib import Path
from statistics import mean


def main():
    root = Path(__file__).parent / "artifacts"
    records = [(p, json.loads(p.read_text())) for p in sorted(root.glob("seed_*/results.json"))]
    rows = ["# Generated synthetic pilot results", "",
            "Each role has 12 held-out operation/constant pairs per seed. These are",
            "arithmetic generation and one edit-action tasks, not repository-level agent tasks.", "",
            "| Seed | Model | Parameters | Coding successes | Edit-action successes | Test NLL |",
            "|---:|---|---:|---:|---:|---:|"]
    aggregate = {}
    for path, data in records:
        for name, run in data["runs"].items():
            def score(role):
                traces = [t for t in run["test"]["traces"] if t["role"] == role]
                return f'{sum(t["passed"] for t in traces)}/{len(traces)}'
            rows.append(f'| {data["seed"]} | {name} | {run["parameters"]:,} | '
                        f'{score("coding")} | {score("agentic")} | {run["test"]["response_nll"]:.5f} |')
            aggregate.setdefault(name, []).append(run["test"]["success_by_role"])
    rows.extend(["", "Mean success across seeds (small correlated task set; no significance claim):", ""])
    for name, scores in aggregate.items():
        rows.append(f'* {name}: coding {mean(s["coding"] for s in scores):.1%}, '
                    f'edit action {mean(s["agentic"] for s in scores):.1%}.')
    rows.extend(["", "Comparisons are token-matched; MoE activates two FFN experts per token versus",
                 "one FFN in the dense control. Training compute is therefore not matched.",
                 "Group choice is supplied by the task role. Export agreement establishes",
                 "equivalence to the masked parent, not quality retention from the unrestricted parent.",
                 "Shared attention, embeddings and head are duplicated in each independent export.", ""])
    (root / "summary.md").write_text("\n".join(rows))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
