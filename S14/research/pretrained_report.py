"""Audit saved generations and produce the local baseline report."""
import hashlib
import json
from collections import Counter
from pretrained_baseline import OUT, suite, check, extract


def main():
    path=OUT/"results.json"
    data=json.loads(path.read_text())
    assert data["status"]=="completed_bounded_validation"
    tasks={t["id"]:t for t in suite()}
    audits=[]
    for trace in data["traces"]:
        task=tasks[trace["task"]]
        # Rotation explicitly asks to preserve its input; verify that too.
        audited=check(extract(trace["raw_response"]),task["tests"],
                      preserve_inputs=task["id"]=="rotate")
        assert audited["passed"]==trace["passed"], "Recorded pass count disagrees with audit"
        audits.append(dict(task=trace["task"],role=trace["role"],style=trace["style"],**audited))
    (OUT/"audit.json").write_text(json.dumps(dict(
        results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        input_preservation_checked_for="rotate",outcomes=audits),indent=2)+"\n")
    total=len(data["traces"])
    passed=sum(t["passed"] for t in data["traces"])
    seconds=sum(t["elapsed_seconds"] for t in data["traces"])
    tokens=sum(t["generated_tokens"] for t in data["traces"])
    peak=max(t["peak_mlx_bytes"] for t in data["traces"])/1024**3
    failures=Counter(t["error"] for t in data["traces"] if not t["passed"])
    lines=["# Generated M2 pretrained baseline report", "",
        f'Model: `{data["model"]}`; revision `{data["revision"]}`.',
        "Greedy, non-thinking, community 4-bit quantization; authored development validation tasks.","",
        "| Task mode | Passed |", "|---|---:|"]
    for name,score in data["scores"].items():
        lines.append(f'| {name} | {score["passed"]}/{score["total"]} |')
    lines.extend(["",f'Overall: {passed}/{total} ({passed/total:.1%}).',
        f'Generation throughput including prompt processing: {tokens/seconds:.2f} tokens/s.',
        f'Peak MLX allocation: {peak:.2f} GiB (does not include total system RAM).',
        f'Aggregate generation time: {seconds:.1f} seconds; loading/download excluded.',"",
        "Failure counts:","", *[f'* {kind}: {count}' for kind,count in failures.items()],"",
        "Raw responses remain unchanged. Audit re-executes them against their tests and also checks rotation input preservation.",
        "The initial smoke checker rejected valid set.add calls; initial_smoke_restricted.json preserves that diagnostic.",
        "The corrected checker permits ordinary collection methods and still excludes imports and introspection.",
        "A wrong function name remains a failed API contract; the report does not rename it or substitute a reference solution.",
        "This is narrow generation/direct-repair validation, not multi-step agent evaluation or a locked final benchmark.",
        "No training, MoE conversion, expert quality gain or production readiness is demonstrated by this run.",""])
    (OUT/"summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__=="__main__":
    main()
