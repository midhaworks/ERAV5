"""Recompute equality and timing claims without rerunning model inference."""
import hashlib
import json
from efficiency_probe import OUT,REPO_OUT,replay_bundles,native_parser
from repository_suite import load
from repository_efficiency import parse_calls


def main():
    path=OUT/"results.json"
    result=json.loads(path.read_text())
    assert result["status"]=="completed_paired_efficiency_probe"
    assert result["plan"]==json.loads((OUT/"plan.json").read_text())
    source_path=REPO_OUT/"model_run"/"results.json"
    assert hashlib.sha256(source_path.read_bytes()).hexdigest()==result["plan"]["source_sha256"]
    source=json.loads(source_path.read_text())
    data,_,_=load()
    assert replay_bundles(source,data)==result["bundles"]
    parser=native_parser()
    expected={(task,turn) for task,turns in result["plan"]["selection"].items() for turn in turns}
    assert {(p["task"],p["turn"]) for p in result["pairs"]}==expected
    assert len(result["pairs"])==len(expected)
    rows,checks=[],[]
    for pair in result["pairs"]:
        a,b=pair["arms"]["fresh"],pair["arms"]["cached"]
        matched=all(a[k]==b[k] for k in ("raw_response","token_ids","finish_reason"))
        assert pair["matched"]==matched
        try: action_equal=parse_calls(a["raw_response"],parser)==parse_calls(b["raw_response"],parser)
        except (ValueError,KeyError,TypeError): action_equal=False
        checks.append(dict(task=pair["task"],turn=pair["turn"],exact_match=matched,action_match=action_equal))
        rows.append(f'| {pair["task"]}:{pair["turn"]} | {"PASS" if matched else "FAIL"} | '
                    f'{"PASS" if action_equal else "FAIL"} | {a["elapsed_seconds"]:.2f} | '
                    f'{b["elapsed_seconds"]:.2f} | {b["cache_metrics"]["reused_tokens"]} |')
    gate=all(c["exact_match"] for c in checks)
    assert gate==result["cache_gate_passed"]
    matched=[p for p in result["pairs"] if p["matched"]]
    fresh=sum(p["arms"]["fresh"]["elapsed_seconds"] for p in matched)
    cached=sum(p["arms"]["cached"]["elapsed_seconds"] for p in matched)
    matched_tokens=sum(len(p["arms"]["fresh"]["token_ids"]) for p in matched)
    audit=dict(results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checks=checks,
               bundles_replayed=len(result["bundles"]),cache_gate_passed=gate,
               exact_matching_pairs=len(matched),total_pairs=len(checks),
               action_matching_pairs=sum(c["action_match"] for c in checks),
               matched_subset_generated_tokens=matched_tokens,
               matched_subset_fresh_seconds=fresh,matched_subset_cached_seconds=cached,
               matched_subset_speed_ratio=fresh/cached,
               peak_mlx_bytes=max(a["peak_mlx_bytes"] for p in result["pairs"] for a in p["arms"].values()))
    (OUT/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    summary=f"""# Repository harness efficiency probe

No training. Old repository results and score remain unchanged. This is a fixed-history
microbenchmark, not a new live repair run or a quality improvement.

Bundled reads: **{audit['bundles_replayed']}/{audit['bundles_replayed']}** historical rejected bundles now parse and
execute with exactly the same responses as individual reads, without modifying files.
Only read_file calls may be bundled (at most eight). Edits/tests/finish cannot be mixed in.
This fixed-history probe is not a live profile benchmark. Live results, when available,
are recorded separately under `repository_v1/bundled_run/` and `repository_v1/comparison.md`.

Prefix-cache exact-output gate: **{'PASS' if gate else 'FAIL'}**, {len(matched)}/{len(checks)} exact token/text/stop matches.
Parsed-action equality: {audit['action_matching_pairs']}/{len(checks)}, a separate diagnostic, NOT a replacement gate.
Cache reuse remains **disabled in the repair runner**. The cause of divergence has not
yet been isolated; numerical changes from prefill segmentation are a hypothesis, not a finding.

| Fixed prompt | Exact output | Parsed actions | Fresh seconds | Cached seconds | Reused prefix tokens |
|---|---|---|---:|---:|---:|
{chr(10).join(rows)}

On the {len(matched)} exact-matching pairs ONLY, total fresh time is {fresh:.2f}s and cached time
is {cached:.2f}s: {fresh/cached:.2f}x ratio. This is a postselected diagnostic subset, not a
full-workload speedup or evidence that the failed method should be deployed. Different
outputs make the remaining comparisons unequal work. One measurement per arm, alternating
order, shared warmup, no confidence interval. Timing includes prefix preparation/copying.
Peak MLX allocation across arms: {audit['peak_mlx_bytes']/1024**3:.2f}GiB; persistent prefix state may be resident
during both arms, so this is not isolated per-method memory measurement.

The cache stores canonical history before the generation header, requires an exact token
prefix, resets on rewrites/shortening, and branches generation into a copy. Recurrent
state is never trimmed. Changed tokens/text count as failure even when code edits agree.
`plan.json` fixes nine historical prompts before inference; raw tokens/text, timings,
prefix counts and mismatches are retained in `results.json`. `audit.json` independently
recomputes equality/timing claims and reexecutes bundled reads on reconstructed files.
"""
    (OUT/"summary.md").write_text(summary)
    print(summary)


if __name__=="__main__": main()
