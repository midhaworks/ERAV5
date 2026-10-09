"""Audited single-run comparison, not a matched-output microbenchmark."""
import hashlib
import json
from repository_suite import OUT


def metrics(result):
    episodes=result["episodes"]
    actions=[a for e in episodes for a in e["actions"]]
    calls=[c for a in actions for c in a.get("calls",[a["call"]] if "call" in a else [])]
    generation=sum(a["elapsed_seconds"] for a in actions)
    tokens=sum(a["generated_tokens"] for a in actions)
    return dict(passed=sum(e["passed"] for e in episodes),total=len(episodes),turns=len(actions),
        accepted_tool_calls=len(calls),read_calls=sum(c["name"]=="read_file" for c in calls),
        bundled_read_turns=sum(len(a.get("calls",[]))>1 for a in actions),
        protocol_invalid_turns=sum("call" not in a and "calls" not in a for a in actions),
        generated_tokens=tokens,generation_seconds=generation,tokens_per_second=tokens/generation,
        episode_seconds=sum(e["elapsed_seconds"] for e in episodes),
        peak_mlx_gib=max(e["peak_mlx_bytes"] for e in episodes)/1024**3)


def main():
    records={}
    hashes={}
    for name,folder in [("serialized","model_run"),("bundled","bundled_run")]:
        path=OUT/folder/"results.json"
        result=json.loads(path.read_text())
        audit=json.loads((path.parent/"audit.json").read_text())
        assert result["status"]=="completed_frozen_repository_evaluation"
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        assert audit["results_sha256"]==sha and all(e["replay_matched"] for e in audit["episodes"])
        assert result["score"]==audit["score"]
        records[name]=result
        hashes[name]=sha
    old,new=records["serialized"],records["bundled"]
    assert old["manifest"]==new["manifest"] and old["revision"]==new["revision"] and old["model"]==new["model"]
    assert old["packages"]==new["packages"] and old["tools"]==new["tools"]
    assert new["protocol"]==dict(old["protocol"],tool_policy="up to eight read_file calls per turn; other tools single",prefix_cache=False)
    assert new["bundled_reads"] and not old.get("bundled_reads",False)
    assert [e["task"] for e in old["episodes"]]==[e["task"] for e in new["episodes"]]
    values={name:metrics(result) for name,result in records.items()}
    a,b=values["serialized"],values["bundled"]
    comparison=dict(scope="single observed run per profile on reused authored development repositories",
        source_sha256=hashes,metrics=values,
        observed_episode_time_ratio=a["episode_seconds"]/b["episode_seconds"],
        observed_episode_time_reduction=1-b["episode_seconds"]/a["episode_seconds"],
        same_tasks_and_model=True,prefix_cache_enabled=False,
        causal_kernel_speedup_claim=False,production_qualified=False)
    (OUT/"comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
    rows=[]
    for metric,label,fmt in [("turns","Model turns","d"),("accepted_tool_calls","Accepted tool calls","d"),
        ("protocol_invalid_turns","Protocol-invalid turns","d"),("bundled_read_turns","Bundled read turns","d"),
        ("generated_tokens","Generated tokens","d"),("generation_seconds","Generation seconds",".2f"),
        ("episode_seconds","Episode seconds incl. tools",".2f"),("tokens_per_second","Generated tokens/s",".2f"),
        ("peak_mlx_gib","Peak MLX allocation (GiB)",".2f")]:
        rows.append(f'| {label} | {a[metric]:{fmt}} | {b[metric]:{fmt}} |')
    tasks=[]
    for previous,current in zip(old["episodes"],new["episodes"]):
        tasks.append(f'| {current["task"]} | {"PASS" if previous["passed"] else "FAIL"} | '
                     f'{"PASS" if current["passed"] else "FAIL"} | {len(previous["actions"])} | {len(current["actions"])} |')
    summary=f"""# Serialized versus bundled-read repair profiles

Same pinned model, package versions, three task repositories and generation limits.
Bundled mode permits up to eight read_file calls in a turn; mutations/tests/finish remain
single. Reads execute sequentially; this reduces model turns, not filesystem concurrency.
Prefix-cache reuse is **disabled** because its separate exact-output gate failed.

Hidden-graded completion: serialized **{a['passed']}/{a['total']}**, bundled **{b['passed']}/{b['total']}**.
Both raw-action/file/test audits independently replay.

| Repository | Serialized | Bundled | Serialized turns | Bundled turns |
|---|---|---|---:|---:|
{chr(10).join(tasks)}

| Measured quantity | Serialized | Bundled |
|---|---:|---:|
{chr(10).join(rows)}

Observed episode-time ratio: **{comparison['observed_episode_time_ratio']:.2f}x**;
bundled time is **{comparison['observed_episode_time_reduction']*100:.1f}%** lower in these runs.
This is a single end-to-end observation per profile, not a controlled kernel speedup or
repeatability claim. The system instruction and accepted-tool policy differ; generated
outputs, token counts and subsequent prompts can differ too. There is no confidence
interval, matched-output workload, randomized run order, or production-scale conclusion.
Episode time includes inference and tools but excludes model loading. Peak MLX allocation
is cumulative within each run, not total system memory. Tasks are reused local authored
qualification fixtures, not independent held-out production evaluation or SWE-bench.

`comparison.json` derives every number from audited traces and records their hashes.
Old results are retained unchanged; the new profile lives in `bundled_run/`.
This is a harness efficiency experiment, not dense/MoE training or a model-quality gain.
"""
    (OUT/"comparison.md").write_text(summary)
    print(summary)


if __name__=="__main__": main()
