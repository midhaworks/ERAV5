"""Re-parse raw actions, replay files, and reexecute visible/hidden integration tests."""
import hashlib
import importlib.metadata
import importlib.util
import json
import argparse
from repository_suite import OUT,load
from repository_environment import Environment,TOOLS,initial_messages
from tool_feedback import parse_call


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--bundled-reads",action="store_true")
    args=cli.parse_args()
    data,protocol,manifest=load()
    if args.bundled_reads:
        from repository_efficiency import parse_calls,execute_calls
        protocol=dict(protocol,tool_policy="up to eight read_file calls per turn; other tools single",prefix_cache=False)
    out=OUT/("bundled_run" if args.bundled_reads else "model_run")
    path=out/"results.json"
    result=json.loads(path.read_text())
    assert result["status"]=="completed_frozen_repository_evaluation"
    assert result["protocol"]==protocol and result["manifest"]==manifest and result["tools"]==TOOLS
    assert result["revision"]==protocol["revision"]
    assert [e["task"] for e in result["episodes"]]==[t["id"] for t in data]
    parser_path=importlib.metadata.distribution("mlx-lm").locate_file("mlx_lm/tool_parsers/qwen3_coder.py")
    spec=importlib.util.spec_from_file_location("native_parser",parser_path)
    parser=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    audits,rows,log=[],[],["Reconstructed execution log from saved raw model actions; tests reexecuted by audit."]
    for task,episode in zip(data,result["episodes"]):
        assert episode["initial_messages"]==initial_messages(task,args.bundled_reads)
        assert len(episode["actions"])<=protocol["max_calls"]
        with Environment(task) as env:
            for number,action in enumerate(episode["actions"],1):
                assert action["turn"]==number
                assert action["generated_tokens"]<=protocol["max_tokens"]
                if args.bundled_reads and "calls" in action:
                    calls=parse_calls(action["raw_response"],parser.parse_tool_call)
                    assert calls==action["calls"]
                    assert execute_calls(env,calls)==action["feedback"]
                elif "call" in action:
                    call=parse_call(action["raw_response"],parser.parse_tool_call,TOOLS)
                    assert call==action["call"]
                    assert env.execute(call["name"],call["arguments"])==action["feedback"]
                else:
                    try:
                        if args.bundled_reads:
                            calls=parse_calls(action["raw_response"],parser.parse_tool_call)
                            execute_calls(env,calls)  # valid syntax can still violate read-only bundle rules
                        else: parse_call(action["raw_response"],parser.parse_tool_call,TOOLS)
                    except (ValueError,KeyError,TypeError) as error:
                        assert action["feedback"]==dict(error=f"invalid_tool_call: {error}")
                    else: raise AssertionError("Recorded malformed call is valid")
                log.append(f'{task["id"]} turn {number}: {json.dumps(action.get("calls",action.get("call")))} => {json.dumps(action["feedback"])}')
            assert env.snapshot()==episode["final_files"]
            grade=env.grade()
            assert all(episode[k]==v for k,v in grade.items())
        flat=[(c,f) for a in episode["actions"] for c,f in
              (zip(a["calls"],a["feedback"]) if "calls" in a else [(a.get("call",{}),a["feedback"])])]
        edited={c["arguments"]["path"] for c,f in flat if c.get("name")=="edit_file" and f.get("edited")}
        failures=sum(c.get("name")=="run_tests" and not f.get("passed",False) for c,f in flat)
        malformed=sum("call" not in a and "calls" not in a for a in episode["actions"])
        audits.append(dict(task=task["id"],replay_matched=True,edited_files=sorted(edited),
                           failed_visible_test_calls=failures,malformed_calls=malformed,**grade))
        rows.append(f'| {task["id"]} | {"PASS" if grade["passed"] else "FAIL"} | {len(episode["actions"])} | '
                    f'{len(edited)} | {failures} | {malformed} |')
        log.append(f'{task["id"]} final: {json.dumps(grade)}')
    score=dict(passed=sum(a["passed"] for a in audits),total=len(data))
    assert score==result["score"]
    tokens=sum(a["generated_tokens"] for e in result["episodes"] for a in e["actions"])
    seconds=sum(a["elapsed_seconds"] for e in result["episodes"] for a in e["actions"])
    peak=max(e["peak_mlx_bytes"] for e in result["episodes"])
    audit=dict(results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),score=score,episodes=audits,
               generated_tokens=tokens,generation_seconds=seconds,tokens_per_second=tokens/seconds,
               episode_seconds=sum(e["elapsed_seconds"] for e in result["episodes"]),peak_mlx_bytes=peak)
    (out/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    (out/"run.log").write_text("\n".join(log)+"\n")
    failures="\n".join(f'- `{a["task"]}`: {json.dumps(a["hidden_outcome"])}' for a in audits if not a["passed"]) or "No failed tasks."
    policy=("Up to eight read_file calls may be bundled; other tools remain single. Cache reuse is disabled."
            if args.bundled_reads else "The profile accepts exactly one tool call per turn. Several valid native calls in a response are rejected.")
    summary=f"""# Frozen multi-file repository repair

Checkpoint: `{result['model']}`, revision `{result['revision']}`. No training or new download.
Three authored miniature repositories, each with two initial bugs in separate modules.
Limits frozen before inference: {protocol['max_calls']} turns, {protocol['max_tokens']} generated tokens per turn,
greedy non-thinking decoding. No shell or external dependencies.

Hidden-graded completion: **{score['passed']}/{score['total']}**.

| Repository | Completion | Turns | Edited modules | Failed visible calls | Protocol-invalid turns |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

{failures}

Generated tokens: {tokens}; generation time including full chat reprocessing: {seconds:.2f}s;
**{tokens/seconds:.2f} tokens/s**. Peak MLX allocation: **{peak/1024**3:.2f}GiB**, not total system memory.
Saved `audit.json` re-parses every raw call, reconstructs real temporary files, reruns visible
and hidden tests, and checks final file contents and grades. `run.log` is reconstructed
from these saved actions, not a separate original stdout capture.

Files are real and local imports execute; tests and README are read-only. Hidden integration
cases stay outside the repair workspace and never enter tool feedback. Edits invalidate a
previous pass. References pass all 24 tests, and reintroducing each of the six individual
bugs is detected before inference. Only controlled AST-limited Python is supported;
this is not a general security sandbox or arbitrary real-project environment.
{policy} The protocol-invalid count is NOT necessarily a broken-XML count.
Tool-profile restrictions are harness limitations, not a general parallel-tool capability measure.

This is local authored qualification, not SWE-bench, a third-party benchmark, or production
readiness. Pretraining contamination is unknown. There is no tuned training comparison,
MoE quality claim, or 30B scaling evidence. Cross-turn KV-cache reuse is not optimized.
"""
    (out/"summary.md").write_text(summary)
    print(summary)


if __name__=="__main__": main()
