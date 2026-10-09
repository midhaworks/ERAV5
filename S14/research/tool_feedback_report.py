"""Independently replay tool-state transitions and recompute hidden grading."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

from tool_feedback import Environment, parse_call, initial_messages


def report(directory):
    path = Path(directory) / "results.json"
    result = json.loads(path.read_text())
    assert result["status"] == "completed_bounded_tool_evaluation"
    tasks = json.loads((path.parent / "tasks.json").read_text())
    assert hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest() == result["task_sha256"]
    by_id = {t["id"]: t for t in tasks}
    frozen = result.get("frozen_manifest")
    if frozen:
        from frozen_suite import load, digest
        locked_tasks, protocol, manifest = load()
        assert tasks == locked_tasks and frozen == manifest and result["frozen_protocol"] == protocol
        assert digest(tasks) == manifest["tasks_sha256"]
        assert result["revision"] == protocol["model_revision"]
        assert result["limits"]["max_calls"] == protocol["max_calls"]
        assert result["limits"]["max_tokens"] == protocol["max_tokens"]
        assert result["temperature"] == protocol["temperature"] and result["enable_thinking"] == protocol["enable_thinking"]
    parser_path = importlib.metadata.distribution("mlx-lm").locate_file("mlx_lm/tool_parsers/qwen3_coder.py")
    spec = importlib.util.spec_from_file_location("native_tool_parser", parser_path)
    parser_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser_module)
    assert len(result["episodes"]) == len(tasks)
    assert len({e["task"] for e in result["episodes"]}) == len(tasks)
    audits, rows = [], []
    for episode in result["episodes"]:
        env = Environment(by_id[episode["task"]])
        assert episode["initial_messages"] == initial_messages(by_id[episode["task"]], env)
        assert env.visible == episode["visible_tests"] and env.hidden == episode["hidden_tests"]
        for action in episode["actions"]:
            if "call" in action:
                assert parse_call(action["raw_response"], parser_module.parse_tool_call) == action["call"]
                actual = env.execute(action["call"]["name"], action["call"]["arguments"])
                assert actual == action["feedback"], (episode["task"], action["turn"])
            else:
                try:
                    parse_call(action["raw_response"], parser_module.parse_tool_call)
                except (ValueError, KeyError, TypeError):
                    pass
                else:
                    raise AssertionError("Recorded invalid call parses successfully")
        grade = env.grade()
        assert env.code == episode["final_code"]
        assert all(episode[k] == v for k, v in grade.items())
        failures = sum(a.get("call", {}).get("name") == "run_tests" and
                       not a["feedback"].get("passed", False) for a in episode["actions"])
        malformed = sum("call" not in a for a in episode["actions"])
        audits.append(dict(task=episode["task"], state_replay_matched=True, **grade))
        rows.append(f'| {episode["task"]} | {"PASS" if grade["passed"] else "FAIL"} | '
                    f'{len(episode["actions"])} | {failures} | {malformed} |')
    passed = sum(a["passed"] for a in audits)
    assert result["score"] == dict(passed=passed, total=len(tasks))
    tokens = sum(a["generated_tokens"] for e in result["episodes"] for a in e["actions"])
    generation_seconds = sum(a["elapsed_seconds"] for e in result["episodes"] for a in e["actions"])
    total_seconds = sum(e["elapsed_seconds"] for e in result["episodes"])
    peak = max(e["peak_mlx_bytes"] for e in result["episodes"])
    recovered = sum(e["passed"] and any(
        a.get("call", {}).get("name") == "run_tests" and not a["feedback"].get("passed", False)
        for a in e["actions"]) for e in result["episodes"])
    audit = dict(results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), episodes=audits,
                 generated_tokens=tokens, generation_seconds=generation_seconds,
                 episode_seconds=total_seconds, peak_mlx_bytes=peak,
                 tasks_recovered_after_visible_failure=recovered,
                 tokens_per_generation_second=tokens/generation_seconds)
    (path.parent / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    data_caveat = ("These are new authored qualification families frozen before this run, with no exact "
                   "task/code overlap against the prior six local development tasks. Pretraining contamination "
                   "is unknown; this is not an independent third-party or repository-level benchmark."
                   if frozen else "This is **reused development data**, not an independent "
                   "held-out benchmark; hidden means withheld during this episode.")
    failure_rows = [f'- `{e["task"]}`: {json.dumps(e["hidden_outcome"], ensure_ascii=False)}'
                    for e in result["episodes"] if not e["passed"]]
    failure_details = "\n".join(failure_rows) if failure_rows else "No failed tasks."
    summary = f"""# Bounded tool-feedback repair evaluation

Model: `{result['model']}`, pinned revision `{result['revision']}`.
Greedy, non-thinking; at most {result['limits']['max_calls']} turns and
{result['limits']['max_tokens']} generated tokens per turn. No training.

Hidden-graded completion: **{passed}/{len(tasks)}**. Native read/edit/test/finish tool
calls operate on a single virtual file. Edits are actually executed in an isolated,
restricted Python child. Passing tests are invalidated after every edit. Hidden tests
are evaluated only for reporting and are never returned to the model.

| Task | Completion | Turns | Failed visible test calls | Malformed calls |
|---|---|---:|---:|---:|
{chr(10).join(rows)}

Final failure records (unchanged grading):

{failure_details}

Generated tokens: {tokens}; generation wall time (including prompt processing):
{generation_seconds:.2f}s; {tokens/generation_seconds:.2f} tokens/s.
Total episode time including tools: {total_seconds:.2f}s.
Peak MLX allocation: {peak/1024**3:.2f}GiB, not total system memory.
Successful tasks with a recorded failed-visible-test/retry sequence: {recovered}.
Each turn reprocesses its complete chat history; this harness does not yet optimize
cross-turn KV-cache reuse, and speed is not a production-serving claim.

`audit.json` independently replays recorded tool-state transitions, reexecutes tests,
and checks final file contents and grading against `results.json`. Raw responses and
tool feedback are retained. {data_caveat} It is not real
repository-level agentic evaluation. Extra turns/test feedback change the inference
budget, so scores are not a compute-matched comparison to the one-shot baseline.
"""
    (path.parent / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--directory", default=str(Path(__file__).parent / "artifacts/tool_feedback/tasks_6"))
    report(cli.parse_args().directory)
