"""Bounded single-file repair agent using the checkpoint's native tool format."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from pretrained_baseline import ROOT, MODEL, CACHE, suite, check

REVISION = "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"


def tool(name, description, properties):
    return dict(type="function", function=dict(name=name, description=description,
        parameters=dict(type="object", properties=properties, required=list(properties),
                        additionalProperties=False)))


TOOLS = [
    tool("read_file", "Read the current contents of solution.py.",
         {"path": {"type": "string"}}),
    tool("edit_file", "Replace solution.py with a complete plain Python solve function. No imports.",
         {"path": {"type": "string"}, "code": {"type": "string"}}),
    tool("run_tests", "Execute visible tests against the current file and return real feedback.", {}),
    tool("finish", "Finish after the current version passes visible tests. Hidden tests grade completion.", {}),
]


class Environment:
    """A virtual single file, executed by the isolated restricted checker."""
    def __init__(self, task):
        self.task = task
        self.code = task["faulty"]
        # Deterministic development split, NOT an independent benchmark split.
        # Balanced needs its third example to reveal the faulty implementation.
        split = task.get("visible_count", 3 if task["id"] == "balanced" else 2)
        self.visible = task["tests"][:split]
        self.hidden = task["tests"][split:]
        self.tested_code = None
        self.public_pass = False
        self.finished = False
        self.preserve_inputs = task.get("preserve_inputs", task["id"] == "rotate")

    def execute(self, name, arguments):
        if self.finished:
            return dict(error="already_finished")
        schema = next((t["function"] for t in TOOLS if t["function"]["name"] == name), None)
        if schema is None or not isinstance(arguments, dict):
            return dict(error="unknown_tool_or_invalid_arguments")
        if set(arguments) != set(schema["parameters"]["required"]):
            return dict(error="wrong_arguments")
        if any(not isinstance(value, str) for value in arguments.values()):
            return dict(error="arguments_must_be_strings")
        if "path" in arguments and arguments["path"] != "solution.py":
            return dict(error="only_solution.py_is_available")
        if name == "read_file":
            return dict(path="solution.py", code=self.code)
        if name == "edit_file":
            if len(arguments["code"]) > 8000:
                return dict(error="code_too_long")
            self.code = arguments["code"]
            self.public_pass = False
            self.tested_code = None
            return dict(edited=True, requires_fresh_tests=True)
        if name == "run_tests":
            self.tested_code = self.code
            outcome = check(self.code, self.visible, self.preserve_inputs)
            self.public_pass = outcome["passed"]
            return dict(visible_tests=len(self.visible), **outcome)
        if not self.public_pass or self.tested_code != self.code:
            return dict(error="run_tests_must_pass_on_current_file_before_finish")
        self.finished = True
        # Hidden outcomes are deliberately NOT sent back to the model.
        return dict(finished=True)

    def grade(self):
        outcome = check(self.code, self.hidden, self.preserve_inputs)
        return dict(completed=self.finished, visible_pass=self.public_pass,
                    hidden_outcome=outcome,
                    passed=self.finished and self.public_pass and outcome["passed"])


def parse_call(raw, parser, tools=TOOLS):
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", raw, re.DOTALL)
    if len(blocks) != 1:
        raise ValueError("Expected exactly one complete native tool call")
    if raw.split("</tool_call>", 1)[1].strip():
        raise ValueError("Unexpected suffix after tool call")
    return parser(blocks[0].strip(), tools=tools)


def initial_messages(task, env):
    return [dict(role="system", content="You are repairing a single virtual Python file. "
        "Use exactly one tool call per turn. Read solution.py, replace its contents, run tests, "
        "and finish only after the current file passes. The function must be named solve. "
        "No imports. Use tool feedback to correct mistakes. Hidden tests are not available."),
        dict(role="user", content=f'Repair solution.py. Requirement: {task["instruction"]}\n'
             f'Visible tests (arguments, expected): {json.dumps(env.visible)}')]


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--limit-tasks", type=int, default=6)
    cli.add_argument("--max-calls", type=int, default=10)
    cli.add_argument("--max-tokens", type=int, default=384)
    cli.add_argument("--frozen", action="store_true", help="Run the frozen six-family qualification once")
    args = cli.parse_args()
    if not 1 <= args.limit_tasks <= 6 or not 1 <= args.max_calls <= 20 or args.max_tokens < 1:
        raise ValueError("Invalid bounded evaluation limits")
    manifest, protocol = None, None
    if args.frozen:
        from frozen_suite import load
        data, protocol, manifest = load()
        if (args.limit_tasks, args.max_calls, args.max_tokens) != (6, protocol["max_calls"], protocol["max_tokens"]):
            raise ValueError("Frozen evaluation requires all six tasks and the frozen limits")
        if protocol["model_revision"] != REVISION:
            raise ValueError("Model revision differs from frozen protocol")
    else:
        data = suite()[:args.limit_tasks]
    out = (ROOT / "artifacts" / "qualification_v1" / "model_run" if args.frozen else
           ROOT / "artifacts" / "tool_feedback" / f"tasks_{args.limit_tasks}")
    if (out / "results.json").exists():
        raise FileExistsError(f"Preserving existing evidence: {out}")
    out.mkdir(parents=True, exist_ok=True)
    path = CACHE / "hub" / "models--mlx-community--Qwen3.5-4B-4bit" / "snapshots" / REVISION
    if not path.exists():
        raise FileNotFoundError("Run pretrained_baseline.py to download the pinned snapshot first")
    for task in data:
        env = Environment(task)
        assert env.hidden and not check(env.code, env.visible)["passed"]
        assert check(task["reference"], task["tests"], env.preserve_inputs)["passed"]
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    mx.set_memory_limit(5 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    model, tokenizer = load(str(path))
    mx.eval(model.parameters())
    if not tokenizer.has_tool_calling or tokenizer.tool_parser is None:
        raise RuntimeError("Checkpoint has no native tool parser")
    result = dict(status="running", model=MODEL, revision=REVISION,
                  limits=vars(args), tools=TOOLS, temperature=0, enable_thinking=False,
                  data_source=("frozen new authored qualification families; pretraining contamination unknown"
                               if args.frozen else "reused authored development tasks; hidden within episode only"),
                  frozen_manifest=manifest, frozen_protocol=protocol,
                  task_sha256=hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
                  episodes=[])
    (out / "tasks.json").write_text(json.dumps(data, indent=2) + "\n")
    for task in data:
        env = Environment(task)
        messages = initial_messages(task, env)
        episode = dict(task=task["id"], initial_messages=messages.copy(),
                       visible_tests=env.visible, hidden_tests=env.hidden, actions=[])
        started = time.perf_counter()
        for turn in range(args.max_calls):
            prompt = tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            turn_started = time.perf_counter()
            chunks, last = [], None
            for response in stream_generate(model, tokenizer, prompt=prompt,
                    max_tokens=args.max_tokens, sampler=make_sampler(temp=0)):
                chunks.append(response.text)
                last = response
            raw = "".join(chunks)
            action = dict(turn=turn+1, raw_response=raw,
                elapsed_seconds=time.perf_counter()-turn_started,
                generated_tokens=last.generation_tokens if last else 0,
                prompt_tokens=last.prompt_tokens if last else 0,
                finish_reason=last.finish_reason if last else None)
            try:
                call = parse_call(raw, tokenizer.tool_parser)
                feedback = env.execute(call["name"], call["arguments"])
                action["call"] = call
                messages.append(dict(role="assistant", content="", tool_calls=[
                    dict(type="function", function=call)]))
            except (ValueError, KeyError, TypeError) as error:
                feedback = dict(error=f"invalid_tool_call: {error}")
                messages.append(dict(role="assistant", content=raw))
            action["feedback"] = feedback
            episode["actions"].append(action)
            messages.append(dict(role="tool", content=json.dumps(feedback)))
            print(f'{task["id"]} turn {turn+1}: {action.get("call", {}).get("name", "invalid")} '
                  f'{feedback}', flush=True)
            if env.finished:
                break
        episode.update(env.grade())
        episode.update(final_code=env.code, elapsed_seconds=time.perf_counter()-started,
                       peak_mlx_bytes=mx.get_peak_memory())
        result["episodes"].append(episode)
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f'{task["id"]}: hidden-graded success={episode["passed"]}', flush=True)
    result["status"] = "completed_bounded_tool_evaluation"
    result["score"] = dict(passed=sum(e["passed"] for e in result["episodes"]), total=len(data))
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["score"]), flush=True)


if __name__ == "__main__":
    main()
