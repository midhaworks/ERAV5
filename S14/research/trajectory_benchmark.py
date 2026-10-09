"""Compute-proxy matched synthetic read/edit/test benchmark.

The FLOP proxy includes dense attention, FFNs, router and vocabulary projection.
It excludes optimizer, nonlinearities, dispatch and hardware effects; wall time is
reported separately. Role masks remain oracle labels.
"""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from benchmark import LM, batch, check_response, generate, nll
from grouped_moe import GroupedMoE, parameters


def cases():
    records = []
    for op_index, operation in enumerate(("+", "-", "*")):
        for number in range(12):
            split = "test" if op_index == number % 3 else (
                "validation" if number % 4 == 0 and op_index == (number + 1) % 3 else "train")
            records.append(dict(operation=operation, number=number, split=split))
    return records


def prompt(case, phase, code=None):
    goal = f'x{case["operation"]}{case["number"]}'
    if phase == "coding":
        return f"Write Python solve(x) returning {goal}.\n"
    if phase == "read":
        return f"Repair solve.py to return {goal}.\nNext JSON action?\n"
    if phase == "edit":
        return f"Goal {goal}. Read solve.py: def solve(x): return x\nNext JSON action?\n"
    return f"Goal {goal}. Edited solve.py:\n{code}Next JSON action?\n"


def reference_code(case):
    return f'def solve(x):\n return x{case["operation"]}{case["number"]}\n'


def training_records(task_cases):
    out = []
    for case in task_cases:
        code = reference_code(case)
        for phase in ("coding", "read", "edit", "test"):
            answer = code if phase == "coding" else json.dumps(
                {"tool": "edit", "code": code} if phase == "edit" else {"tool": phase},
                separators=(",", ":"))
            out.append({**case, "role": "coding" if phase == "coding" else "agentic",
                        "phase": phase, "prompt": prompt(case, phase, code), "answer": answer})
    return out


def flops_proxy(model, x):
    """Approximate fwd+backward FLOPs, 2 FLOPs/MAC and backward=2*fwd.

    Attention is charged as a full LxL matrix including masked positions, matching
    the reference dense attention implementation's conceptual padded computation.
    """
    b, length = x.shape
    width = model.token.embedding_dim
    macs = b * length * width * model.head.out_features
    for block in model.blocks:
        macs += 4*b*length*width*width + 2*b*length*length*width
        if isinstance(block.ffn, GroupedMoE):
            hidden = block.ffn.experts[0][0].out_features
            macs += b*length*width*len(block.ffn.experts)
            macs += 2*b*length*width*hidden*block.ffn.top_k
        else:
            hidden = block.ffn[0].out_features
            macs += 2*b*length*width*hidden
    return int(6 * macs)


def batches(records, seed):
    rng = random.Random(seed)
    step = 0
    while True:
        role = ("coding", "agentic")[step % 2]
        subset = [r for r in records if r["role"] == role]
        yield batch(rng.choices(subset, k=8)), role
        step += 1


def budget(model, records, steps, seed):
    stream = batches(records, seed)
    return sum(flops_proxy(model, next(stream)[0][0]) for _ in range(steps))


def fit(model, records, limit, seed, lr, grouped=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    spent, targets, positions, step = 0, 0, 0, 0
    trace = []
    started = time.perf_counter()
    for (x, y), role in batches(records, seed):
        cost = flops_proxy(model, x)
        if spent + cost > limit:
            break
        model.train()
        loss = F.cross_entropy(model(x, role if grouped else None).flatten(0, 1), y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        step += 1
        spent += cost
        targets += int((y != -100).sum())
        positions += x.numel()
        if step == 1 or step % 100 == 0:
            trace.append(dict(step=step, loss=float(loss.detach()), flops_proxy=spent))
    elapsed = time.perf_counter() - started
    return dict(steps=step, flops_proxy=spent, budget=limit, budget_fraction=spent/limit,
                target_tokens=targets, padded_positions=positions, elapsed_seconds=elapsed,
                target_tokens_per_second=targets/elapsed, trace=trace)


def trajectory(case, responder):
    """Real state transitions; generated edit content is used by the test tool."""
    events = []
    code = None
    for phase in ("read", "edit", "test"):
        text = prompt(case, phase, code)
        response = responder(text)
        event = dict(phase=phase, prompt=text, response=response, accepted=False)
        events.append(event)
        try:
            action = json.loads(response)
            if not isinstance(action, dict) or action.get("tool") != phase:
                return dict(passed=False, events=events, failure="wrong_tool_or_schema")
            expected_keys = {"tool", "code"} if phase == "edit" else {"tool"}
            if set(action) != expected_keys:
                return dict(passed=False, events=events, failure="wrong_tool_or_schema")
            if phase == "edit":
                code = action["code"]
                if not isinstance(code, str) or len(code.encode()) > 60:
                    return dict(passed=False, events=events, failure="invalid_edit")
            event["accepted"] = True
            if phase == "read":
                event["tool_result"] = "def solve(x): return x"
            elif phase == "edit":
                event["tool_result"] = "edit accepted"
            else:
                passed = check_response({**case, "role": "coding"}, code)
                event["tool_result"] = "tests passed" if passed else "tests failed"
                return dict(passed=passed, events=events, failure=None if passed else "tests_failed")
        except (ValueError, TypeError):
            return dict(passed=False, events=events, failure="invalid_json")


def evaluate(model, task_cases, role_mask=False, only_role=None, selector=None,
             transform_prompt=None):
    traces = []
    routing = []
    started = time.perf_counter()
    def respond(text, expected_role):
        text = transform_prompt(text) if transform_prompt else text
        group = selector(text) if selector else (expected_role if role_mask else None)
        routing.append(dict(prompt=text, chosen_group=group, expected_role=expected_role))
        return generate(model, text, group)
    for case in task_cases:
        if only_role != "agentic":
            text = respond(prompt(case, "coding"), "coding")
            traces.append({**case, "role": "coding", "response": text,
                           "passed": check_response({**case, "role": "coding"}, text)})
        if only_role != "coding":
            result = trajectory(case, lambda p: respond(p, "agentic"))
            traces.append({**case, "role": "agentic", **result})
    scores = {}
    for role in sorted({t["role"] for t in traces}):
        selected = [t for t in traces if t["role"] == role]
        successes = sum(t["passed"] for t in selected)
        calls = sum(len(t["events"]) if role == "agentic" else 1 for t in selected)
        scores[role] = dict(passed=successes, total=len(selected), success_rate=successes/len(selected),
                            model_calls=calls, calls_per_success=calls/successes if successes else None)
    return dict(scores=scores, traces=traces, routing=routing,
                elapsed_seconds=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    data = cases()
    train = training_records([c for c in data if c["split"] == "train"])
    val = training_records([c for c in data if c["split"] == "validation"])
    test = [c for c in data if c["split"] == "test"]
    out = Path(__file__).parent / "artifacts" / "trajectories" / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(training_records(data), sort_keys=True).encode()
    (out / "tasks.json").write_bytes(payload)
    result = dict(seed=args.seed, status="synthetic_compute_proxy_pilot",
                  torch=torch.__version__, data_sha256=hashlib.sha256(payload).hexdigest(),
                  device="cpu", role_selector="oracle", config=dict(width=48, layers=2, heads=4,
                  top_k=2, experts=4, steps=args.steps, learning_rates=[0.001, 0.002]), runs={})
    dense = LM()
    result["pretraining"] = fit(dense, train, budget(dense, train, args.steps, args.seed),
                                args.seed, 0.002)
    continuation_budget = budget(dense, train, args.steps, args.seed+1)
    for name in ("dense_continuation", "unrestricted_moe", "grouped_moe"):
        print(f'seed={args.seed} {name} tuning', flush=True)
        grouped = name == "grouped_moe"
        torch.manual_seed(args.seed+1)
        initial = copy.deepcopy(dense) if name == "dense_continuation" else dense.upcycle()
        best, best_score, trials = None, float("inf"), []
        for lr in (0.001, 0.002):
            model = copy.deepcopy(initial)
            training = fit(model, train, continuation_budget, args.seed+1, lr, grouped)
            score = nll(model, val, grouped)
            trial = dict(lr=lr, validation_nll=score, training=training)
            trials.append(trial)
            print(f'lr={lr} steps={training["steps"]} validation_nll={score:.4f}', flush=True)
            if score < best_score:
                best, best_score, selected = model, score, trial
        run = dict(parameters=parameters(best), trials=trials, selected_lr=selected["lr"],
                   selected_training=selected["training"], validation_nll=best_score,
                   train_nll=nll(best, train, grouped), test=evaluate(best, test, grouped))
        run["total_search_flops_proxy"] = sum(t["training"]["flops_proxy"] for t in trials)
        torch.save(best.state_dict(), out / f'{name}.pt')
        if grouped:
            run["exports"] = {}
            for role in ("coding", "agentic"):
                exported = best.export(role)
                x, _ = batch([r for r in val if r["role"] == role])
                with torch.no_grad():
                    error = float((best(x, role)-exported(x)).abs().max())
                run["exports"][role] = dict(parameters=parameters(exported),
                    masked_parent_max_logit_error=error, test=evaluate(exported, test, only_role=role))
                torch.save(exported.state_dict(), out / f'export_{role}.pt')
        result["runs"][name] = run
        (out / "results.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(run["test"]["scores"]), flush=True)


if __name__ == "__main__":
    main()
