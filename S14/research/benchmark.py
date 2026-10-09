"""Small byte LM experiment. Synthetic tasks; oracle role masks; no 30B claim."""
import argparse
import ast
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from grouped_moe import GroupedMoE, parameters

PAD, EOS, VOCAB = 256, 257, 258
CONTEXT = 160


def tasks():
    # Split by (operation, constant), not individual duplicated prompts.
    records = []
    for role in ("coding", "agentic"):
        for operation in ("+", "-", "*"):
            for number in range(12):
                code = f"def solve(x):\n return x{operation}{number}\n"
                if role == "coding":
                    prompt = f"Write Python solve(x) returning x{operation}{number}.\n"
                    answer = code
                else:
                    prompt = (f"File solve.py returns x. Test requires x{operation}{number}.\n"
                              "Emit JSON edit action with tool and code.\n")
                    answer = json.dumps({"tool": "edit", "code": code}, separators=(",", ":"))
                records.append(dict(role=role, operation=operation, number=number,
                                    prompt=prompt, answer=answer,
                                    split="test" if ("+", "-", "*").index(operation) == number % 3 else "train"))
    return records


def encode(record):
    prompt = list(record["prompt"].encode())
    response = list(record["answer"].encode()) + [EOS]
    stream = prompt + response
    if len(stream) > CONTEXT:
        raise ValueError("Task exceeds context")
    inputs, labels = stream[:-1], stream[1:]
    labels = [-100 if i < len(prompt) - 1 else token for i, token in enumerate(labels)]
    return inputs, labels


def batch(records):
    encoded = [encode(r) for r in records]
    length = max(len(x) for x, _ in encoded)
    x = torch.full((len(records), length), PAD, dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, (tokens, targets) in enumerate(encoded):
        x[i, :len(tokens)] = torch.tensor(tokens)
        y[i, :len(targets)] = torch.tensor(targets)
    return x, y


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(),
                                 nn.Linear(width * 2, width))

    def forward(self, x, group):
        normalized = self.norm1(x)
        mask = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool).triu(1)
        x = x + self.attn(normalized, normalized, normalized,
                          attn_mask=mask, need_weights=False)[0]
        normalized = self.norm2(x)
        return x + (self.ffn(normalized, group) if isinstance(self.ffn, GroupedMoE)
                    else self.ffn(normalized))


class LM(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.token = nn.Embedding(VOCAB, width)
        self.position = nn.Embedding(CONTEXT, width)
        self.blocks = nn.ModuleList([Block(width) for _ in range(2)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, VOCAB, bias=False)
        self.head.weight = self.token.weight

    def forward(self, x, group=None):
        h = self.token(x) + self.position(torch.arange(x.shape[1]))
        for block in self.blocks:
            h = block(h, group)
        return self.head(self.norm(h))

    def upcycle(self):
        model = copy.deepcopy(self)
        for block in model.blocks:
            experts = [copy.deepcopy(block.ffn) for _ in range(4)]
            width = block.norm1.normalized_shape[0]
            block.ffn = GroupedMoE(experts, nn.Linear(width, 4),
                                   {"coding": (0, 1), "agentic": (2, 3)}, top_k=2)
        return model

    def export(self, role):
        model = copy.deepcopy(self)
        for block in model.blocks:
            block.ffn = block.ffn.export(role)
        return model


def nll(model, records, grouped=False):
    model.eval()
    total, count = 0., 0
    with torch.no_grad():
        for role in ("coding", "agentic"):
            subset = [r for r in records if r["role"] == role]
            if not subset:
                continue
            x, y = batch(subset)
            logits = model(x, role if grouped else None)
            total += F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                                     reduction="sum").item()
            count += (y != -100).sum().item()
    return total / count


def train(model, records, steps, seed, grouped=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    rng = random.Random(seed)
    trace, target_tokens = [], 0
    started = time.perf_counter()
    for step in range(steps):
        model.train()
        role = ("coding", "agentic")[step % 2]
        selected = rng.choices([r for r in records if r["role"] == role], k=8)
        x, y = batch(selected)
        logits = model(x, role if grouped else None)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        target_tokens += int((y != -100).sum())
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            trace.append({"step": step + 1, "loss": loss.item()})
            print(f"step={step+1}/{steps} response_nll={loss.item():.4f}", flush=True)
    elapsed = time.perf_counter() - started
    return dict(trace=trace, elapsed_seconds=elapsed, target_tokens=target_tokens,
                target_tokens_per_second=target_tokens / elapsed)


def generate(model, prompt, role=None):
    tokens = list(prompt.encode())
    response = []
    model.eval()
    with torch.no_grad():
        for _ in range(min(80, CONTEXT - len(tokens))):
            token = int(model(torch.tensor([tokens]), role)[0, -1].argmax())
            if token == EOS:
                break
            if token > 255:
                return "<invalid special token>"
            response.append(token)
            tokens.append(token)
    return bytes(response).decode("utf-8", errors="replace")


def check_response(record, response):
    # Restricted AST interpretation executes only arithmetic. No generated code exec.
    try:
        if record["role"] == "agentic":
            action = json.loads(response)
            if set(action) != {"tool", "code"} or action["tool"] != "edit":
                return False
            response = action["code"]
        tree = ast.parse(response)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            return False
        fn = tree.body[0]
        if fn.name != "solve" or len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return):
            return False
        if (fn.decorator_list or fn.returns or fn.args.posonlyargs or fn.args.kwonlyargs
                or fn.args.defaults or fn.args.kw_defaults or fn.args.vararg or fn.args.kwarg):
            return False
        if len(fn.args.args) != 1 or fn.args.args[0].arg != "x":
            return False
        if fn.args.args[0].annotation:
            return False
        expr = fn.body[0].value
        def value(node, x):
            if isinstance(node, ast.Name) and node.id == "x":
                return x
            if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 100:
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult):
                a, b = value(node.left, x), value(node.right, x)
                return {ast.Add: lambda: a+b, ast.Sub: lambda: a-b,
                        ast.Mult: lambda: a*b}[type(node.op)]()
            raise ValueError("Unsupported expression")
        for x in (-9, -1, 0, 2, 13):
            n = record["number"]
            expected = {"+": x+n, "-": x-n, "*": x*n}[record["operation"]]
            if value(expr, x) != expected:
                return False
        return True
    except (ValueError, SyntaxError, TypeError, KeyError, RecursionError):
        return False


def evaluate(model, records, role_mask=False):
    started = time.perf_counter()
    traces = []
    for record in records:
        response = generate(model, record["prompt"], record["role"] if role_mask else None)
        traces.append({**record, "response": response,
                       "passed": check_response(record, response)})
    return dict(response_nll=nll(model, records, role_mask),
                success_by_role={role: sum(t["passed"] for t in traces if t["role"] == role) /
                                 sum(t["role"] == role for t in traces)
                                 for role in {r["role"] for r in records}},
                elapsed_seconds=time.perf_counter()-started, traces=traces)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    records = tasks()
    training = [r for r in records if r["split"] == "train"]
    test = [r for r in records if r["split"] == "test"]
    out = Path(__file__).parent / "artifacts" / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(records, sort_keys=True).encode()
    (out / "tasks.json").write_bytes(payload)
    dense = LM()
    result = {"status": "synthetic_pilot", "seed": args.seed, "steps_per_phase": args.steps,
              "data_sha256": hashlib.sha256(payload).hexdigest(), "torch": torch.__version__,
              "device": "cpu", "comparisons": "token-matched, not compute-matched",
              "routing": "explicit oracle role mask, no learned group selector", "runs": {}}
    print("Pretraining dense model", flush=True)
    result["pretraining"] = train(dense, training, args.steps, args.seed)
    for name in ("dense_continuation", "unrestricted_moe", "grouped_moe"):
        print(name, flush=True)
        torch.manual_seed(args.seed + 1)
        model = copy.deepcopy(dense) if name == "dense_continuation" else dense.upcycle()
        grouped = name == "grouped_moe"
        torch.manual_seed(args.seed + 1)
        run = {"parameters": parameters(model), "before_train_nll": nll(model, training, grouped)}
        run["training"] = train(model, training, args.steps, args.seed + 1, grouped)
        run["after_train_nll"] = nll(model, training, grouped)
        run["test"] = evaluate(model, test, grouped)
        torch.save(model.state_dict(), out / f"{name}.pt")
        if grouped:
            run["exports"] = {}
            for role in ("coding", "agentic"):
                exported = model.export(role)
                subset = [r for r in test if r["role"] == role]
                x, _ = batch(subset)
                with torch.no_grad():
                    error = float((model(x, role) - exported(x)).abs().max())
                run["exports"][role] = {"parameters": parameters(exported),
                    "masked_parent_max_logit_error": error, "test": evaluate(exported, subset)}
                torch.save(exported.state_dict(), out / f"export_{role}.pt")
        result["runs"][name] = run
        result["completed_runs"] = list(result["runs"])
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"name": name, "train_nll": run["after_train_nll"],
                          "test_success": run["test"]["success_by_role"]}), flush=True)


if __name__ == "__main__":
    main()
