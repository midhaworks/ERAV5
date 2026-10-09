"""Bounded MLX pretrained qualification. No training or automatic MoE launch."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MODEL = "mlx-community/Qwen3.5-4B-4bit"
OUT = ROOT / "artifacts" / "pretrained"
CACHE = ROOT / "model_cache"


def suite():
    return [
        dict(id="stable_unique", instruction="Remove duplicates from a list of integers, keeping first-occurrence order.",
             reference="def solve(xs):\n out=[]\n for x in xs:\n  if x not in out: out.append(x)\n return out",
             faulty="def solve(xs): return sorted(set(xs))",
             tests=[([[]], []), ([[3,1,3,2,1]], [3,1,2]), ([[0,-1,0,-1]], [0,-1])]),
        dict(id="balanced", instruction="Return whether a string of '(' and ')' has balanced parentheses.",
             reference="def solve(s):\n n=0\n for c in s:\n  n += 1 if c=='(' else -1\n  if n<0: return False\n return n==0",
             faulty="def solve(s): return s.count('(')==s.count(')')",
             tests=[([""],True),(["()()"],True),([")("],False),(["(()"],False),(["(())"],True)]),
        dict(id="first_index", instruction="Given a sorted integer list and a target, return its first index, or -1 if absent.",
             reference="def solve(xs,target):\n for i,x in enumerate(xs):\n  if x==target: return i\n return -1",
             faulty="def solve(xs,target):\n for i in range(len(xs)-1,-1,-1):\n  if xs[i]==target: return i\n return -1",
             tests=[([[],2],-1),([[1,2,2,2,4],2],1),([[1,2,4],3],-1),([[0,0],0],0)]),
        dict(id="runs", instruction="Encode consecutive equal characters as a list of [character, count] pairs.",
             reference="def solve(s):\n out=[]\n for c in s:\n  if out and out[-1][0]==c: out[-1][1]+=1\n  else: out.append([c,1])\n return out",
             faulty="def solve(s): return [[c,s.count(c)] for c in sorted(set(s))]",
             tests=[([""],[]),(["aabbaa"],[["a",2],["b",2],["a",2]]),(["xyz"],[["x",1],["y",1],["z",1]])]),
        dict(id="rotate", instruction="Rotate an integer list right by k places, preserving the input. Empty lists return []. k is nonnegative.",
             reference="def solve(xs,k):\n if not xs: return []\n k=k%len(xs)\n if k==0: return xs[:]\n return xs[-k:]+xs[:-k]",
             faulty="def solve(xs,k): return xs[k:]+xs[:k]",
             tests=[([[],3],[]),([[1,2,3],1],[3,1,2]),([[1,2,3],0],[1,2,3]),([[1,2,3],4],[3,1,2])]),
        dict(id="positive_sum", instruction="Sum only strictly positive integers in a list.",
             reference="def solve(xs): return sum(x for x in xs if x>0)",
             faulty="def solve(xs): return sum(xs)",
             tests=[([[]],0),([[-3,2,0,4]],6),([[-1,-2]],0),([[1,2,3]],6)]),
    ]


def check(code, tests, preserve_inputs=False):
    payload = json.dumps(dict(code=code, tests=tests, preserve_inputs=preserve_inputs))
    try:
        run = subprocess.run([sys.executable, "-I", str(ROOT / "restricted_check.py")],
                             input=payload, text=True, capture_output=True, timeout=5)
        return json.loads(run.stdout) if run.returncode == 0 else dict(passed=False, error="checker_process_failed")
    except (subprocess.TimeoutExpired, ValueError):
        return dict(passed=False, error="timeout_or_invalid_result")


def extract(text):
    text = text.strip()
    if text.startswith("```python\n") and text.endswith("```"):
        return text[len("```python\n"):-3].strip()
    if text.startswith("```\n") and text.endswith("```"):
        return text[4:-3].strip()
    return text


def task_prompt(task, role, style):
    if style == "direct":
        request = "Implement" if role == "coding" else "Repair"
        text = f'{request} Python solve. Requirement: {task["instruction"]}'
    else:
        text = f'Please provide a correct solve function for this behavior: {task["instruction"]}'
    if role == "repair":
        text += f'\nFaulty implementation:\n{task["faulty"]}'
    return text + "\nReturn only the complete Python function. No imports, explanations, or examples."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--limit-tasks", type=int, default=6)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()
    if args.limit_tasks < 1 or args.limit_tasks > 6:
        raise ValueError("limit-tasks must be 1..6")
    # All caches stay in the workspace; no repeated unquantized download.
    os.environ["HF_HOME"] = str(CACHE)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi, snapshot_download
    OUT.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(MODEL, revision=args.revision, files_metadata=True)
    files = [s for s in info.siblings if s.rfilename.endswith((".json", ".safetensors", ".model", ".jinja"))]
    bytes_required = sum(s.size or 0 for s in files)
    free = shutil.disk_usage(ROOT).free
    preflight = dict(model=MODEL, revision=info.sha, download_bytes=bytes_required,
                     free_disk_bytes=free, reserve_bytes=3*1024**3,
                     files=[dict(path=s.rfilename,bytes=s.size) for s in files])
    (OUT / "preflight.json").write_text(json.dumps(preflight,indent=2)+"\n")
    print(json.dumps(preflight,indent=2),flush=True)
    if args.preflight:
        return
    if free < bytes_required + preflight["reserve_bytes"]:
        raise RuntimeError("Insufficient disk for snapshot plus reserve")
    data = suite()[:args.limit_tasks]
    for task in data:
        assert check(task["reference"],task["tests"])["passed"],task["id"]
        assert not check(task["faulty"],task["tests"])["passed"],task["id"]
    path = snapshot_download(MODEL, revision=info.sha, cache_dir=str(CACHE / "hub"),
                             allow_patterns=[s.rfilename for s in files])
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    # Cap MLX's allocation budget on the 8GiB host. This is a guard, not a RAM fit claim.
    mx.set_memory_limit(5*1024**3)
    mx.set_cache_limit(256*1024**2)
    load_started = time.perf_counter()
    model, tokenizer = load(path)
    mx.eval(model.parameters())
    result = dict(status="running_validation_qualification", model=MODEL,revision=info.sha,
        quantization="community 4-bit checkpoint",max_tokens=args.max_tokens,temperature=0,
        enable_thinking=False, load_seconds=time.perf_counter()-load_started,
        data_source="original authored development validation tasks",
        data_sha256=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest(),
        packages={p:importlib.metadata.version(p) for p in ("mlx","mlx-lm","transformers")},traces=[])
    (OUT / "tasks.json").write_text(json.dumps(data,indent=2)+"\n")
    for task in data:
        for role in ("coding","repair"):
            for style in ("direct","reworded"):
                user_prompt = task_prompt(task,role,style)
                messages = [{"role":"user","content":user_prompt}]
                prompt = tokenizer.apply_chat_template(messages,tokenize=False,
                    add_generation_prompt=True,enable_thinking=False)
                started = time.perf_counter()
                chunks, last = [], None
                for response in stream_generate(model,tokenizer,prompt=prompt,max_tokens=args.max_tokens,
                                                sampler=make_sampler(temp=0)):
                    chunks.append(response.text)
                    last = response
                raw = "".join(chunks)
                elapsed = time.perf_counter()-started
                generated = last.generation_tokens if last else 0
                outcome = check(extract(raw),task["tests"])
                trace = dict(task=task["id"],role=role,style=style,prompt=user_prompt,
                    raw_response=raw,extracted_code=extract(raw),generated_tokens=generated,
                    elapsed_seconds=elapsed,tokens_per_second=generated/elapsed,
                    prompt_tokens=last.prompt_tokens if last else 0,
                    prompt_tokens_per_second=last.prompt_tps if last else None,
                    decode_tokens_per_second=last.generation_tps if last else None,
                    finish_reason=last.finish_reason if last else None,
                    reached_token_cap=generated>=args.max_tokens,peak_mlx_bytes=mx.get_peak_memory(),**outcome)
                result["traces"].append(trace)
                (OUT / "results.json").write_text(json.dumps(result,indent=2)+"\n")
                print(f'{task["id"]} {role} {style}: pass={outcome["passed"]} '
                      f'tokens={generated} seconds={elapsed:.1f} peak_MiB={mx.get_peak_memory()/1024**2:.0f}',flush=True)
    result["status"] = "completed_bounded_validation"
    result["scores"] = {}
    for role in ("coding","repair"):
        for style in ("direct","reworded"):
            traces=[t for t in result["traces"] if t["role"]==role and t["style"]==style]
            result["scores"][f"{role}_{style}"]=dict(passed=sum(t["passed"] for t in traces),total=len(traces))
    (OUT / "results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["scores"],indent=2),flush=True)


if __name__ == "__main__":
    main()
