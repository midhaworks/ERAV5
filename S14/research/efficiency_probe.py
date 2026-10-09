"""Paired fresh/cached inference on fixed historical prompts; no new quality score."""
import hashlib
import importlib.metadata
import importlib.util
import json
import time

from pretrained_baseline import ROOT,MODEL,CACHE
from tool_feedback import REVISION
from repository_suite import OUT as REPO_OUT,load
from repository_environment import Environment,TOOLS
from repository_efficiency import PrefixCache,parse_calls,execute_calls

OUT=ROOT/"artifacts"/"efficiency_v1"
# Frozen selection BEFORE inference: first, mid-history edit, and final turn per repo.
SELECTED={"shipping":{1,9,13},"pagination":{1,7,11},"ledger":{1,7,11}}


def native_parser():
    path=importlib.metadata.distribution("mlx-lm").locate_file("mlx_lm/tool_parsers/qwen3_coder.py")
    spec=importlib.util.spec_from_file_location("native_parser",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_tool_call


def replay_bundles(source,data):
    parser=native_parser()
    records=[]
    for task,episode in zip(data,source["episodes"]):
        with Environment(task) as env:
            for action in episode["actions"]:
                if "call" in action:
                    call=action["call"]
                    assert env.execute(call["name"],call["arguments"])==action["feedback"]
                else:
                    # The reference environment state remains unchanged: bundled reads
                    # must return exactly the single-read results without side effects.
                    calls=parse_calls(action["raw_response"],parser)
                    assert len(calls)>1
                    before=env.snapshot()
                    results=execute_calls(env,calls)
                    singles=[env.execute(c["name"],c["arguments"]) for c in calls]
                    assert results==singles and env.snapshot()==before
                    records.append(dict(task=task["id"],turn=action["turn"],calls=calls,
                                        feedback=results,single_read_equivalent=True))
    return records


def main():
    if (OUT/"results.json").exists(): raise FileExistsError("Preserving efficiency evidence")
    source_path=REPO_OUT/"model_run"/"results.json"
    source=json.loads(source_path.read_text())
    assert source["status"]=="completed_frozen_repository_evaluation"
    data,protocol,manifest=load()
    assert source["manifest"]==manifest
    bundles=replay_bundles(source,data)
    OUT.mkdir(parents=True,exist_ok=True)
    plan=dict(selection={k:sorted(v) for k,v in SELECTED.items()},max_tokens=512,
              revision=REVISION,source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
              comparison="same fixed historical prompts; fresh versus history-only prefix cache",
              order="alternate fresh-first/cached-first by selected-prompt index",
              gate="exact token ids, text and finish reason match; report failures without enabling reuse")
    (OUT/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    import mlx.core as mx
    from mlx_lm import load as load_model,stream_generate
    from mlx_lm.sample_utils import make_sampler
    mx.set_memory_limit(5*1024**3)
    mx.set_cache_limit(256*1024**2)
    path=CACHE/"hub"/"models--mlx-community--Qwen3.5-4B-4bit"/"snapshots"/REVISION
    model,tokenizer=load_model(str(path))
    mx.eval(model.parameters())
    # Equal unscored warm-up; initialize kernels before either timed arm.
    for _ in stream_generate(model,tokenizer,prompt="Warmup",max_tokens=2,sampler=make_sampler(temp=0)): pass
    result=dict(status="running",model=MODEL,plan=plan,bundles=bundles,pairs=[])
    def encode(text):
        special=tokenizer.bos_token is None or not text.startswith(tokenizer.bos_token)
        return tokenizer.encode(text,add_special_tokens=special)
    for episode in source["episodes"]:
        messages=episode["initial_messages"].copy()
        cache=PrefixCache(model)
        for action in episode["actions"]:
            if action["turn"] in SELECTED[episode["task"]]:
                prompt=tokenizer.apply_chat_template(messages,tools=TOOLS,tokenize=False,
                    add_generation_prompt=True,enable_thinking=False)
                history=tokenizer.apply_chat_template(messages,tools=TOOLS,tokenize=False,
                    add_generation_prompt=False,enable_thinking=False)
                tokens,base=encode(prompt),encode(history)
                pair=dict(task=episode["task"],turn=action["turn"],full_prompt_tokens=tokens,
                          prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),arms={})
                order=("fresh","cached") if len(result["pairs"])%2==0 else ("cached","fresh")
                for arm in order:
                    mx.synchronize()
                    mx.reset_peak_memory()
                    tick=time.perf_counter()
                    kwargs={}
                    metrics=dict(reused_tokens=0,full_prompt_tokens=len(tokens))
                    input_tokens=tokens
                    if arm=="cached":
                        input_tokens,branch,metrics=cache.prepare(tokens,base)
                        kwargs["prompt_cache"]=branch
                    parts,ids,last=[],[],None
                    for response in stream_generate(model,tokenizer,prompt=input_tokens,
                            max_tokens=plan["max_tokens"],sampler=make_sampler(temp=0),**kwargs):
                        parts.append(response.text)
                        ids.append(response.token)
                        last=response
                    mx.synchronize()
                    pair["arms"][arm]=dict(raw_response="".join(parts),token_ids=ids,
                        finish_reason=last.finish_reason,elapsed_seconds=time.perf_counter()-tick,
                        peak_mlx_bytes=mx.get_peak_memory(),cache_metrics=metrics)
                    if arm=="cached":
                        del branch,kwargs
                a,b=pair["arms"]["fresh"],pair["arms"]["cached"]
                pair["matched"]=all(a[k]==b[k] for k in ("raw_response","token_ids","finish_reason"))
                pair["matches_historical_text"]=a["raw_response"]==action["raw_response"]
                result["pairs"].append(pair)
                (OUT/"results.json").write_text(json.dumps(result,indent=2)+"\n")
                print(f'{episode["task"]} turn {action["turn"]}: match={pair["matched"]} '
                      f'fresh={a["elapsed_seconds"]:.2f}s cached={b["elapsed_seconds"]:.2f}s '
                      f'reused={b["cache_metrics"]["reused_tokens"]}',flush=True)
            if "call" in action:
                messages.append(dict(role="assistant",content="",tool_calls=[dict(type="function",function=action["call"])]))
            else: messages.append(dict(role="assistant",content=action["raw_response"]))
            messages.append(dict(role="tool",content=json.dumps(action["feedback"])))
    result["status"]="completed_paired_efficiency_probe"
    result["cache_gate_passed"]=all(p["matched"] for p in result["pairs"])
    (OUT/"results.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__": main()
