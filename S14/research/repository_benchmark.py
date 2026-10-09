"""Run the frozen miniature repository suite with a cached MLX checkpoint."""
import importlib.metadata
import argparse
import json
import time

from pretrained_baseline import MODEL,CACHE
from tool_feedback import REVISION,parse_call
from repository_suite import OUT,load as load_suite
from repository_environment import Environment,TOOLS,initial_messages


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--bundled-reads",action="store_true",help="Separate read-only bundling profile; no prefix-cache reuse")
    args=cli.parse_args()
    data,protocol,manifest=load_suite()
    if protocol["revision"]!=REVISION: raise ValueError("Frozen revision mismatch")
    if args.bundled_reads:
        from repository_efficiency import parse_calls,execute_calls
        protocol=dict(protocol,tool_policy="up to eight read_file calls per turn; other tools single",prefix_cache=False)
    out=OUT/("bundled_run" if args.bundled_reads else "model_run")
    if (out/"results.json").exists(): raise FileExistsError("Preserving existing full-run evidence")
    path=CACHE/"hub"/"models--mlx-community--Qwen3.5-4B-4bit"/"snapshots"/REVISION
    if not path.exists(): raise FileNotFoundError("Cached pinned model required")
    import mlx.core as mx
    from mlx_lm import load,stream_generate
    from mlx_lm.sample_utils import make_sampler
    mx.set_memory_limit(5*1024**3)
    mx.set_cache_limit(256*1024**2)
    model,tokenizer=load(str(path))
    mx.eval(model.parameters())
    if not tokenizer.has_tool_calling: raise RuntimeError("Native tools unavailable")
    out.mkdir(parents=True,exist_ok=True)
    (out/"plan.json").write_text(json.dumps(dict(protocol=protocol,manifest=manifest),indent=2)+"\n")
    result=dict(status="running",model=MODEL,revision=REVISION,protocol=protocol,manifest=manifest,
                bundled_reads=args.bundled_reads,
                packages={p:importlib.metadata.version(p) for p in ("mlx","mlx-lm","transformers")},
                tools=TOOLS,episodes=[])
    for task in data:
        with Environment(task) as env:
            messages=initial_messages(task,args.bundled_reads)
            episode=dict(task=task["id"],initial_messages=messages.copy(),actions=[])
            started=time.perf_counter()
            for turn in range(protocol["max_calls"]):
                prompt=tokenizer.apply_chat_template(messages,tools=TOOLS,tokenize=False,
                    add_generation_prompt=True,enable_thinking=protocol["enable_thinking"])
                tick=time.perf_counter()
                chunks,last=[],None
                for response in stream_generate(model,tokenizer,prompt=prompt,max_tokens=protocol["max_tokens"],
                                                sampler=make_sampler(temp=protocol["temperature"])):
                    chunks.append(response.text)
                    last=response
                raw="".join(chunks)
                action=dict(turn=turn+1,raw_response=raw,elapsed_seconds=time.perf_counter()-tick,
                    generated_tokens=last.generation_tokens if last else 0,prompt_tokens=last.prompt_tokens if last else 0,
                    finish_reason=last.finish_reason if last else None)
                try:
                    if args.bundled_reads:
                        calls=parse_calls(raw,tokenizer.tool_parser)
                        feedback=execute_calls(env,calls)
                        action["calls"]=calls
                    else:
                        call=parse_call(raw,tokenizer.tool_parser,TOOLS)
                        calls=[call]
                        feedback=env.execute(call["name"],call["arguments"])
                        action["call"]=call
                    messages.append(dict(role="assistant",content="",tool_calls=[dict(type="function",function=c) for c in calls]))
                except (ValueError,KeyError,TypeError) as error:
                    feedback=dict(error=f"invalid_tool_call: {error}")
                    messages.append(dict(role="assistant",content=raw))
                action["feedback"]=feedback
                episode["actions"].append(action)
                responses=feedback if "calls" in action else [feedback]
                for item in responses: messages.append(dict(role="tool",content=json.dumps(item)))
                names=[c["name"] for c in action.get("calls",[action.get("call",{"name":"invalid"})])]
                print(f'{task["id"]} {turn+1}: {names} {feedback}',flush=True)
                if env.finished: break
            episode.update(env.grade())
            episode.update(final_files=env.snapshot(),elapsed_seconds=time.perf_counter()-started,
                           peak_mlx_bytes=mx.get_peak_memory())
            result["episodes"].append(episode)
            (out/"results.json").write_text(json.dumps(result,indent=2)+"\n")
            print(f'{task["id"]}: hidden-graded completion={episode["passed"]}',flush=True)
    result["status"]="completed_frozen_repository_evaluation"
    result["score"]=dict(passed=sum(e["passed"] for e in result["episodes"]),total=len(data))
    (out/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["score"]),flush=True)


if __name__=="__main__": main()
