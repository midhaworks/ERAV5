"""Generate evidence including blocked/not-run states, never invent a model score."""
import json
from external_data import OUT,load


def main():
    data,protocol,manifest=load()
    preflight=json.loads((OUT/"preflight.json").read_text())
    assert preflight["manifest"]==manifest
    checks=preflight["isolation"]["checks"]
    ready=preflight["ready"]
    memory=preflight["isolation"]["memory_guard"]["supported"]
    assert not ready or (memory and all(checks.values()))
    evidence=dict(source_integrity="PASS",selection="PASS",evaluation_only_api="implemented",
        os_isolation="PASS" if all(checks.values()) else "FAIL",
        memory_guard="PASS" if memory else "FAIL",
        reference_checks="NOT_RUN" if not preflight["tasks"] else "RECORDED_IN_PREFLIGHT",
        model_inference="NOT_RUN",model_score=None,ready=ready,manifest=manifest)
    (OUT/"evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
    rows="\n".join(f'| {name} | {"PASS" if passed else "FAIL"} |' for name,passed in checks.items())
    summary=f"""# External evaluation preparation

Source: [official HumanEval](https://github.com/openai/human-eval), MIT license retained in
`upstream/LICENSE`, pinned revision `{manifest['source_revision']}`.
Downloaded source contains {manifest['source_task_count']} tasks; selection is the first eight
numeric IDs, without filtering on model performance. Tasks, unchanged official tests,
canonical solutions and protocol are hash-frozen. Canonical solutions and tests are
evaluation-side only, never model prompt material. The loader rejects training use.

**Model inference: NOT RUN. Score: none. Ready: {ready}.**
The current host rejects the requested data-segment cap even in a trusted process
outside the child sandbox. The candidate evaluator fails closed; the memory guard is
not removed to manufacture a benchmark result. Reference execution is also not run.

| Measured OS-isolation probe | Result |
|---|---|
{rows}
| memory_guard | {'PASS' if memory else 'FAIL'} |

The probes bypass candidate AST gates on purpose and use fake sentinel files, a local
listener, and a fake environment variable. They demonstrate these tested restrictions,
not a complete security audit. Metadata and trusted runtime-library reads are allowed;
user-file contents, writes, network, and forking were blocked in these probes. Root-directory
lookup was needed for interpreter startup; it does not permit reading descendant files.
Historical startup diagnostic results are retained separately.

Required next step: a compatible memory-limited execution backend, such as a separately
approved container/Linux executor. Candidate generation can remain on the M2, while
candidate tests run in that isolated backend. No Docker installation or external-server
coordination has been performed without direction.

HumanEval is an established 2021 benchmark, not latest SOTA or guaranteed uncontaminated
evaluation. This eight-task subset is not full HumanEval, a production gate, or an agentic
repository benchmark. Some underlying primitives overlap our local tasks; externally
authored does not mean semantically disjoint or unseen during pretraining.
"""
    (OUT/"summary.md").write_text(summary)
    print(summary)


if __name__=="__main__": main()
