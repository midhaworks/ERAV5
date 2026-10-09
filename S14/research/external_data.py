"""Pin an externally authored evaluation slice without performance-based selection."""
import gzip
import hashlib
import json
import ssl
import urllib.request
from pretrained_baseline import ROOT
from tool_feedback import REVISION
from frozen_suite import digest

OUT=ROOT/"artifacts"/"external_v1"
SOURCE_REVISION="6d43fb980f9fee3c892a914eda09951f772ad10d"
BASE=f"https://raw.githubusercontent.com/openai/human-eval/{SOURCE_REVISION}"
PROTOCOL=dict(selection="HumanEval/0 through HumanEval/7, numeric ID order; no task substitution",
    task_ids=[f"HumanEval/{i}" for i in range(8)],model_revision=REVISION,
    max_tokens=512,temperature=0,enable_thinking=False,
    metric="one greedy complete-function generation per task; unchanged official tests",
    scope="eight-task subset, not full HumanEval, SOTA, agentic or production evaluation",
    contamination="externally authored, but public since 2021; pretraining contamination unknown",
    safety="macOS sandbox required before executing candidates; no unrestricted fallback")


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    upstream=OUT/"upstream"
    upstream.mkdir(exist_ok=True)
    records=[]
    for remote,name in [("data/HumanEval.jsonl.gz","HumanEval.jsonl.gz"),("LICENSE","LICENSE")]:
        path=upstream/name
        url=f"{BASE}/{remote}"
        if not path.exists():
            with urllib.request.urlopen(url,context=ssl.create_default_context(),timeout=30) as response:
                payload=response.read(2*1024**2)
            path.write_bytes(payload)
        records.append(dict(path=name,url=url,bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    rows=[json.loads(line) for line in gzip.decompress((upstream/"HumanEval.jsonl.gz").read_bytes()).decode().splitlines()]
    by_id={r["task_id"]:r for r in rows}
    data=[by_id[i] for i in PROTOCOL["task_ids"]]
    manifest=dict(source="https://github.com/openai/human-eval",source_revision=SOURCE_REVISION,
        license="MIT",license_file="upstream/LICENSE",upstream_files=records,
        source_task_count=len(rows),selected_task_count=len(data),
        tasks_sha256=digest(data),protocol_sha256=digest(PROTOCOL))
    files=[("tasks.json",data),("protocol.json",PROTOCOL),("manifest.json",manifest)]
    for name,value in files:
        path=OUT/name
        if path.exists() and json.loads(path.read_text())!=value: raise ValueError("Frozen external data changed")
    for name,value in files:
        path=OUT/name
        if not path.exists(): path.write_text(json.dumps(value,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


def load(for_training=False):
    if for_training: raise ValueError("External benchmark data is evaluation-only")
    data=json.loads((OUT/"tasks.json").read_text())
    protocol=json.loads((OUT/"protocol.json").read_text())
    manifest=json.loads((OUT/"manifest.json").read_text())
    if digest(data)!=manifest["tasks_sha256"] or digest(protocol)!=manifest["protocol_sha256"]:
        raise ValueError("Frozen external data hash mismatch")
    for record in manifest["upstream_files"]:
        if hashlib.sha256((OUT/"upstream"/record["path"]).read_bytes()).hexdigest()!=record["sha256"]:
            raise ValueError("Upstream byte hash mismatch")
    return data,protocol,manifest


if __name__=="__main__": prepare()
