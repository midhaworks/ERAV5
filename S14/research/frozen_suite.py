"""Prepare and validate new, frozen qualification tasks BEFORE model inference."""
import hashlib
import json
from pathlib import Path
from pretrained_baseline import ROOT, suite, check

OUT = ROOT / "artifacts" / "qualification_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def tasks():
    def task(name, instruction, reference, faulty, tests, mutants):
        return dict(id=name, family=name, instruction=instruction, reference=reference,
                    faulty=faulty, tests=tests, visible_count=3, preserve_inputs=True,
                    mutants=[faulty]+mutants, source="original authored qualification task", license="CC0-1.0")
    return [
        task("merge_intervals", "Merge closed integer intervals [start,end], including touching intervals. "
             "Input order is arbitrary, start<=end. Return sorted disjoint intervals as lists. Preserve the input.",
             "def solve(xs):\n out=[]\n for a,b in sorted(xs):\n  if out and a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)\n  else: out.append([a,b])\n return out",
             "def solve(xs): return sorted(xs)",
             [([[]],[]),([[[3,5],[1,3]]],[[1,5]]),([[[1,8],[2,4]]],[[1,8]]),
              ([[[7,9],[1,2],[2,6],[6,7]]],[[1,9]]),([[[0,0]]],[[0,0]]),
              ([[[-4,-2],[-1,1],[5,7]]],[[-4,-2],[-1,1],[5,7]]),
              ([[[2,3],[1,2],[1,2]]],[[1,3]]),([[[5,6],[1,2]]],[[1,2],[5,6]])],
             ["def solve(xs):\n out=[]\n for a,b in sorted(xs):\n  if out and a<out[-1][1]: out[-1][1]=max(out[-1][1],b)\n  else: out.append([a,b])\n return out",
              "def solve(xs):\n xs.sort()\n out=[]\n for a,b in xs:\n  if out and a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)\n  else: out.append([a,b])\n return out"]),
        task("transpose", "Transpose a rectangular integer matrix, returning lists of columns. "
             "An empty matrix or zero columns returns []. Preserve the input.",
             "def solve(xs):\n if not xs: return []\n return [[row[j] for row in xs] for j in range(len(xs[0]))]",
             "def solve(xs): return xs",
             [([[]],[]),([[[1,2],[3,4]]],[[1,3],[2,4]]),([[[1,2,3]]],[[1],[2],[3]]),
              ([[[4],[5],[6]]],[[4,5,6]]),([[[],[]]],[]),
              ([[[0,-1],[2,3],[4,5]]],[[0,2,4],[-1,3,5]]),([[[8]]],[[8]])],
             ["def solve(xs): return [[row[j] for row in xs] for j in range(len(xs))]",
              "def solve(xs): return []"]),
        task("pair_sum", "Return [i,j] for i<j with xs[i]+xs[j]==target. Among matches choose "
             "the lexicographically smallest index pair. Return [] if none. Preserve the input.",
             "def solve(xs,target):\n for i in range(len(xs)):\n  for j in range(i+1,len(xs)):\n   if xs[i]+xs[j]==target: return [i,j]\n return []",
             "def solve(xs,target):\n for i in range(len(xs)):\n  for j in range(i,len(xs)):\n   if xs[i]+xs[j]==target: return [i,j]\n return []",
             [([[],2],[]),([[3],6],[]),([[1,3,2,2],4],[0,1]),
              ([[2,2,2],4],[0,1]),([[-2,5,3,0],3],[0,1]),
              ([[1,4,2,3],5],[0,1]),([[1,2],9],[]),([[0,0],0],[0,1])],
             ["def solve(xs,target):\n for i in range(len(xs)-1,-1,-1):\n  for j in range(i+1,len(xs)):\n   if xs[i]+xs[j]==target: return [i,j]\n return []",
              "def solve(xs,target): return []"]),
        task("overlap_matches", "Return all starting indices where nonempty pattern occurs in text, "
             "including overlapping matches, in ascending order. Inputs are strings.",
             "def solve(text,pattern): return [i for i in range(len(text)-len(pattern)+1) if text[i:i+len(pattern)]==pattern]",
             "def solve(text,pattern): return list(range(text.count(pattern)))",
             [(["","a"],[]),(["aaaa","aa"],[0,1,2]),(["abcabc","bc"],[1,4]),
              (["abaaba","aba"],[0,3]),(["abc","abcd"],[]),(["ééé","éé"],[0,1]),
              (["xyz","q"],[]),(["a","a"],[0])],
             ["def solve(text,pattern): return [i for i in range(len(text)-len(pattern)) if text[i:i+len(pattern)]==pattern]",
              "def solve(text,pattern): return []"]),
        task("group_records", "Given [string_key,integer_value] records, return a dictionary mapping "
             "each key to its values in input order, preserving repeated values. Preserve the input.",
             "def solve(rows):\n out={}\n for key,value in rows:\n  if key not in out: out[key]=[]\n  out[key].append(value)\n return out",
             "def solve(rows): return {key:[value] for key,value in rows}",
             [([[]],{}),([[["a",1],["b",2],["a",3]]],{"a":[1,3],"b":[2]}),
              ([[["x",4],["x",4]]],{"x":[4,4]}),([[["",0],["",-1]]],{"": [0,-1]}),
              ([[["é",2],["a",1],["é",0]]],{"é":[2,0],"a":[1]}),
              ([[["z",9]]],{"z":[9]}),([[["b",2],["a",3],["b",1]]],{"b":[2,1],"a":[3]})],
             ["def solve(rows):\n out={}\n for key,value in rows:\n  if key not in out: out[key]=[]\n  if value not in out[key]: out[key].append(value)\n return out",
              "def solve(rows):\n out={}\n for key,value in rows:\n  if key not in out: out[key]=[]\n  out[key].append(value)\n return {k:sorted(v) for k,v in out.items()}"]),
        task("grid_neighbors", "Given rows, cols, r, c for a valid cell in a positive-size grid, "
             "return its in-bounds orthogonal neighbor coordinates [row,col], lexicographically sorted. No diagonals.",
             "def solve(rows,cols,r,c): return sorted([[a,b] for a,b in [[r-1,c],[r+1,c],[r,c-1],[r,c+1]] if 0<=a<rows and 0<=b<cols])",
             "def solve(rows,cols,r,c): return sorted([[r-1,c],[r+1,c],[r,c-1],[r,c+1]])",
             [([1,1,0,0],[]),([3,3,1,1],[[0,1],[1,0],[1,2],[2,1]]),([2,3,0,0],[[0,1],[1,0]]),
              ([1,4,0,2],[[0,1],[0,3]]),([4,1,2,0],[[1,0],[3,0]]),
              ([3,2,2,1],[[1,1],[2,0]]),([2,2,0,1],[[0,0],[1,1]])],
             ["def solve(rows,cols,r,c): return sorted([[a,b] for a in range(max(0,r-1),min(rows,r+2)) for b in range(max(0,c-1),min(cols,c+2)) if [a,b]!=[r,c]])",
              "def solve(rows,cols,r,c): return []"]),
    ]


PROTOCOL = dict(version=1, model_revision="0e7ffd5c629ef7719d4cbc04069232580bfa9d9c",
    max_calls=10, max_tokens=384, temperature=0, enable_thinking=False,
    primary_metric="tasks finished after fresh visible pass AND all hidden tests pass",
    secondary_metrics=["tool_calls", "failed_test_recoveries", "malformed_calls", "wall_seconds", "peak_mlx_bytes"],
    comparison="unchanged checkpoint; no training; no claim of matched-compute improvement",
    advancement="inspect all failures and repeat on new repository-level tasks before any scale-up",
    stop_rule="one full six-task run; no selective reruns or prompt changes for published score",
    contamination="disjoint from six local development families; pretraining contamination unknown",
    scope="single-file qualification, NOT locked final production evaluation")


def validate(data):
    old = suite()
    assert len(data)==6 and len({t['id'] for t in data})==6
    assert not {t['id'] for t in data} & {t['id'] for t in old}
    for field in ("instruction", "reference", "faulty"):
        assert not {digest(t[field]) for t in data} & {digest(t[field]) for t in old}
    rows=[]
    for task in data:
        split=task["visible_count"]
        assert 0<split<len(task["tests"])
        visible, hidden=task["tests"][:split],task["tests"][split:]
        assert not {digest(t) for t in visible} & {digest(t) for t in hidden}
        assert check(task["reference"],task["tests"],task["preserve_inputs"])["passed"],task["id"]
        assert not check(task["faulty"],visible,task["preserve_inputs"])["passed"],task["id"]
        killed=[not check(m,task["tests"],task["preserve_inputs"])["passed"] for m in task["mutants"]]
        assert all(killed),task["id"]
        rows.append(dict(task=task["id"],visible_tests=len(visible),hidden_tests=len(hidden),
                         reference_passed=True,mutants_rejected=sum(killed),mutants_total=len(killed)))
    return dict(status="validated_not_model_evaluated",local_exact_overlap=False,
                exact_overlap_checked_fields=["id","instruction","reference","faulty"],
                semantic_family_separation="manually authored different task families",tasks=rows)


def load(directory=OUT):
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text())
    data=json.loads((directory/"tasks.json").read_text())
    protocol=json.loads((directory/"protocol.json").read_text())
    if digest(data)!=manifest["tasks_sha256"] or digest(protocol)!=manifest["protocol_sha256"]:
        raise ValueError("Frozen suite hash mismatch")
    return data,protocol,manifest


def prepare(directory=OUT):
    directory=Path(directory)
    # JSON canonicalization keeps tuple fixtures equivalent to their on-disk lists.
    # Without this, an unchanged second preparation incorrectly looked like tampering.
    data=json.loads(json.dumps(tasks()))
    result=validate(data)
    manifest=dict(tasks_sha256=digest(data),protocol_sha256=digest(PROTOCOL),
                  previous_development_sha256=digest(suite()),created_before_inference=True)
    directory.mkdir(parents=True,exist_ok=True)
    files=[("tasks.json",data),("protocol.json",PROTOCOL),("manifest.json",manifest)]
    for name,value in files:
        path=directory/name
        if path.exists() and json.loads(path.read_text())!=value:
            raise ValueError(f"Refusing to change frozen file: {path}")
    for name,value in files:
        path=directory/name
        if not path.exists():
            path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    (directory/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    prepare()
