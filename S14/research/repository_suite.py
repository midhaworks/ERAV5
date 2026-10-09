"""Frozen multi-file repair fixtures; hidden integration tests stay outside the workspace."""
import json
from pathlib import Path
from frozen_suite import digest
from pretrained_baseline import ROOT
from tool_feedback import REVISION

OUT = ROOT / "artifacts" / "repository_v1"
PROTOCOL = dict(version=1, revision=REVISION, max_calls=16, max_tokens=512,
    temperature=0, enable_thinking=False, full_run_tasks=3,
    primary_metric="finished with fresh visible pass and all hidden integration tests passing",
    stop_rule="one full run; no selective retries or edits to suite after inference",
    scope="authored multi-file miniature repositories, not public benchmark or production qualification")


def tasks():
    def case(entry, args, expected):
        return dict(entry=entry, args=args, expected=expected)
    shipping = dict(id="shipping", instruction="Repair the shipping quotation library. Units and rates are "
        "nonnegative integers. Each shipment costs ceil(units/1000)*rate; zero units cost zero. "
        "A quote sums independent shipment costs. There is no base fee. Preserve inputs and function signatures.",
        files={"weights.py": "def billable(units): return units // 1000\n",
               "tariffs.py": "from weights import billable\ndef charge(units, rate): return billable(units)*rate + rate\n",
               "api.py": "from tariffs import charge\ndef solve(items, rate): return sum(charge(n, rate) for n in items)\n"},
        reference={"weights.py": "def billable(units): return (units+999)//1000\n",
                   "tariffs.py": "from weights import billable\ndef charge(units, rate): return billable(units)*rate\n"},
        visible=[case("weights.billable",[1],1),case("api.solve",[[1000],5],5),case("api.solve",[[],5],0)],
        hidden=[case("api.solve",[[0],7],0),case("api.solve",[[1,1000,1001,2000],3],18),
                case("api.solve",[[999,2001],2],8),case("api.solve",[[123456],0],0),case("tariffs.charge",[0,9],0)])
    pages = dict(id="pagination", instruction="Repair the pagination library. Page numbers start at 1, page size "
        "is a positive integer. window(page,size) returns [inclusive_start,exclusive_end]. "
        "solve(rows,page,size) returns that list slice, naturally truncated at the end. "
        "Beyond the last row returns []. Preserve inputs and function signatures.",
        files={"bounds.py":"def window(page, size): return [page*size, (page+1)*size]\n",
               "pages.py":"from bounds import window\ndef page_items(rows, page, size):\n start,end=window(page,size)\n return rows[start:end-1]\n",
               "api.py":"from pages import page_items\ndef solve(rows, page, size): return page_items(rows,page,size)\n"},
        reference={"bounds.py":"def window(page, size): return [(page-1)*size, page*size]\n",
                   "pages.py":"from bounds import window\ndef page_items(rows, page, size):\n start,end=window(page,size)\n return rows[start:end]\n"},
        visible=[case("bounds.window",[1,2],[0,2]),case("api.solve",[[0,1,2],1,2],[0,1]),case("api.solve",[[0,1,2],2,2],[2])],
        hidden=[case("api.solve",[[],1,4],[]),case("api.solve",[[1,2],5,2],[]),
                case("api.solve",[[1,2,3,4],2,2],[3,4]),case("api.solve",[[8,9,10],2,1],[9]),
                case("pages.page_items",[[1,2,3],1,9],[1,2,3])])
    def row(status,amount,fee):
        return dict(status=status,amount=amount,fee=fee)
    ledger = dict(id="ledger", instruction="Repair the settlement library. Each row has status (string), "
        "amount (integer) and fee (nonnegative integer). Only status exactly 'paid' with amount>=0 is accepted. "
        "Each accepted row contributes amount-fee; others contribute zero. Fees may exceed amount. "
        "solve(rows) sums contributions. Preserve inputs and function signatures.",
        files={"validation.py":"def accepted(row): return row['status']=='paid'\n",
               "totals.py":"from validation import accepted\ndef net_total(rows): return sum(r['amount'] for r in rows if accepted(r))\n",
               "api.py":"from totals import net_total\ndef solve(rows): return net_total(rows)\n"},
        reference={"validation.py":"def accepted(row): return row['status']=='paid' and row['amount']>=0\n",
                   "totals.py":"from validation import accepted\ndef net_total(rows): return sum(r['amount']-r['fee'] for r in rows if accepted(r))\n"},
        visible=[case("validation.accepted",[row("paid",-2,0)],False),
                 case("api.solve",[[row("paid",10,2)]],8),case("api.solve",[[row("cancelled",5,0)]],0)],
        hidden=[case("api.solve",[[]],0),case("api.solve",[[row("paid",0,1)]],-1),
                case("api.solve",[[row("paid",8,2),row("paid",-2,1),row("pending",20,0),row("paid",2,0)]],8),
                case("api.solve",[[row("Paid",10,1)]],0),case("totals.net_total",[[row("paid",4,9)]],-5)])
    result = [shipping,pages,ledger]
    for task in result:
        task["source"]="original authored miniature repository"
        task["license"]="CC0-1.0"
        task["files"]["README.md"]=task["instruction"]+"\n"
        task["files"]["visible_tests.json"]=json.dumps(task["visible"],indent=2)+"\n"
    return result


def load(directory=OUT):
    directory=Path(directory)
    data=json.loads((directory/"tasks.json").read_text())
    protocol=json.loads((directory/"protocol.json").read_text())
    manifest=json.loads((directory/"manifest.json").read_text())
    if digest(data)!=manifest["tasks_sha256"] or digest(protocol)!=manifest["protocol_sha256"]:
        raise ValueError("Frozen repository suite hash mismatch")
    return data,protocol,manifest


def prepare(directory=OUT):
    from repository_environment import Environment
    directory=Path(directory)
    data=tasks()
    rows=[]
    for task in data:
        with Environment(task) as env:
            assert not env.check(task["visible"])["passed"]
            kills=[]
            for path,code in task["reference"].items():
                env.edit(path,code)
            assert env.check(task["visible"]+task["hidden"])["passed"],task["id"]
            # Reintroduce each bug individually; integration tests must expose both.
            for path,code in task["reference"].items():
                env.edit(path,task["files"][path])
                kills.append(not env.check(task["visible"]+task["hidden"])["passed"])
                env.edit(path,code)
            assert all(kills)
            rows.append(dict(task=task["id"],reference_passed=True,individual_bugs_rejected=sum(kills),
                             visible_tests=len(task["visible"]),hidden_tests=len(task["hidden"])))
    manifest=dict(tasks_sha256=digest(data),protocol_sha256=digest(PROTOCOL),
                  scope="authored repository qualification frozen before inference")
    files=[("tasks.json",data),("protocol.json",PROTOCOL),("manifest.json",manifest)]
    directory.mkdir(parents=True,exist_ok=True)
    for name,value in files:
        path=directory/name
        if path.exists() and json.loads(path.read_text())!=value:
            raise ValueError("Refusing to change frozen suite")
    for name,value in files:
        path=directory/name
        if not path.exists(): path.write_text(json.dumps(value,indent=2)+"\n")
    result=dict(status="validated_not_model_evaluated",tasks=rows)
    (directory/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    prepare()
