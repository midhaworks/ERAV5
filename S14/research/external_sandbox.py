"""Fail-closed macOS sandbox executor and independently measured isolation preflight."""
import json
import argparse
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
from unittest.mock import patch
from external_data import OUT,load
from pretrained_baseline import ROOT


def child_python():
    # Homebrew's framework launcher spawns its application executable. Invoke the
    # real interpreter directly so startup does not need process-fork permission.
    app=Path(sys.base_prefix)/"Resources/Python.app/Contents/MacOS/Python"
    return app.resolve() if app.exists() else Path(sys.executable).resolve()


def profile(workspace,broad_read=False,broad_process=False):
    # No write permissions, network permissions, or process-fork permission.
    # Only the interpreter can exec; project files outside the checker/env cannot be read.
    quote=lambda s: json.dumps(str(s))
    executable=child_python()
    reads=[Path(sys.base_prefix).resolve(),Path(sys.prefix).resolve(),Path("/opt/homebrew"),
           Path("/System/Library"),Path("/usr/lib"),Path("/Library/Apple"),
           Path("/System/Volumes/Preboot/Cryptexes/OS"),Path("/private/preboot/Cryptexes/OS"),
           Path("/private/var/db/dyld"),Path("/usr/share")]
    clauses=" ".join(f"(subpath {quote(p)})" for p in reads)
    read_rule=("(allow file-read* file-map-executable)" if broad_read else
        f'''(allow file-read* {clauses} (literal {quote(ROOT/'external_check.py')})
 (literal "/dev/null") (literal "/dev/random") (literal "/dev/urandom"))
(allow file-map-executable {clauses})''')
    process_rule=("(allow process*)(deny process-fork)" if broad_process else f"(allow process-exec (literal {quote(executable)}))")
    return f'''(version 1)
(deny default)
(allow file-read-metadata)
(allow file-read-data (literal "/"))
{read_rule}
{process_rule}
(allow process-info*)
(allow signal (target self))
(allow sysctl-read)
(allow mach-lookup)
'''


def run(payload,workspace):
    if platform.system()!="Darwin" or not Path("/usr/bin/sandbox-exec").exists():
        raise RuntimeError("Required macOS sandbox unavailable; no unrestricted fallback")
    command=["/usr/bin/sandbox-exec","-p",profile(workspace),str(child_python()),"-I",str(ROOT/"external_check.py")]
    try:
        result=subprocess.run(command,input=json.dumps(payload),text=True,capture_output=True,timeout=6,
                              cwd=workspace,env={"PATH":"/usr/bin:/bin","PYTHONDONTWRITEBYTECODE":"1","LC_CTYPE":"UTF-8"})
        if result.returncode!=0:
            raise RuntimeError(f"Sandbox child failed rc={result.returncode}: stderr={result.stderr[:500]!r} stdout={result.stdout[:500]!r}")
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return dict(passed=False,error="sandbox_wall_timeout")


def check(code,task):
    with tempfile.TemporaryDirectory(prefix="s14-eval-") as workspace:
        return run(dict(code=code,entry_point=task["entry_point"],test=task["test"]),workspace)


def preflight():
    data,_,manifest=load()
    with tempfile.TemporaryDirectory(prefix="s14-secret-") as outside,tempfile.TemporaryDirectory(prefix="s14-eval-") as workspace:
        secret=Path(outside)/"sentinel.txt"
        secret.write_text("not a real secret; isolation-test sentinel")
        with socket.socket() as listener,patch.dict(os.environ,{"S14_SECRET_SENTINEL":"fake-parent-only-sentinel"}):
            listener.bind(("127.0.0.1",0))
            listener.listen()
            isolation=run(dict(probe=True,outside_read=str(secret),outside_write=str(Path(outside)/"new.txt"),
                workspace_write=str(Path(workspace)/"new.txt"),port=listener.getsockname()[1]),workspace)
        if not isolation["passed"]: raise RuntimeError(f"Sandbox isolation failed: {isolation}")
        rules=profile(workspace)
    if not isolation["memory_guard"]["supported"]:
        result=dict(status="blocked_memory_guard_no_model_inference",manifest=manifest,
                    isolation=isolation,sandbox_profile=rules,ready=False,tasks=[],
                    next_requirement="compatible hard memory-limited execution backend; do not disable the guard")
        (OUT/"preflight.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))
        return
    rows=[]
    for task in data:
        reference=check(task["prompt"]+task["canonical_solution"],task)
        wrong=check(f'def {task["entry_point"]}(*args, **kwargs): return None',task)
        rows.append(dict(task=task["task_id"],reference=reference,deliberately_wrong=wrong))
    result=dict(status="preflight_completed_no_model_inference",manifest=manifest,
                isolation=isolation,sandbox_profile=rules,tasks=rows,
                ready=all(r["reference"]["passed"] and not r["deliberately_wrong"]["passed"] for r in rows))
    (OUT/"preflight.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


def diagnose():
    rows=[]
    # Broader permissions are used ONLY for this fixed print command, never candidates.
    with tempfile.TemporaryDirectory(prefix="s14-startup-") as workspace:
        for variant,extra in [("scoped",""),("read_only","(allow file-read*)"),
                              ("map_only","(allow file-map-executable)"),("device_read","(allow file-read* (subpath \"/dev\"))")]:
            command=["/usr/bin/sandbox-exec","-p",profile(workspace)+extra,str(child_python()),"-I","-c","print('startup_ok')"]
            run_result=subprocess.run(command,capture_output=True,text=True,timeout=6,cwd=workspace,
                env={"PATH":"/usr/bin:/bin","PYTHONDONTWRITEBYTECODE":"1","LC_CTYPE":"UTF-8"})
            rows.append(dict(variant=variant,returncode=run_result.returncode,
                             stdout=run_result.stdout,stderr=run_result.stderr))
    (OUT/"startup_diagnostics.json").write_text(json.dumps(rows,indent=2)+"\n")
    print(json.dumps(rows,indent=2))


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--diagnose-startup",action="store_true")
    if cli.parse_args().diagnose_startup: diagnose()
    else: preflight()
