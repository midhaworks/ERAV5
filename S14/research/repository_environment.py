"""Real temporary files and local-module execution for controlled repair fixtures."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from pretrained_baseline import ROOT
from tool_feedback import tool

TOOLS=[tool("list_files","List available repository files and whether each is editable.",{}),
       tool("read_file","Read an available file; hidden tests are not available.",{"path":{"type":"string"}}),
       tool("edit_file","Replace an editable module. Keep public function signatures; use plain unannotated functions and local from-imports only.",
            {"path":{"type":"string"},"code":{"type":"string"}}),
       tool("run_tests","Execute the visible tests against current files. Return actual failure details.",{}),
       tool("finish","Complete only after the current repository passes visible tests.",{})]


class Environment:
    def __init__(self,task):
        self.task=task
        self.temp=tempfile.TemporaryDirectory(prefix="s14-repair-")
        self.root=Path(self.temp.name)
        self.editable={f for f in task["files"] if f.endswith(".py")}
        self.finished=False
        self.public_pass=False
        self.tested_snapshot=None
        for path,code in task["files"].items():
            if Path(path).name!=path:
                self.temp.cleanup()
                raise ValueError("Only declared root files allowed")
            (self.root/path).write_text(code)

    def __enter__(self): return self
    def __exit__(self,*args): self.temp.cleanup()

    def snapshot(self):
        return {f:(self.root/f).read_text() for f in sorted(self.task["files"])}

    def edit(self,path,code):
        if path not in self.editable or len(code)>8000:
            raise ValueError("File is not editable or code is too long")
        (self.root/path).write_text(code)
        self.public_pass=False
        self.tested_snapshot=None

    def check(self,tests):
        payload=dict(root=str(self.root),files=sorted(self.editable),tests=tests)
        try:
            run=subprocess.run([sys.executable,"-I",str(ROOT/"repository_check.py")],
                input=json.dumps(payload),text=True,capture_output=True,timeout=5)
            return json.loads(run.stdout) if run.returncode==0 else dict(passed=False,error="checker_process_failed")
        except (ValueError,subprocess.TimeoutExpired):
            return dict(passed=False,error="timeout_or_invalid_checker_result")

    def execute(self,name,arguments):
        if self.finished: return dict(error="already_finished")
        schema=next((t["function"] for t in TOOLS if t["function"]["name"]==name),None)
        if schema is None or not isinstance(arguments,dict): return dict(error="invalid_tool")
        if set(arguments)!=set(schema["parameters"]["required"]) or any(not isinstance(v,str) for v in arguments.values()):
            return dict(error="invalid_arguments")
        if name=="list_files":
            return dict(files=[dict(path=f,editable=f in self.editable) for f in sorted(self.task["files"])])
        if name=="read_file":
            path=arguments["path"]
            if path not in self.task["files"]: return dict(error="file_not_available")
            return dict(path=path,content=(self.root/path).read_text())
        if name=="edit_file":
            try: self.edit(arguments["path"],arguments["code"])
            except ValueError as error: return dict(error=str(error))
            return dict(edited=True,requires_fresh_tests=True)
        if name=="run_tests":
            outcome=self.check(self.task["visible"])
            self.public_pass=outcome["passed"]
            self.tested_snapshot=self.snapshot()
            return outcome
        if not self.public_pass or self.tested_snapshot!=self.snapshot():
            return dict(error="fresh_visible_pass_required")
        self.finished=True
        return dict(finished=True)

    def grade(self):
        hidden=self.check(self.task["hidden"])
        return dict(completed=self.finished,visible_pass=self.public_pass,hidden_outcome=hidden,
                    passed=self.finished and self.public_pass and hidden["passed"])


def initial_messages(task,bundled_reads=False):
    policy=("one native call per turn, or up to eight read_file calls bundled in a turn" if bundled_reads
            else "exactly one native tool call per turn")
    return [dict(role="system",content=f"Repair a small Python repository using {policy}. "
        "Inspect files and dependencies, edit modules, run visible tests, and finish only after they pass. "
        "Keep the public function names and signatures. Only plain unannotated functions, basic Python collections, "
        "and from-imports of existing local module functions are supported. No external imports, shell, classes, "
        "decorators, or top-level executable statements. Tests and README are read-only. Hidden tests are unavailable."),
        dict(role="user",content=task["instruction"]+"\nStart by listing and reading repository files.")]
