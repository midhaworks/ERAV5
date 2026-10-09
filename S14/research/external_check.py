"""Run inside OS sandbox; AST restrictions are additional defense, not the sandbox."""
import ast
import builtins
import json
import math
import re
import resource
import sys
import typing

SAFE={"len","range","enumerate","sum","min","max","abs","sorted","list","set","dict","tuple",
      "zip","reversed","int","bool","str","float","round","all","any","ValueError","TypeError"}
ATTRS={"append","count","sort","reverse","add","extend","insert","pop","remove","discard",
       "get","keys","values","items","find","index","split","strip","lstrip","rstrip",
       "join","replace","startswith","endswith","lower","upper","isdigit","isspace",
       "floor","ceil","fabs","sqrt","isclose","findall","search","match","fullmatch","sub",
       "List","Tuple","Dict","Set","Optional","Union"}
MODULES={"math":math,"re":re,"typing":typing}


def evaluate(code,entry_point,test):
    tree=ast.parse(code)
    if len(code)>12000: raise ValueError("candidate_too_long")
    for node in tree.body:
        if not isinstance(node,(ast.FunctionDef,ast.Import,ast.ImportFrom)):
            raise ValueError("only_functions_and_supported_imports")
    for node in ast.walk(tree):
        if isinstance(node,(ast.ClassDef,ast.AsyncFunctionDef,ast.Global,ast.Nonlocal,ast.With,ast.AsyncWith)):
            raise ValueError("unsupported_construct")
        if isinstance(node,ast.FunctionDef) and node.decorator_list: raise ValueError("decorators_not_supported")
        if isinstance(node,ast.Name) and (node.id.startswith("_") or node.id in {"open","exec","eval","compile","globals","locals","vars","getattr","setattr","delattr","input","help","breakpoint"}):
            raise ValueError("unsupported_name")
        if isinstance(node,ast.Attribute) and node.attr not in ATTRS: raise ValueError("unsupported_attribute")
        if isinstance(node,ast.Import) and any(a.name not in MODULES for a in node.names):
            raise ValueError("unsupported_import")
        if isinstance(node,ast.ImportFrom) and (node.level or node.module not in MODULES or
                any(a.name.startswith("_") or a.name=="*" for a in node.names)):
            raise ValueError("unsupported_import")
    def safe_import(name,globals=None,locals=None,fromlist=(),level=0):
        if name not in MODULES or level: raise ValueError("unsupported_import")
        return MODULES[name]
    env={"__builtins__":dict({n:getattr(builtins,n) for n in SAFE},__import__=safe_import)}
    exec(compile(tree,"<candidate>","exec"),env)
    if entry_point not in env: return dict(passed=False,error="entry_point_missing")
    # Official test source is pinned, separate from generated source, and runs unchanged.
    tests={"__builtins__":{n:getattr(builtins,n) for n in SAFE}}
    exec(compile(test,"<official_tests>","exec"),tests)
    tests["check"](env[entry_point])
    return dict(passed=True,error=None)


def probes(payload):
    """Trusted probes intentionally bypass AST gates to test OS restrictions themselves."""
    import os
    import socket
    checks={}
    for label,path,mode in [("outside_read_blocked",payload["outside_read"],"r"),
                            ("outside_write_blocked",payload["outside_write"],"w"),
                            ("workspace_write_blocked",payload["workspace_write"],"w")]:
        try:
            with open(path,mode): pass
            checks[label]=False
        except PermissionError: checks[label]=True
    try:
        with socket.socket() as sock: sock.connect(("127.0.0.1",payload["port"]))
        checks["network_blocked"]=False
    except PermissionError: checks["network_blocked"]=True
    except OSError as error: checks["network_probe_error"]=str(error)
    try:
        pid=os.fork()
        if pid==0: os._exit(0)
        os.waitpid(pid,0)
        checks["fork_blocked"]=False
    except PermissionError: checks["fork_blocked"]=True
    checks["environment_sanitized"]="S14_SECRET_SENTINEL" not in os.environ
    return dict(passed=all(checks.get(k) is True for k in (
        "outside_read_blocked","outside_write_blocked","workspace_write_blocked","network_blocked",
        "fork_blocked","environment_sanitized")),checks=checks)


if __name__=="__main__":
    try:
        def cap(kind,requested):
            _,hard=resource.getrlimit(kind)
            value=requested if hard==resource.RLIM_INFINITY else min(requested,hard)
            try: resource.setrlimit(kind,(value,hard))
            except ValueError as error: raise ValueError(f"resource_cap_failed kind={kind} requested={value} existing={resource.getrlimit(kind)}: {error}")
        cap(resource.RLIMIT_CPU,3)
        payload=json.load(sys.stdin)
        try:
            cap(resource.RLIMIT_DATA,64*1024**2)
            memory_guard=dict(supported=True,limit=resource.getrlimit(resource.RLIMIT_DATA))
        except ValueError as error:
            memory_guard=dict(supported=False,error=str(error))
        if payload.get("probe"):
            # Trusted bounded isolation checks may run; candidates fail closed below.
            answer=probes(payload)
            answer["memory_guard"]=memory_guard
        elif not memory_guard["supported"]:
            answer=dict(passed=False,error="required_memory_guard_unavailable",memory_guard=memory_guard)
        else: answer=evaluate(payload["code"],payload["entry_point"],payload["test"])
        print(json.dumps(answer))
    except Exception as error:
        print(json.dumps(dict(passed=False,error=f"{type(error).__name__}: {error}")))
