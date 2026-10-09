"""Isolated arithmetic/list code evaluator, not a general secure Python sandbox."""
import ast
import json
import resource
import sys


def main():
    payload=json.load(sys.stdin)
    resource.setrlimit(resource.RLIMIT_CPU,(2,2))
    code=payload["code"]
    if len(code)>8000:
        raise ValueError("Code too long")
    tree=ast.parse(code)
    allowed_calls={"solve","len","range","enumerate","sum","min","max","abs","sorted",
                   "list","set","dict","tuple","zip","reversed","int","bool","str"}
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.Global,ast.Nonlocal,
                            ast.With,ast.AsyncWith,ast.AsyncFunctionDef)):
            raise ValueError("Unsupported construct")
        if isinstance(node,ast.Name) and "__" in node.id:
            raise ValueError("Unsupported name")
        if isinstance(node,ast.Attribute) and node.attr not in {
                "append","count","sort","reverse","add","extend","insert","pop",
                "remove","discard","get","keys","values","items","fromkeys"}:
            raise ValueError("Unsupported attribute")
        if isinstance(node,ast.Call):
            if isinstance(node.func,ast.Name) and node.func.id not in allowed_calls:
                raise ValueError("Unsupported call")
            if not isinstance(node.func,(ast.Name,ast.Attribute)):
                raise ValueError("Unsupported call")
        if isinstance(node,ast.FunctionDef) and (node.decorator_list or node.name!="solve"):
            raise ValueError("Only plain solve function allowed")
        if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)) and abs(node.value)>100000:
            raise ValueError("Constant too large")
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
        raise ValueError("Expected one function")
    builtins=__builtins__ if isinstance(__builtins__,dict) else vars(__builtins__)
    env={"__builtins__":{k:builtins[k] for k in allowed_calls if k!="solve"}}
    exec(compile(tree,"<generated>","exec"),env)
    for args,expected in payload["tests"]:
        before=json.loads(json.dumps(args))
        actual=env["solve"](*args)
        if payload.get("preserve_inputs") and args!=before:
            print(json.dumps(dict(passed=False,error="input_mutated")))
            return
        # Distinguish booleans from integer answers; allow tuples as sequence equivalents.
        if actual!=expected or (isinstance(expected,bool) and type(actual) is not bool):
            print(json.dumps(dict(passed=False,error="wrong_answer",actual=actual,expected=expected)))
            return
    print(json.dumps(dict(passed=True,error=None)))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(passed=False,error=f"{type(error).__name__}: {error}")))
