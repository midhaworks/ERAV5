"""Restricted local-module executor. Controlled fixtures only, not an OS security sandbox."""
import ast
import json
from pathlib import Path
import resource
import sys
import types

SAFE = {"len", "range", "enumerate", "sum", "min", "max", "abs", "sorted", "list",
        "set", "dict", "tuple", "zip", "reversed", "int", "bool", "str"}
ATTRS = {"append", "count", "sort", "reverse", "add", "extend", "insert", "pop",
         "remove", "discard", "get", "keys", "values", "items"}


def execute(root, files, tests):
    sources = {Path(f).stem: (root / f).read_text() for f in files}
    trees = {name: ast.parse(code) for name, code in sources.items()}
    exports = {name: {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
               for name, tree in trees.items()}
    for name, tree in trees.items():
        if len(sources[name]) > 8000:
            raise ValueError("module_too_long")
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                if node.level or node.module not in trees or any(a.name not in exports[node.module] for a in node.names):
                    raise ValueError("only_declared_local_function_imports_allowed")
                imported.update(a.asname or a.name for a in node.names)
            elif not isinstance(node, ast.FunctionDef):
                raise ValueError("only_functions_and_local_imports_at_module_level")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ClassDef, ast.Global, ast.Nonlocal,
                                 ast.With, ast.AsyncWith, ast.AsyncFunctionDef)):
                raise ValueError("unsupported_construct")
            if isinstance(node, ast.ImportFrom) and node not in tree.body:
                raise ValueError("only_module_level_imports_allowed")
            if isinstance(node, ast.FunctionDef):
                if node.decorator_list or node.args.defaults or node.args.kw_defaults or node.returns:
                    raise ValueError("plain_functions_required")
                if any(a.annotation for a in node.args.args + node.args.posonlyargs + node.args.kwonlyargs):
                    raise ValueError("annotations_not_supported")
                if "__" in node.name or (node.args.vararg and "__" in node.args.vararg.arg):
                    raise ValueError("unsupported_name")
            if isinstance(node, ast.Name) and "__" in node.id:
                raise ValueError("unsupported_name")
            if isinstance(node, ast.Attribute) and node.attr not in ATTRS:
                raise ValueError("unsupported_attribute")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id not in SAFE | exports[name] | imported:
                    raise ValueError("unsupported_call")
                if not isinstance(node.func, (ast.Name, ast.Attribute)):
                    raise ValueError("unsupported_call")
    modules, loading = {}, set()
    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in trees or any(n not in exports[name] for n in fromlist):
            raise ValueError("forbidden_import")
        return load(name)
    def load(name):
        if name in loading:
            raise ValueError("cyclic_import")
        if name not in modules:
            loading.add(name)
            module = types.ModuleType(name)
            safe_builtins = {n: getattr(__import__("builtins"), n) for n in SAFE}
            module.__dict__["__builtins__"] = dict(safe_builtins, __import__=local_import)
            exec(compile(trees[name], name+".py", "exec"), module.__dict__)
            modules[name] = module
            loading.remove(name)
        return modules[name]
    # Validate and import all editable modules, even ones not reached by tests.
    for name in trees:
        load(name)
    for index, case in enumerate(tests):
        module_name, function_name = case["entry"].split(".")
        if function_name not in exports.get(module_name, set()):
            raise ValueError("entry_point_missing")
        args = json.loads(json.dumps(case["args"]))
        before = json.loads(json.dumps(args))
        actual = getattr(load(module_name), function_name)(*args)
        if args != before:
            return dict(passed=False, error="input_mutated", test_index=index)
        expected = case["expected"]
        if actual != expected or (isinstance(expected, bool) and type(actual) is not bool):
            return dict(passed=False, error="wrong_answer", test_index=index,
                        entry=case["entry"], args=before, actual=actual, expected=expected)
    return dict(passed=True, error=None, tests_run=len(tests))


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    payload = json.load(sys.stdin)
    files = payload["files"]
    if any(Path(f).name != f or not f.endswith(".py") for f in files):
        raise ValueError("invalid_module_path")
    print(json.dumps(execute(Path(payload["root"]), files, payload["tests"])))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(passed=False, error=f"{type(error).__name__}: {error}")))
