import ast
import unittest
from external_data import load,PROTOCOL
from external_check import evaluate


class ExternalChecks(unittest.TestCase):
    def test_evaluation_only_api(self):
        with self.assertRaises(ValueError): load(for_training=True)

    def test_deterministic_selection(self):
        self.assertEqual(PROTOCOL["task_ids"],[f"HumanEval/{i}" for i in range(8)])
        self.assertIn("no task substitution",PROTOCOL["selection"])

    def test_candidate_policy_rejects_unsafe_code_before_execution(self):
        for code in ["import os\ndef f(): return 1", "def f(): return open('anything')",
                     "def f(): return [].__class__", "from pathlib import Path\ndef f(): return 1"]:
            with self.assertRaises(ValueError): evaluate(code,"f","def check(candidate): pass")

    def test_official_float_test_tolerance_is_not_replaced(self):
        tests="def check(candidate):\n assert abs(candidate(1.33)-0.33)<1e-6"
        self.assertTrue(evaluate("def f(x): return x-int(x)","f",tests)["passed"])
        with self.assertRaises(AssertionError): evaluate("def f(x): return x", "f",tests)

    def test_reference_fields_are_not_prompt_content(self):
        data,_,_=load()
        for task in data:
            ast.parse(task["prompt"]+task["canonical_solution"])
            self.assertNotIn("def check(candidate)",task["prompt"])
        self.assertEqual([t["task_id"] for t in data],PROTOCOL["task_ids"])


if __name__=="__main__": unittest.main()
