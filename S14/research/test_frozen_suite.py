import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from frozen_suite import tasks, validate, prepare, load
from tool_feedback import Environment, initial_messages


class FrozenSuiteChecks(unittest.TestCase):
    def test_references_mutants_and_overlap(self):
        result=validate(tasks())
        self.assertFalse(result["local_exact_overlap"])
        self.assertEqual(sum(t["mutants_rejected"] for t in result["tasks"]),18)

    def test_hash_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(folder)
                prepare(folder)  # unchanged preparation is idempotent
            self.assertEqual(len(load(folder)[0]),6)
            path=Path(folder)/"tasks.json"
            changed=json.loads(path.read_text())
            changed[0]["instruction"]="tampered"
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                load(folder)
            with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
                prepare(folder)

    def test_explicit_visible_split_and_input_preservation(self):
        task=tasks()[0]
        env=Environment(task)
        self.assertEqual(len(env.visible),3)
        self.assertTrue(env.preserve_inputs)
        messages=initial_messages(task,env)
        expected=json.dumps(task["tests"][:3])
        self.assertTrue(messages[1]["content"].endswith(expected))
        # The mutating reference-equivalent implementation must not pass.
        env.execute("edit_file",dict(path="solution.py",code=task["mutants"][-1]))
        self.assertFalse(env.execute("run_tests",{})["passed"])


if __name__=="__main__":
    unittest.main()
