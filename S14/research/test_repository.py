import contextlib
import io
import tempfile
import unittest
from repository_suite import tasks,prepare,load
from repository_environment import Environment


class RepositoryChecks(unittest.TestCase):
    def test_reference_and_individual_bugs(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            prepare(directory)
            prepare(directory)
            self.assertEqual(len(load(directory)[0]),3)

    def test_local_imports_actually_execute(self):
        task=tasks()[0]
        with Environment(task) as env:
            self.assertFalse(env.execute("run_tests",{})["passed"])
            for path,code in task["reference"].items(): env.execute("edit_file",dict(path=path,code=code))
            self.assertTrue(env.execute("run_tests",{})["passed"])
            self.assertEqual(env.execute("finish",{}),dict(finished=True))
            self.assertTrue(env.grade()["passed"])

    def test_readonly_and_hidden_isolation(self):
        with Environment(tasks()[0]) as env:
            self.assertIn("error",env.execute("read_file",dict(path="../hidden.json")))
            self.assertIn("error",env.execute("read_file",dict(path="hidden_tests.json")))
            self.assertIn("error",env.execute("edit_file",dict(path="visible_tests.json",code="[]")))
            self.assertNotIn("hidden_tests.json",env.snapshot())

    def test_stale_tests_cannot_finish(self):
        task=tasks()[0]
        with Environment(task) as env:
            for path,code in task["reference"].items(): env.edit(path,code)
            self.assertTrue(env.execute("run_tests",{})["passed"])
            env.edit("weights.py",task["files"]["weights.py"])
            self.assertIn("error",env.execute("finish",{}))

    def test_external_import_and_introspection_rejected(self):
        with Environment(tasks()[0]) as env:
            env.edit("api.py","import os\ndef solve(items,rate): return 0")
            self.assertFalse(env.execute("run_tests",{})["passed"])
            env.edit("api.py","def solve(items,rate): return items.__class__")
            self.assertFalse(env.execute("run_tests",{})["passed"])

    def test_hidden_integration_rejects_visible_overfit(self):
        with Environment(tasks()[0]) as env:
            env.edit("weights.py","def billable(units): return 1")
            env.edit("api.py","def solve(items,rate): return rate if items else 0")
            self.assertTrue(env.execute("run_tests",{})["passed"])
            env.execute("finish",{})
            self.assertFalse(env.grade()["passed"])


if __name__=="__main__": unittest.main()
