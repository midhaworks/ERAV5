import unittest
from pretrained_baseline import suite
from tool_feedback import Environment, parse_call


class ToolFeedbackChecks(unittest.TestCase):
    def test_real_failure_then_repair(self):
        task = suite()[0]
        env = Environment(task)
        self.assertFalse(env.execute("run_tests", {})["passed"])
        self.assertIn("error", env.execute("finish", {}))
        env.execute("edit_file", dict(path="solution.py", code=task["reference"]))
        self.assertTrue(env.execute("run_tests", {})["passed"])
        self.assertTrue(env.execute("finish", {})["finished"])
        self.assertTrue(env.grade()["passed"])

    def test_edit_invalidates_passing_tests(self):
        task = suite()[0]
        env = Environment(task)
        env.execute("edit_file", dict(path="solution.py", code=task["reference"]))
        env.execute("run_tests", {})
        env.execute("edit_file", dict(path="solution.py", code=task["faulty"]))
        self.assertIn("error", env.execute("finish", {}))

    def test_hidden_failure_is_not_feedback(self):
        env = Environment(suite()[0])
        env.execute("edit_file", dict(path="solution.py", code=
            "def solve(xs): return [] if not xs else [3,1,2]"))
        self.assertTrue(env.execute("run_tests", {})["passed"])
        self.assertEqual(env.execute("finish", {}), {"finished": True})
        self.assertFalse(env.grade()["passed"])

    def test_path_and_argument_validation(self):
        env = Environment(suite()[0])
        for name, args in [("read_file", {"path": "../private"}),
                           ("run_tests", {"hidden": True}), ("edit_file", {"path": "solution.py"})]:
            self.assertIn("error", env.execute(name, args))

    def test_rotation_large_k_cannot_false_pass(self):
        env = Environment(suite()[4])
        env.execute("edit_file", dict(path="solution.py", code=
            "def solve(xs,k): return xs[-k:]+xs[:-k] if xs else []"))
        self.assertTrue(env.execute("run_tests", {})["passed"])
        env.execute("finish", {})
        self.assertFalse(env.grade()["passed"])

    def test_native_call_framing(self):
        seen = []
        def parser(text, tools):
            seen.append(text)
            return {"name": "run_tests", "arguments": {}}
        self.assertEqual(parse_call("<tool_call><function=run_tests></function></tool_call>", parser)["name"], "run_tests")
        self.assertEqual(seen, ["<function=run_tests></function>"])
        for invalid in ["text", "<tool_call>x</tool_call>suffix", "<tool_call>x</tool_call>"*2]:
            with self.assertRaises(ValueError):
                parse_call(invalid, parser)


if __name__ == "__main__":
    unittest.main()
