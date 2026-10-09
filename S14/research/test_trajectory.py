import json
import unittest
from benchmark import LM, batch
from trajectory_benchmark import cases, training_records, trajectory, flops_proxy, reference_code


class TrajectoryTests(unittest.TestCase):
    def test_reference_completes_and_wrong_edit_fails(self):
        case = dict(operation="+", number=3)
        actions = iter([{"tool":"read"}, {"tool":"edit", "code":reference_code(case)}, {"tool":"test"}])
        self.assertTrue(trajectory(case, lambda p: json.dumps(next(actions)))["passed"])
        actions = iter([{"tool":"read"}, {"tool":"edit", "code":"def solve(x): return x-3"}, {"tool":"test"}])
        result = trajectory(case, lambda p: json.dumps(next(actions)))
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure"], "tests_failed")

    def test_wrong_tool_does_not_advance(self):
        result = trajectory(dict(operation="+", number=3), lambda p: '{"tool":"test"}')
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["events"]), 1)

    def test_context_and_split(self):
        data = cases()
        batch(training_records(data))
        for split in ("train", "validation", "test"):
            self.assertTrue(any(c["split"] == split for c in data))
        self.assertEqual(len({(c["operation"],c["number"]) for c in data}), len(data))

    def test_compute_proxy_accounts_for_active_experts(self):
        x, _ = batch(training_records(cases()[:1]))
        dense = LM()
        self.assertGreater(flops_proxy(dense.upcycle(), x), flops_proxy(dense, x))


if __name__ == "__main__":
    unittest.main()
