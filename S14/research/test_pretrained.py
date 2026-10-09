import unittest
from pretrained_baseline import suite, check, extract


class QualificationChecks(unittest.TestCase):
    def test_references_and_wrong_solutions(self):
        for task in suite():
            self.assertTrue(check(task["reference"],task["tests"])["passed"],task["id"])
            self.assertFalse(check(task["faulty"],task["tests"])["passed"],task["id"])

    def test_imports_and_introspection_rejected(self):
        self.assertFalse(check('import os\ndef solve(x): return x', [([[1]], [1])])["passed"])
        self.assertFalse(check('def solve(x): return x.__class__', [([[1]], [1])])["passed"])

    def test_valid_set_based_solution_is_accepted(self):
        code='def solve(xs):\n seen=set()\n out=[]\n for x in xs:\n  if x not in seen:\n   seen.add(x)\n   out.append(x)\n return out'
        self.assertTrue(check(code,suite()[0]["tests"])["passed"])

    def test_fences_only_are_removed(self):
        code="def solve(x): return x"
        self.assertEqual(extract(f'```python\n{code}\n```'),code)
        self.assertEqual(extract('Explanation\n'+code),'Explanation\n'+code)

    def test_input_preservation_constraint(self):
        self.assertFalse(check('def solve(xs):\n xs.reverse()\n return xs',
                              [([[1,2]],[2,1])],preserve_inputs=True)["passed"])


if __name__ == "__main__":
    unittest.main()
