import unittest
from unittest.mock import patch
import torch
from selector_benchmark import features, Selector, shifted
from trajectory_benchmark import evaluate, cases, training_records


class SelectorTests(unittest.TestCase):
    def test_features_do_not_depend_on_labels(self):
        torch.testing.assert_close(features(["abc"]), features(["cba"]))
        self.assertEqual(tuple(features(["", "abc"]).shape), (2,256))
        self.assertTrue(torch.isfinite(features([""])).all())

    def test_selector_uses_text_only(self):
        selector = Selector()
        self.assertIn(selector.choose("Write a function"), ("coding", "agentic"))
        self.assertNotEqual(shifted("Write Python solve(x) returning x+3.\n"),
                            "Write Python solve(x) returning x+3.\n")

    def test_wrong_selector_choice_is_not_replaced_with_oracle(self):
        observed = []
        def generated(model, prompt, group):
            observed.append(group)
            return "def solve(x): return x+3"
        with patch("trajectory_benchmark.generate", side_effect=generated):
            result = evaluate(None, [dict(operation="+",number=3)], only_role="coding",
                              selector=lambda prompt: "agentic")
        self.assertEqual(observed, ["agentic"])
        self.assertEqual(result["routing"][0]["chosen_group"], "agentic")
        self.assertEqual(result["routing"][0]["expected_role"], "coding")

    def test_wording_shift_has_training_byte_coverage(self):
        records = training_records(cases())
        seen = set("".join(r["prompt"]+r["answer"] for r in records if r["split"] == "train").encode())
        stress = set("".join(shifted(r["prompt"]) for r in records if r["split"] == "test").encode())
        self.assertTrue(stress <= seen)


if __name__ == "__main__":
    unittest.main()
