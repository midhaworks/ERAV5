import unittest
from benchmark import batch
from trajectory_benchmark import cases, training_records
from diversity_benchmark import diversified, wording


class DiversityTests(unittest.TestCase):
    def test_no_heldout_prompt_or_unseen_byte(self):
        data = cases()
        training = diversified(training_records([c for c in data if c["split"] == "train"]))
        heldout = [wording(r["prompt"], "heldout") for r in training_records(data)]
        seen = set("".join(r["prompt"]+r["answer"] for r in training).encode())
        self.assertTrue(set("".join(heldout).encode()) <= seen)
        self.assertFalse(set(heldout) & {r["prompt"] for r in training})
        batch(training)


if __name__ == "__main__":
    unittest.main()
