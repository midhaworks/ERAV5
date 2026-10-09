import unittest
import torch
from benchmark import LM, tasks, encode, check_response


class BenchmarkTests(unittest.TestCase):
    def test_checker_accepts_reference_and_rejects_wrong_actions(self):
        for record in tasks():
            self.assertTrue(check_response(record, record["answer"]))
            self.assertFalse(check_response(record, "def solve(x):\n return 999\n"))
        record = next(r for r in tasks() if r["role"] == "agentic")
        self.assertFalse(check_response(record, '{"tool":"read","code":"def solve(x): return x"}'))
        record = dict(role="coding", operation="+", number=3)
        self.assertFalse(check_response(record, '@unknown\ndef solve(x): return x+3'))
        self.assertFalse(check_response(record, 'def solve(x=unknown): return x+3'))

    def test_split_has_no_task_pair_overlap(self):
        train = {(r["operation"], r["number"]) for r in tasks() if r["split"] == "train"}
        test = {(r["operation"], r["number"]) for r in tasks() if r["split"] == "test"}
        self.assertFalse(train & test)

    def test_shift_masks_prompt_and_predicts_first_answer_byte(self):
        record = tasks()[0]
        inputs, labels = encode(record)
        boundary = len(record["prompt"].encode()) - 1
        self.assertTrue(all(y == -100 for y in labels[:boundary]))
        self.assertEqual(labels[boundary], record["answer"].encode()[0])
        self.assertEqual(inputs[boundary], record["prompt"].encode()[-1])

    def test_lm_upcycle_and_export_logits(self):
        torch.manual_seed(14)
        model = LM()
        x = torch.tensor([list(b"hello")])
        converted = model.upcycle()
        torch.testing.assert_close(model(x), converted(x))
        exported = converted.export("coding")
        torch.testing.assert_close(converted(x, "coding"), exported(x))


if __name__ == "__main__":
    unittest.main()
