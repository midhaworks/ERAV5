import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_demo import Vocab, TinyLM, chunked_ce, masked_ce  # noqa: E402


class S9HarnessTests(unittest.TestCase):
    def test_string_shift_is_forward(self):
        vocab = Vocab(["red green blue"])
        sequence = vocab.encode("red green blue")
        inputs = [vocab.decode_id(i) for i in sequence[:-1]]
        targets = [vocab.decode_id(i) for i in sequence[1:]]
        self.assertEqual(inputs, ["red", "green", "blue"])
        self.assertEqual(targets, ["green", "blue", "<eos>"])

    def test_chunked_cross_entropy_matches_ordinary(self):
        torch.manual_seed(2)
        logits = torch.randn(13, 7)
        targets = torch.randint(7, (13,))
        mask = torch.ones(13, dtype=torch.bool)
        ordinary, _ = masked_ce(logits, targets, mask)
        chunked = chunked_ce(logits, targets, mask, chunk=4)
        self.assertTrue(torch.allclose(ordinary, chunked, atol=1e-7, rtol=0.0))

    def test_model_shape_contract(self):
        model = TinyLM(11)
        tokens = torch.randint(1, 11, (2, 5))
        attention = torch.ones_like(tokens, dtype=torch.bool)
        hidden = model(tokens, attention)
        self.assertEqual(tuple(hidden.shape), (2, 5, 48))
        self.assertEqual(tuple(model.head1(hidden).shape), (2, 5, 11))
        self.assertEqual(tuple(model.head2(hidden).shape), (2, 5, 11))


if __name__ == "__main__":
    unittest.main()
