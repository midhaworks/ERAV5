import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_demo import TinyLanguageModel, ce, float_bits  # noqa: E402


class S10Tests(unittest.TestCase):
    def test_causal_loss_shape_and_finiteness(self):
        model = TinyLanguageModel(9)
        tokens = torch.randint(0, 9, (2, 6))
        logits = model(tokens)
        self.assertEqual(tuple(logits[:, :-1].reshape(-1, 9).shape), (10, 9))
        self.assertTrue(torch.isfinite(ce(logits, tokens)))

    def test_float_bit_records(self):
        bits = float_bits()["0.1"]
        self.assertEqual(bits["fp32"]["hex"], "0x3dcccccd")
        self.assertEqual(bits["bf16"]["hex"], "0x3dcd")
        self.assertEqual(bits["fp8_e4m3"]["hex"], "0x1d")

    def test_gradient_exists(self):
        model = TinyLanguageModel(9)
        tokens = torch.randint(0, 9, (2, 6))
        ce(model(tokens), tokens).backward()
        self.assertIsNotNone(model.embedding.weight.grad)
        self.assertGreater(float(model.embedding.weight.grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
