import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tiny_kcode_exact_token_finetune import _code_scores  # noqa: E402


class ExactTokenObjectiveTests(unittest.TestCase):
    def test_code_score_is_sum_through_eos_and_ignores_pad(self):
        log_probs = torch.tensor(
            [[[0.0, -1.0, -2.0, -3.0], [0.0, -4.0, -5.0, -6.0], [0.0, -7.0, -8.0, -9.0]]],
            requires_grad=True,
        )
        # State 1 is EOS and state 0 is post-EOS PAD.
        codes = torch.tensor([[3, 1, 0], [2, 3, 1]])
        scores = _code_scores(log_probs, codes)
        self.assertTrue(torch.equal(scores, torch.tensor([[-7.0, -15.0]])))
        scores.sum().backward()
        self.assertIsNotNone(log_probs.grad)
        self.assertEqual(float(log_probs.grad[0, 2, 0]), 0.0)

    def test_chunked_log_normalizer_matches_direct(self):
        log_probs = torch.log_softmax(torch.arange(24, dtype=torch.float32).view(2, 3, 4), -1)
        codes = torch.tensor([[1, 0, 0], [2, 1, 0], [3, 2, 1], [1, 3, 1], [2, 2, 1]])
        direct = torch.logsumexp(_code_scores(log_probs, codes), -1)
        chunks = [_code_scores(log_probs, codes[:2]), _code_scores(log_probs, codes[2:])]
        chunked = torch.logaddexp(torch.logsumexp(chunks[0], -1), torch.logsumexp(chunks[1], -1))
        self.assertTrue(torch.allclose(chunked, direct, atol=1e-6, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
