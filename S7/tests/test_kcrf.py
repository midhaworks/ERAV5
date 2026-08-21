import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kcrf import CONT, EOS, KCRFHead


class KCRFTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.head = KCRFHead(12, torch.nn.Embedding(259, 8), rank=4)

    def test_exact_utf8_viterbi_and_eos(self):
        # UTF-8 for é is C3 A9; the final position must be EOS.
        unary = torch.full((3, 259), -20.0)
        unary[0, 3 + 0xC3] = 4.0
        unary[1, 3 + 0xA9] = 4.0
        unary[2, EOS] = 4.0
        result = self.head.viterbi(unary)
        self.assertEqual(result.ids, (3 + 0xC3, 3 + 0xA9, EOS))
        self.assertTrue(torch.isfinite(self.head.nll(unary, torch.tensor(result.ids))))

    def test_invalid_ascii_eos_path_is_excluded(self):
        unary = torch.zeros(2, 259)
        unary[0, EOS] = 100.0  # illegal before the last position
        unary[0, 3 + ord("a")] = 1.0
        unary[1, EOS] = 1.0
        result = self.head.viterbi(unary)
        self.assertEqual(result.ids, (3 + ord("a"), EOS))

    def test_cont_is_only_final_and_parameter_saving_is_explicit(self):
        report = self.head.parameter_report()
        self.assertEqual(report["transition_parameters"], 2 * 259 * 4)
        self.assertEqual(report["dense_transition_parameters_equivalent"], 259 * 259)
        unary = torch.zeros(2, 259)
        unary[0, 3 + ord("a")] = 1.0
        unary[1, CONT] = 1.0
        self.assertEqual(self.head.viterbi(unary).ids[-1], CONT)

    def test_exact_nll_has_gradients(self):
        unary = torch.randn(3, 259, requires_grad=True)
        target = torch.tensor([3 + ord("a"), 3 + ord("b"), EOS])
        loss = self.head.nll(unary, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(unary.grad)
        self.assertTrue(torch.isfinite(unary.grad).all())

    def test_unicode_dfa_accepts_multibyte_and_rejects_incomplete_eos(self):
        lead, cont, cont2 = 3 + 0xE0, 3 + 0xA4, 3 + 0x80
        active = (EOS, CONT, lead, cont, cont2)
        head = KCRFHead(12, torch.nn.Embedding(259, 8), rank=4, active_states=active)
        unary = torch.full((4, 259), -10.0)
        unary[0, lead] = 2.0; unary[1, cont] = 2.0; unary[2, cont2] = 2.0; unary[3, EOS] = 2.0
        loss = head.nll(unary, torch.tensor([lead, cont, cont2, EOS]))
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(ValueError):
            head.nll(unary, torch.tensor([lead, EOS, EOS, EOS]))


if __name__ == "__main__":
    unittest.main()
