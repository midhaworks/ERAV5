import unittest

import torch

from run_demo import DenseLinear, TopKMoE, make_data, parameter_count


class S14Invariants(unittest.TestCase):
    def test_conversion_has_expected_shape_and_parameter_count(self):
        dense = DenseLinear(8)
        moe = TopKMoE.from_dense(dense, experts=4, top_k=2)
        x, _, _, _ = make_data(n_train=5, n_val=3)
        y = moe(x)
        self.assertEqual(tuple(y.shape), (5, 1))
        self.assertEqual(parameter_count(dense), 9)
        self.assertEqual(parameter_count(moe), 72)

    def test_router_selects_top_k(self):
        dense = DenseLinear(8)
        moe = TopKMoE.from_dense(dense, experts=4, top_k=2)
        x, _, _, _ = make_data(n_train=7, n_val=2)
        _, indices, _ = moe(x, return_router=True)
        self.assertEqual(tuple(indices.shape), (7, 2))
        self.assertTrue(torch.all(indices[:, 0] != indices[:, 1]))


if __name__ == "__main__":
    unittest.main()
