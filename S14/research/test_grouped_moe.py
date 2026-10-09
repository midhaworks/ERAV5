import unittest
import torch
from torch import nn
from grouped_moe import GroupedMoE, parameters


class ExportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        self.dense = nn.Linear(8, 4).double()
        self.model = GroupedMoE.from_linear(self.dense, 4, 2)
        self.x = torch.randn(2, 7, 8, dtype=torch.float64)

    def test_conversion_preserves_dense_function(self):
        for group in (None, "coding", "agentic"):
            torch.testing.assert_close(self.model(self.x, group), self.dense(self.x),
                                       atol=1e-12, rtol=1e-12)

    def test_export_matches_masked_parent_after_specialization(self):
        with torch.no_grad():
            for i, expert in enumerate(self.model.experts):
                expert.weight.add_(torch.randn_like(expert.weight) * (i + 1))
        for group in self.model.groups:
            exported = self.model.export(group)
            torch.testing.assert_close(exported(self.x), self.model(self.x, group))
            self.assertLess(parameters(exported), parameters(self.model))
            with torch.no_grad():
                exported.experts[0].weight.add_(1)
            self.assertFalse(torch.equal(exported.experts[0].weight,
                            self.model.experts[self.model.groups[group][0]].weight))

    def test_excluded_experts_receive_no_gradients(self):
        self.model(self.x, "coding").square().mean().backward()
        for i in self.model.groups["agentic"]:
            self.assertIsNone(self.model.experts[i].weight.grad)
        self.assertGreater(self.model.router.weight.grad.abs().sum().item(), 0)
        self.assertEqual(self.model.router.weight.grad[4:].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
