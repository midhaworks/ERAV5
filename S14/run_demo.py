"""Session 14: train a linear model, convert it to a small MoE, and continue training.

The task is deliberately small and synthetic.  The first input coordinate selects one
of two linear rules, so a single linear layer must compromise while a routed collection
of experts can learn both rules.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


class DenseLinear(nn.Module):
    """The starting model: one ordinary linear layer."""

    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TopKMoE(nn.Module):
    """A pedagogical top-k mixture of linear experts.

    The router selects k experts for each example.  Only those selected expert outputs
    are combined, with router probabilities renormalized over the selected experts.
    This implementation uses a Python loop for readability; it is not a production
    fused MoE kernel.
    """

    def __init__(self, input_dim: int, output_dim: int, experts: int = 4, top_k: int = 2):
        super().__init__()
        if not 1 <= top_k <= experts:
            raise ValueError("top_k must be between 1 and experts")
        self.input_dim, self.output_dim = input_dim, output_dim
        self.experts, self.top_k = experts, top_k
        self.router = nn.Linear(input_dim, experts)
        self.expert_layers = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) for _ in range(experts)]
        )

    @classmethod
    def from_dense(cls, dense: DenseLinear, experts: int = 4, top_k: int = 2) -> "TopKMoE":
        """Convert a trained dense layer by cloning its weights into every expert."""
        out = cls(dense.linear.in_features, dense.linear.out_features, experts, top_k)
        with torch.no_grad():
            # Give positive/negative values of x[0] an initially interpretable split.
            # This still leaves all experts trainable and lets the demo converge quickly.
            out.router.weight.zero_()
            signs = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=out.router.weight.dtype)
            out.router.weight[:, 0] = signs[:experts]
            out.router.bias.zero_()
            for expert in out.expert_layers:
                expert.weight.copy_(dense.linear.weight)
                expert.bias.copy_(dense.linear.bias)
                expert.weight.add_(0.02 * torch.randn_like(expert.weight))
                expert.bias.add_(0.02 * torch.randn_like(expert.bias))
        return out

    def forward(self, x: torch.Tensor, return_router: bool = False):
        router_probs = self.router(x).softmax(dim=-1)
        top_weights, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        output = x.new_zeros((x.shape[0], self.output_dim))
        for expert_id, expert in enumerate(self.expert_layers):
            selected = top_indices == expert_id
            rows = selected.any(dim=-1)
            if rows.any():
                # An example can select this expert once because topk indices are unique.
                slot = selected[rows].nonzero(as_tuple=False)[:, 1]
                values = expert(x[rows]) * top_weights[rows, slot].unsqueeze(-1)
                output[rows] += values
        if return_router:
            return output, top_indices, router_probs
        return output


def make_data(seed: int = 14, n_train: int = 4096, n_val: int = 1024):
    g = torch.Generator().manual_seed(seed)
    input_dim = 8
    w_positive = torch.tensor([2.0, 1.0, -0.7, 0.4, 0.2, -0.3, 0.5, 0.8])
    w_negative = torch.tensor([-2.0, -0.4, 1.1, -0.6, 0.7, 0.2, -0.8, -0.5])

    def sample(n: int):
        x = torch.randn(n, input_dim, generator=g)
        positive = x[:, :1] >= 0
        y_pos = x @ w_positive
        y_neg = x @ w_negative
        y = torch.where(positive[:, 0], y_pos, y_neg).unsqueeze(-1)
        y = y + 0.03 * torch.randn(n, 1, generator=g)
        return x, y

    return (*sample(n_train), *sample(n_val))


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float(F.mse_loss(model(x), y))


def train_dense(model, x, y, steps: int, lr: float, trace, log_every: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for step in range(1, steps + 1):
        loss = F.mse_loss(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            value = float(loss.detach())
            trace.append({"phase": "dense", "step": step, "train_mse": value})
            print(f"[dense] step {step:4d}/{steps} train_mse {value:.6f}")


def train_moe(model, x, y, steps: int, lr: float, trace, log_every: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for step in range(1, steps + 1):
        prediction = model(x)
        loss = F.mse_loss(prediction, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            value = float(loss.detach())
            trace.append({"phase": "moe", "step": step, "train_mse": value})
            print(f"[moe]   step {step:4d}/{steps} train_mse {value:.6f}")


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    random.seed(14)
    torch.manual_seed(14)
    ARTIFACTS.mkdir(exist_ok=True)
    x_train, y_train, x_val, y_val = make_data()
    dense = DenseLinear(x_train.shape[1])
    trace = []

    print("S14: dense linear model -> top-k mixture of experts")
    print(f"data train={tuple(x_train.shape)} validation={tuple(x_val.shape)} target=piecewise-linear")
    print(f"dense parameters={parameter_count(dense)}")
    train_dense(dense, x_train, y_train, steps=500, lr=0.05, trace=trace, log_every=100)
    dense_train = evaluate(dense, x_train, y_train)
    dense_val = evaluate(dense, x_val, y_val)

    moe = TopKMoE.from_dense(dense, experts=4, top_k=2)
    converted_train = evaluate(moe, x_train, y_train)
    converted_val = evaluate(moe, x_val, y_val)
    print(f"converted MoE parameters={parameter_count(moe)}")
    print(f"conversion checkpoint train_mse={converted_train:.6f} val_mse={converted_val:.6f}")
    train_moe(moe, x_train, y_train, steps=800, lr=0.02, trace=trace, log_every=100)
    moe_train = evaluate(moe, x_train, y_train)
    moe_val = evaluate(moe, x_val, y_val)

    moe.eval()
    with torch.no_grad():
        _, indices, probs = moe(x_val, return_router=True)
    counts = torch.bincount(indices.reshape(-1), minlength=moe.experts).tolist()
    result = {
        "seed": 14,
        "task": "piecewise linear regression (sign of x[0] selects one of two rules)",
        "train_examples": int(x_train.shape[0]),
        "validation_examples": int(x_val.shape[0]),
        "input_dim": int(x_train.shape[1]),
        "output_dim": 1,
        "dense_steps": 500,
        "moe_steps_after_conversion": 800,
        "experts": moe.experts,
        "top_k": moe.top_k,
        "dense_parameters": parameter_count(dense),
        "moe_parameters": parameter_count(moe),
        "dense_train_mse": dense_train,
        "dense_validation_mse": dense_val,
        "moe_conversion_train_mse": converted_train,
        "moe_conversion_validation_mse": converted_val,
        "moe_final_train_mse": moe_train,
        "moe_final_validation_mse": moe_val,
        "moe_validation_topk_assignments": counts,
        "moe_validation_router_entropy": float((-(probs * probs.clamp_min(1e-9).log()).sum(-1)).mean()),
        "dense_reduced_loss": dense_train < trace[0]["train_mse"],
        "moe_reduced_loss_after_conversion": moe_train < converted_train,
        "moe_beats_dense_validation": moe_val < dense_val,
    }
    (ARTIFACTS / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (ARTIFACTS / "loss_trace.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phase", "step", "train_mse"])
        writer.writeheader()
        writer.writerows(trace)
    print("\nEvidence:")
    print(json.dumps(result, indent=2))
    print(f"wrote {ARTIFACTS / 'results.json'}")
    print(f"wrote {ARTIFACTS / 'loss_trace.csv'}")


if __name__ == "__main__":
    main()
