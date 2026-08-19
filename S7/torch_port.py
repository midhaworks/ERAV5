"""PyTorch parity oracle for the NumPy RKE transformer."""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from rke import Adam, FullByteCodec, TinyTransformer, sha256


class TorchTinyTransformer(nn.Module):
    """Exact torch formulation of ``rke.TinyTransformer``."""

    def __init__(self, numpy_model: TinyTransformer, dtype: torch.dtype = torch.float64,
                 device: torch.device | str = "cpu"):
        super().__init__()
        self.slots = numpy_model.codec.slots
        self.width = numpy_model.codec.width
        self.d_slot = numpy_model.d_slot
        self.d_model = numpy_model.d_model
        for name, value in numpy_model.params.items():
            self.register_parameter(name, nn.Parameter(torch.tensor(value, dtype=dtype, device=device)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch = len(features)
        x = (features @ self.Ein).reshape(batch, 3, self.d_model) + self.pos.unsqueeze(0)
        q, k, v = x @ self.Wq, x @ self.Wk, x @ self.Wv
        scores = q @ k.transpose(1, 2) / math.sqrt(self.d_model)
        causal = torch.triu(torch.ones((3, 3), dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal.unsqueeze(0), -1e9)
        attention = torch.softmax(scores, dim=-1)
        hidden = x + (attention @ v) @ self.Wo
        slots = hidden[:, -1, :].reshape(batch, self.slots, self.d_slot)
        return slots @ self.Ein.T

    @staticmethod
    def masked_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        weights = mask.to(logits.dtype)
        return -(selected * weights).sum() / weights.sum().clamp_min(1.0)


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _one_adam_step(values: dict[str, np.ndarray], grads: dict[str, np.ndarray], lr: float) -> dict[str, np.ndarray]:
    output = {}
    for name, value in values.items():
        grad = np.clip(grads[name], -1.0, 1.0)
        # At step one, Adam's bias-corrected first moment is g and second
        # moment is g^2. This exactly matches rke.Adam from a zero state.
        output[name] = value - lr * grad / (np.sqrt(grad * grad) + 1e-8)
    return output


def run_parity(output: Path | None = None) -> dict[str, Any]:
    codec = FullByteCodec(4)
    numpy_model = TinyTransformer(codec, d_slot=2, seed=90210)
    records = [(b"a", "é".encode("utf-8"), "भारत".encode("utf-8")[:4]),
               (b"xy", b"z", b"q")]
    features = np.stack([np.stack([codec.encode(a), codec.encode(b), codec.encode(c)])
                         for a, b, c in records]).astype(np.float64)
    targets = np.stack([codec.ids(b"ab"), codec.ids("é".encode("utf-8"))])
    mask = targets != 0

    numpy_logits, _ = numpy_model.forward(features)
    numpy_loss, numpy_grads = numpy_model.loss_and_grads(features, targets, mask)
    torch_model = TorchTinyTransformer(numpy_model)
    torch_features = torch.tensor(features, dtype=torch.float64)
    torch_targets = torch.tensor(targets, dtype=torch.long)
    torch_mask = torch.tensor(mask, dtype=torch.bool)
    torch_logits = torch_model(torch_features)
    torch_loss = torch_model.masked_loss(torch_logits, torch_targets, torch_mask)
    torch_loss.backward()

    gradient_errors = {
        name: _max_abs(numpy_grads[name], parameter.grad.detach().cpu().numpy())
        for name, parameter in torch_model.named_parameters()
    }
    lr = .004
    expected_step = _one_adam_step({k: v.copy() for k, v in numpy_model.params.items()}, numpy_grads, lr)
    # Exercise the actual NumPy optimizer too, then independently perform the
    # same documented update on torch tensors.
    numpy_step_model = TinyTransformer(codec, d_slot=2, seed=90210)
    Adam(numpy_step_model.params, lr=lr).update(numpy_step_model.params, numpy_grads)
    numpy_optimizer_error = max(_max_abs(numpy_step_model.params[k], expected_step[k]) for k in expected_step)
    with torch.no_grad():
        for name, parameter in torch_model.named_parameters():
            grad = parameter.grad.clamp(-1.0, 1.0)
            parameter -= lr * grad / (torch.sqrt(grad * grad) + 1e-8)
    step_errors = {name: _max_abs(expected_step[name], parameter.detach().cpu().numpy())
                   for name, parameter in torch_model.named_parameters()}

    tolerance = 1e-9
    checks = {
        "forward_logits": _max_abs(numpy_logits, torch_logits.detach().cpu().numpy()) <= tolerance,
        "masked_loss": abs(numpy_loss - float(torch_loss.detach())) <= tolerance,
        "all_gradients": max(gradient_errors.values()) <= tolerance,
        "numpy_optimizer_formula": numpy_optimizer_error <= tolerance,
        "one_optimizer_step": max(step_errors.values()) <= tolerance,
    }
    device_smoke: dict[str, Any] = {"mps_available": bool(torch.backends.mps.is_available()),
                                    "cuda_available": bool(torch.cuda.is_available())}
    if device_smoke["mps_available"]:
        try:
            mps_model = TorchTinyTransformer(numpy_model, dtype=torch.float32, device="mps")
            value = mps_model(torch.tensor(features, dtype=torch.float32, device="mps"))
            torch.mps.synchronize()
            device_smoke.update({"mps_forward": bool(torch.isfinite(value).all().cpu()), "mps_error": None})
        except Exception as exc:  # recorded as evidence, never silently passed
            device_smoke.update({"mps_forward": False, "mps_error": repr(exc)})
    else:
        device_smoke.update({"mps_forward": None, "mps_error": "device unavailable"})

    result = {
        "experiment": "NumPy-PyTorch RKE parity",
        "torch_version": torch.__version__, "python": platform.python_version(),
        "dtype": "float64", "tolerance": tolerance,
        "errors": {"logits_max_abs": _max_abs(numpy_logits, torch_logits.detach().cpu().numpy()),
                   "loss_abs": abs(numpy_loss - float(torch_loss.detach())),
                   "gradient_max_abs": max(gradient_errors.values()),
                   "gradient_by_parameter": gradient_errors,
                   "numpy_optimizer_max_abs": numpy_optimizer_error,
                   "optimizer_step_max_abs": max(step_errors.values()),
                   "optimizer_step_by_parameter": step_errors},
        "checks": checks, "device_smoke": device_smoke,
        "input_hash": sha256({"features": features.tolist(), "targets": targets.tolist(), "mask": mask.tolist()}),
    }
    result["passed"] = all(checks.values()) and device_smoke.get("mps_forward") is not False
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run_parity(Path(__file__).resolve().parent / "artifacts" / "torch_parity.json")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
