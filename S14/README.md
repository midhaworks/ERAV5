# Session 14 — Linear model converted to a Mixture of Experts

This is a small, runnable demonstration of the assignment:

1. Train one ordinary `nn.Linear` model.
2. Convert that trained linear layer into four linear experts.
3. Add a learned router that selects the top two experts for each example.
4. Continue training after conversion and verify that the loss keeps falling.

Run it from the repository root:

```bash
python3 S14/run_demo.py
```

The run writes generated evidence to `S14/artifacts/results.json` and
`S14/artifacts/loss_trace.csv`.

## What the data teaches

Each input has eight features. The sign of feature `x[0]` chooses one of two different
linear rules:

```text
if x[0] >= 0: y = x · w_positive
else:         y = x · w_negative
```

A single linear layer cannot represent both slopes at once, so it learns a compromise.
The MoE can assign different examples to different experts and represent both rules.
This is synthetic data, intentionally chosen to make the architectural change visible;
it is not a claim about language-model quality.

## How conversion works

The dense model starts with one matrix `W` and bias `b`. Conversion creates four expert
copies initialized from that trained `W, b` (with a tiny deterministic perturbation), then
adds a router:

```text
router_logits = Router(x)
selected = top-2(router_logits)
output = p1 * Expert[selected_1](x) + p2 * Expert[selected_2](x)
```

The router probabilities are renormalized over the selected experts. This is a real
top-k dispatch, not a hardcoded answer. The reference implementation loops over experts
for clarity; a production implementation would use fused grouped dispatch and capacity
limits. Four experts with top-2 means each example uses two of the four experts.

The conversion is not expected to preserve the exact output: the expert perturbation and
router selection make the conversion checkpoint explicit. The important test is that
the converted model continues training and reduces MSE after conversion.

## Evidence to inspect

`results.json` records the seed, shapes, parameter counts, losses at all checkpoints,
router assignment counts, and boolean checks. `loss_trace.csv` is suitable for plotting
the dense and post-conversion MoE curves.

The expected evidence checks are:

| Requirement | Evidence |
|---|---|
| Dense model trains | `dense_reduced_loss` and dense trace |
| Conversion is explicit | `moe_conversion_*` losses and parameter counts |
| MoE continues training | `moe_reduced_loss_after_conversion` |
| MoE improves held-out data | `moe_beats_dense_validation` |
| Routing is active | `moe_validation_topk_assignments` |

## Observed run

With seed 14 on the included deterministic synthetic data, the generated evidence reports:

| Checkpoint | Train MSE | Validation MSE |
|---|---:|---:|
| Dense linear after 500 steps | 3.85889 | 4.34968 |
| Immediately after conversion | 3.85019 | 4.33489 |
| MoE after 800 further steps | 0.11739 | 0.16663 |

The MoE uses all four experts over the validation set (`525, 430, 535, 558` selected
assignments when counting the two selected experts per example). These numbers are a
reproducible local run; rerunning the command regenerates the JSON and CSV evidence.

Parameter counts are intentionally different: one linear layer has 9 parameters, while
four experts plus a router have 72 (4 × 9 expert parameters plus 36 router parameters).
The purpose is to demonstrate conditional
specialization and continued optimization, not parameter compression.
