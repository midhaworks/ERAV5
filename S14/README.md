# Session 14 — Linear model converted to a Mixture of Experts

The newest [external evaluation preparation](research/artifacts/external_v1/summary.md)
pins the official MIT-licensed HumanEval source and freezes IDs 0–7 before inference.
Six measured macOS isolation probes pass, but the required memory-cap call fails on
this host. **No external model score exists:** candidate and reference execution remain
blocked, without an unrestricted fallback. A compatible memory-limited backend needs
to be selected. HumanEval is old/public, not a fresh uncontaminated SOTA benchmark.

The earlier [live bundled-read comparison](research/artifacts/repository_v1/comparison.md)
retains **3/3 hidden-graded repairs** with independent replay audits. Model turns decrease
from 35 to 24; observed episode time decreases from 381.55s to 262.39s (31.2% lower),
with 5.49 generated tokens/s and 3.63GiB peak MLX allocation in the bundled run. Peak
allocation is slightly higher than the serialized run. This is one observation per profile
on reused tasks, not a guaranteed speedup or model-quality improvement. Prefix caching
remains disabled. All 44 automated research tests pass.

The earlier [harness efficiency probe](research/artifacts/efficiency_v1/summary.md)
preserves the old results. All five historical bundled-read turns execute identically
to individual reads; the separate `--bundled-reads` profile is now measured above.
Prefix-cache reuse passes only **7/9 exact-output checks** (parsed
actions match 9/9), so it remains **disabled**. The two mismatches and timings are retained;
its timings are not an accepted cache speedup or model-quality improvement claim.

The earlier [multi-file repair run](research/artifacts/repository_v1/model_run/summary.md)
passes **3/3 authored miniature repositories**, repairing two modules per task and passing
hidden integration checks. Independent action/file/test replay matches. The M2 run measured
4.34 generated tokens/s and 3.55GiB peak MLX allocation. Five bundled-read turns were
rejected by the frozen one-call-per-turn protocol; they remain counted in the evidence.
Both serialization and repeated full-history processing are harness overhead, not model
architecture benchmarks. All 36 automated research tests pass. No pretrained MoE training
or production qualification is claimed.

The earlier [frozen qualification run](research/artifacts/qualification_v1/model_run/summary.md)
tests six new authored task families: **5/6 pass hidden checks**, with 27 valid tool calls,
8.05 generated tokens/s and 3.49GiB peak MLX allocation on the M2. All execution traces
and grading independently replay. Task data and limits were frozen before inference;
all six references pass and all 18 known-wrong implementations are rejected.
Interval merging fails the predeclared adjacent-endpoint case; its wording has an
interpretation risk, documented in the research README. No score was changed after
inspection. This is single-file qualification, not production or 30B-scale evidence.

The earlier [native tool-feedback evaluation](research/artifacts/tool_feedback/tasks_6/summary.md)
completed on the M2: **5/6 single-file repairs pass hidden checks**, with 26 tool turns,
6.97 generated tokens/s including repeated prompt processing and 3.23GiB peak MLX allocation.
One successful repair recovered after a real test failure. Rotation passed visible tests
but failed a hidden `k > len(xs)` case; that failure is preserved. Raw actions, actual test
feedback and independently replayed audits are saved. These are reused development tasks,
not evidence of repository-level agentic capability or 30B readiness.

The earlier [pretrained M2 baseline](research/artifacts/pretrained/summary.md) runs
Qwen3.5-4B in 4-bit MLX on the 8GiB Mac: 17/24 development coding/repair prompts pass,
with 25.25 tokens/s including prompt processing and 2.49GiB peak MLX allocation.
This establishes bounded inference feasibility; broader agentic capability and MoE
training feasibility still require measurement.

The additional [coding/agentic export research track](research/README.md) defines a
separate protocol toward a 30B-total sparse parent and smaller specialized exports.
Its layer conversion/export checks pass and a three-seed synthetic coding/edit-action
pilot has run. Real-world repository coding and broad agentic quality remain unmeasured;
the multi-file result above covers only three controlled miniature repositories.
The newer [three-action research pilot](research/artifacts/trajectories/summary.md)
uses generated read/edit/test actions, validation-selected learning rates, and analytical
compute budgets. Its environment is synthetic; results do not establish 30B-scale capability.
The [learned-selector experiment](research/artifacts/selectors/summary.md) removes oracle
role selection for inference and identifies parent failure on unfamiliar prompt wording.
The [instruction-diversity ablation](research/artifacts/diversity/summary.md) also failed
to improve unseen-wording success at the same estimated compute budget; this failure is
retained in the research evidence rather than counted as a capability improvement.

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
