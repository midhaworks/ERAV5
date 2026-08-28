# S9 — Correct and observable next-token training

This submission replaces the assignment notebook with a reproducible local PyTorch
program. Run it from the repository root:

```bash
python3 S9/run_demo.py
```

It generates `S9/artifacts/run.log`, `evidence.json` and `evidence.md`. The program uses a
small real four-document corpus, a deterministic whitespace vocabulary, and a one-layer
causal Transformer. It is intentionally small so every invariant is inspectable on a CPU.

## What is demonstrated

The harness prints every tensor shape with dimension meanings and prints shifted token
strings (not only integer ids). It verifies that inputs are `t` and targets are `t+1`,
masks padding, packs two documents while masking the cross-document target, reports the
loss before and after that firewall, computes untrained perplexity, and compares tied and
untied output-head parameters.

The memory experiment runs ordinary cross-entropy and the custom `chunked_ce` function on
the same logits and asserts equal loss. It reports the measured materialized-logit peak
bytes and their ratio; this is the activation-memory comparison, not a claim about the
model's entire process RSS. The second head predicts `t+2`; its initial loss, final loss
and sum are recorded separately, making the extra-horizon learning behaviour observable.

## Measured results from the generated run

| Check | Measured result |
|---|---:|
| Vocabulary / untrained perplexity | 21 / 27.3055 |
| Padding count, unmasked → masked | 20 → 18 contributing targets |
| Packed loss before → after boundary mask | 3.047885 → 3.142696 |
| Tied head incremental parameters / untied head parameters | 0 / 1,008 |
| Ordinary / chunked materialized logits | 21,504 / 2,688 bytes (8.00×) |

The packed loss changes because the unmasked calculation trains on one transition from
the final token of document 1 into the first token of document 2. That transition is not a
language-model target; after removing it, only within-document targets remain. The numeric
direction of the change is data/model dependent—the invariant is that exactly one
cross-document contribution is removed.

The tied-head number is incremental: a tied head reuses `embedding.weight`, while the
untied `21 × 48` classifier would add 1,008 learned weights. It is not claiming that the
entire tied model has zero parameters.

For the two-horizon objective, the first-head loss falls from `3.319419` to `0.000127`,
and the second-head (`t+2`) loss falls from `3.126112` to `0.000079`; their final sum is
`0.000206`. The second loss starts slightly lower and ends slightly lower here because
this tiny deterministic corpus makes the two-step patterns easier for this model. Both
heads are nevertheless trained against their own correctly shifted targets, with separate
valid-token masks.

## Evidence

`S9/artifacts/evidence.json` is generated from the run's measured values. The companion
`evidence.md` is a compact tabular index, while `run.log` contains the shape lines,
human-readable shift strings, mask counts, boundary losses, perplexity, parameter counts,
memory ratio and two-head training losses. No result is hard-coded in the evidence files.

The expected sanity checks are that untrained perplexity is close to the vocabulary size,
masked padding contributes fewer tokens than the unmasked calculation, the packed boundary
removes exactly one loss term, and both trained losses decrease. A failure raises an
exception or records `FAIL` instead of silently producing a plausible curve.
