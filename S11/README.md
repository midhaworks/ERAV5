# S11 — Optimizer mechanics, schedules, and tuning evidence

Run the complete local experiment from the repository root:

```bash
python3 S11/run_demo.py
```

The run writes generated evidence to `S11/artifacts/`. It is CPU-sized and deterministic
(seed `1111`), but uses actual PyTorch optimizers, gradients, parameter updates, plots, and
training loops. The JSON and log are authoritative; summary values below are intentionally
derived from that run.

## Adam reproduced by hand

For one scalar weight and gradients `[0.1, -0.2, 0.05, 0.3, -0.1]`, the script computes
`m`, `v`, bias-corrected `m_hat` and `v_hat`, and the Adam step from the recurrence:

```text
m_t = beta1*m_(t-1) + (1-beta1)*g_t
v_t = beta2*v_(t-1) + (1-beta2)*g_t^2
m_hat = m_t/(1-beta1^t)
v_hat = v_t/(1-beta2^t)
step = learning_rate*m_hat/(sqrt(v_hat)+epsilon)
```

It then drives a real `torch.optim.Adam` scalar parameter with the same gradients and
compares every field. `artifacts/evidence.json` stores all five rows from both paths and
the maximum absolute error for each field. The comparison passes at floating-point
precision, rather than comparing only the final weight.

In the generated run, the largest absolute discrepancies were `7.49e-09` for `m_hat`,
`2.54e-09` for `v_hat`, and `3.79e-11` for the resulting step.

## Bias correction

The first 20 updates are plotted in
[`artifacts/bias_correction.png`](artifacts/bias_correction.png). The script defines
“stops mattering” explicitly as three consecutive steps whose absolute difference is below
1% of the corrected update. The measured step is in
`evidence.json::bias_correction::difference_stops_mattering_after_step`; the threshold and
both complete trajectories are stored beside it.
For this beta configuration the measured step is `3,927`; bias correction is therefore
still materially changing the update during the plotted first 20 steps.

## Warmup and update-to-weight ratios

Every optimizer step records `||parameter_update|| / ||parameter_before||` for every
parameter layer in `evidence.json::layer_update_ratios::rows`. The schedule has 20 linear
warmup steps; the measured warmup boundary is recorded as `warmup_end_step=20`, where the
learning rate reaches its plateau and the ratio stops being warmup-scaled. This is a
measurement of this configured run, not a universal optimizer constant.

## Cosine versus WSD

The same width-256 model, initialization, data, optimizer and base learning rate is trained
for 300 steps under each schedule. Before that comparison, each schedule is tuned over
the same three candidate base rates (`0.001`, `0.002`, `0.003`) for 100 steps; the selected
rate for each side is stored under `evidence.json::schedule_comparison::tuning`. Cosine
decays continuously after warmup. WSD warms up,
holds a stable plateau through step 200, then decays. The comparison reports both losses at
step 200 (the requested decision point) and final step 300 losses. The `keep_at_step_200`
field states which one is retained; both arms therefore receive symmetric tuning treatment
before the comparison is accepted.

The tuning stage selected `0.003` for both schedules. At step 200, cosine loss is
`0.03296839` and WSD loss is `0.02065998`; the recorded choice is **WSD**. At step 300 the
losses are `0.02943053` and `0.01152323`, respectively.

## Learning-rate sweep and width 4096 decision

Widths 256, 512 and 1,024 are each evaluated at four learning rates. The measured curves
and highlighted minima are in [`artifacts/lr_sweep.png`](artifacts/lr_sweep.png), while all
individual results and minima are in `evidence.json::learning_rate_sweep`. The proposed
width-4,096 value is selected from the measured minima, but its confidence is explicitly
`low`: width 4,096 was not directly run, so this is an extrapolation and should be tuned
again before a large experiment.

The current sweep minima are:

| Width | Best measured learning rate | Loss after 60 steps |
|---:|---:|---:|
| 256 | 0.003 | 0.05584238 |
| 512 | 0.003 | 0.04584222 |
| 1,024 | 0.003 | 0.03434286 |

The extrapolated width-4,096 choice is `0.003`, with low confidence because that width was
not directly measured.

## Honest interpretation

MFU is not claimed here because this assignment asks optimizer/schedule evidence rather
than a hardware-throughput claim. The comparison is only accepted when both sides use the
same initialization, data, optimizer family and training budget; schedule differences are
the intended independent variable. Generated files are:

- [`run.log`](artifacts/run.log): concise sequence of measured events.
- [`evidence.json`](artifacts/evidence.json): complete machine-readable records.
- [`bias_correction.png`](artifacts/bias_correction.png): corrected versus uncorrected Adam.
- [`lr_sweep.png`](artifacts/lr_sweep.png): width/LR curves with minima marked.
