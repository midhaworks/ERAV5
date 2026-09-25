# S13 — 20M-parameter training and reversible comparison

Run from the repository root:

```bash
python3 S13/train.py --variant baseline --batch-size 8
python3 S13/train.py --variant euler --batch-size 8 --corpus S13/data.txt
python3 S13/train.py --variant euler --auto-max-batch --corpus S13/data.txt
```

The target is 50,000,000 tokens. Use `--max-steps` only for a smoke test; without it the
script computes the steps required to reach the target. The Colab notebooks call this same
script and are the intended submission entry points.

## Model and reversible variant

The model is a byte-level causal language model with vocabulary 256, sequence length 256,
width 512, and six blocks. `BaselineLM` is a conventional pre-norm causal self-attention
model. `EulerReversibleLM` uses six additive-coupling Euler blocks:

```text
y1 = x1 + dt*f(x2)
y2 = x2 + dt*g(y1)
```

The map is invertible by evaluating `g` and `f` in reverse order. This implementation keeps
ordinary autograd tensors for clarity; it does **not** claim activation-rematerialized
reversible backward until that is separately implemented and benchmarked. The included byte
text corpus is intentionally tiny and is sampled by contiguous windows; replace it with a
larger licensed corpus for a meaningful language-quality result. Without `--corpus`, a
deterministic random byte stream is used only for plumbing smoke tests.

Both variants are approximately 20M parameters; the exact count is emitted in each JSON
result. Each run performs real forward, next-token cross-entropy, backward, and AdamW steps.

## Measurements

Each run writes JSON under `S13/artifacts/` with final/initial loss, exact tokens seen,
whether 50M was reached, elapsed time, tokens/second, peak memory, parameter count, batch,
sequence length, and loss trace. CUDA peak memory uses `torch.cuda.max_memory_allocated`; CPU
uses the process high-water mark.

Training prints live progress at step 1, every 100 steps, and the final step. Each progress
line includes step/total, tokens seen/target, current loss, tokens/second, and peak memory.
Use `--log-every 10` for more frequent updates or a larger value for quieter output.

`--auto-max-batch` probes powers of two until allocation fails, records every pass/fail, and
then trains at the largest passing batch. This is a hardware-dependent capacity result,
not a guessed GPU number.

## Important limitation

The current environment does not have PyTorch installed, so this repository does not
fabricate a 50M-token loss, throughput, or memory claim. Run the notebooks on Colab or a
machine with PyTorch; the generated JSON values are then the authoritative report. A true
production result should commit those generated artifacts with the README update.

The currently downloaded JSON files are marked `bounded_smoke_or_partial` and contain
`49,806,600` or `49,808,640` loss-bearing tokens. They are not a completed 50M-token
submission. This exposed and fixed an accounting bug: the old step calculation used 256
input positions, while causal cross-entropy contributes only 255 targets. The corrected
script uses `batch_size × (sequence_length - 1)` when calculating the required steps. The
three experiments must be rerun after pulling this fix; only artifacts with
`status: completed_target` should be reported as final.

## Notebooks

- [`S13_baseline.ipynb`](S13_baseline.ipynb): fixed-batch baseline.
- [`S13_reversible_fixed.ipynb`](S13_reversible_fixed.ipynb): Euler reversible at the same batch.
- [`S13_reversible_max_batch.ipynb`](S13_reversible_max_batch.ipynb): capacity probe and maximum-batch run.
