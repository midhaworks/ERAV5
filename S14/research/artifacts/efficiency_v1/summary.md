# Repository harness efficiency probe

No training. Old repository results and score remain unchanged. This is a fixed-history
microbenchmark, not a new live repair run or a quality improvement.

Bundled reads: **5/5** historical rejected bundles now parse and
execute with exactly the same responses as individual reads, without modifying files.
Only read_file calls may be bundled (at most eight). Edits/tests/finish cannot be mixed in.
This fixed-history probe is not a live profile benchmark. Live results, when available,
are recorded separately under `repository_v1/bundled_run/` and `repository_v1/comparison.md`.

Prefix-cache exact-output gate: **FAIL**, 7/9 exact token/text/stop matches.
Parsed-action equality: 9/9, a separate diagnostic, NOT a replacement gate.
Cache reuse remains **disabled in the repair runner**. The cause of divergence has not
yet been isolated; numerical changes from prefill segmentation are a hypothesis, not a finding.

| Fixed prompt | Exact output | Parsed actions | Fresh seconds | Cached seconds | Reused prefix tokens |
|---|---|---|---:|---:|---:|
| shipping:1 | PASS | PASS | 3.97 | 4.43 | 0 |
| shipping:9 | FAIL | PASS | 14.01 | 17.43 | 680 |
| shipping:13 | PASS | PASS | 9.05 | 2.29 | 1414 |
| pagination:1 | PASS | PASS | 3.92 | 4.09 | 0 |
| pagination:7 | PASS | PASS | 10.57 | 7.41 | 689 |
| pagination:11 | PASS | PASS | 9.44 | 2.45 | 1246 |
| ledger:1 | PASS | PASS | 4.14 | 4.29 | 0 |
| ledger:7 | FAIL | PASS | 24.59 | 21.60 | 694 |
| ledger:11 | PASS | PASS | 15.06 | 3.67 | 1381 |

On the 7 exact-matching pairs ONLY, total fresh time is 56.15s and cached time
is 28.62s: 1.96x ratio. This is a postselected diagnostic subset, not a
full-workload speedup or evidence that the failed method should be deployed. Different
outputs make the remaining comparisons unequal work. One measurement per arm, alternating
order, shared warmup, no confidence interval. Timing includes prefix preparation/copying.
Peak MLX allocation across arms: 3.64GiB; persistent prefix state may be resident
during both arms, so this is not isolated per-method memory measurement.

The cache stores canonical history before the generation header, requires an exact token
prefix, resets on rewrites/shortening, and branches generation into a copy. Recurrent
state is never trimmed. Changed tokens/text count as failure even when code edits agree.
`plan.json` fixes nine historical prompts before inference; raw tokens/text, timings,
prefix counts and mismatches are retained in `results.json`. `audit.json` independently
recomputes equality/timing claims and reexecutes bundled reads on reconstructed files.
