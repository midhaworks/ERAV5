# Serialized versus bundled-read repair profiles

Same pinned model, package versions, three task repositories and generation limits.
Bundled mode permits up to eight read_file calls in a turn; mutations/tests/finish remain
single. Reads execute sequentially; this reduces model turns, not filesystem concurrency.
Prefix-cache reuse is **disabled** because its separate exact-output gate failed.

Hidden-graded completion: serialized **3/3**, bundled **3/3**.
Both raw-action/file/test audits independently replay.

| Repository | Serialized | Bundled | Serialized turns | Bundled turns |
|---|---|---|---:|---:|
| shipping | PASS | PASS | 13 | 6 |
| pagination | PASS | PASS | 11 | 11 |
| ledger | PASS | PASS | 11 | 7 |

| Measured quantity | Serialized | Bundled |
|---|---:|---:|
| Model turns | 35 | 24 |
| Accepted tool calls | 30 | 32 |
| Protocol-invalid turns | 5 | 1 |
| Bundled read turns | 0 | 3 |
| Generated tokens | 1646 | 1424 |
| Generation seconds | 378.98 | 259.43 |
| Episode seconds incl. tools | 381.55 | 262.39 |
| Generated tokens/s | 4.34 | 5.49 |
| Peak MLX allocation (GiB) | 3.55 | 3.63 |

Observed episode-time ratio: **1.45x**;
bundled time is **31.2%** lower in these runs.
This is a single end-to-end observation per profile, not a controlled kernel speedup or
repeatability claim. The system instruction and accepted-tool policy differ; generated
outputs, token counts and subsequent prompts can differ too. There is no confidence
interval, matched-output workload, randomized run order, or production-scale conclusion.
Episode time includes inference and tools but excludes model loading. Peak MLX allocation
is cumulative within each run, not total system memory. Tasks are reused local authored
qualification fixtures, not independent held-out production evaluation or SWE-bench.

`comparison.json` derives every number from audited traces and records their hashes.
Old results are retained unchanged; the new profile lives in `bundled_run/`.
This is a harness efficiency experiment, not dense/MoE training or a model-quality gain.
