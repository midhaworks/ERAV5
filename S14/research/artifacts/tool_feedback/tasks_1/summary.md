# Bounded tool-feedback repair evaluation

Model: `mlx-community/Qwen3.5-4B-4bit`, pinned revision `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`.
Greedy, non-thinking; at most 10 turns and
384 generated tokens per turn. No training.

Hidden-graded completion: **1/1**. Native read/edit/test/finish tool
calls operate on a single virtual file. Edits are actually executed in an isolated,
restricted Python child. Passing tests are invalidated after every edit. Hidden tests
are evaluated only for reporting and are never returned to the model.

| Task | Completion | Turns | Failed visible test calls | Malformed calls |
|---|---|---:|---:|---:|
| stable_unique | PASS | 4 | 0 | 0 |

Generated tokens: 175; generation wall time (including prompt processing):
22.16s; 7.90 tokens/s.
Total episode time including tools: 22.53s.
Peak MLX allocation: 3.15GiB, not total system memory.
Successful tasks with a recorded failed-visible-test/retry sequence: 0.
Each turn reprocesses its complete chat history; this harness does not yet optimize
cross-turn KV-cache reuse, and speed is not a production-serving claim.

`audit.json` independently replays recorded tool-state transitions, reexecutes tests,
and checks final file contents and grading against `results.json`. Raw responses and
tool feedback are retained. This is **reused development data**, not an independent
held-out benchmark; hidden means withheld during this episode. It is not real
repository-level agentic evaluation. Extra turns/test feedback change the inference
budget, so scores are not a compute-matched comparison to the one-shot baseline.
