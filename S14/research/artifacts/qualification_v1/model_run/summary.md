# Bounded tool-feedback repair evaluation

Model: `mlx-community/Qwen3.5-4B-4bit`, pinned revision `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`.
Greedy, non-thinking; at most 10 turns and
384 generated tokens per turn. No training.

Hidden-graded completion: **5/6**. Native read/edit/test/finish tool
calls operate on a single virtual file. Edits are actually executed in an isolated,
restricted Python child. Passing tests are invalidated after every edit. Hidden tests
are evaluated only for reporting and are never returned to the model.

| Task | Completion | Turns | Failed visible test calls | Malformed calls |
|---|---|---:|---:|---:|
| merge_intervals | FAIL | 6 | 1 | 0 |
| transpose | PASS | 4 | 0 | 0 |
| pair_sum | PASS | 4 | 0 | 0 |
| overlap_matches | PASS | 4 | 0 | 0 |
| group_records | PASS | 4 | 0 | 0 |
| grid_neighbors | PASS | 5 | 1 | 0 |

Final failure records (unchanged grading):

- `merge_intervals`: {"passed": false, "error": "wrong_answer", "actual": [[-4, 1], [5, 7]], "expected": [[-4, -2], [-1, 1], [5, 7]]}

Generated tokens: 1674; generation wall time (including prompt processing):
207.84s; 8.05 tokens/s.
Total episode time including tools: 210.17s.
Peak MLX allocation: 3.49GiB, not total system memory.
Successful tasks with a recorded failed-visible-test/retry sequence: 1.
Each turn reprocesses its complete chat history; this harness does not yet optimize
cross-turn KV-cache reuse, and speed is not a production-serving claim.

`audit.json` independently replays recorded tool-state transitions, reexecutes tests,
and checks final file contents and grading against `results.json`. Raw responses and
tool feedback are retained. These are new authored qualification families frozen before this run, with no exact task/code overlap against the prior six local development tasks. Pretraining contamination is unknown; this is not an independent third-party or repository-level benchmark. It is not real
repository-level agentic evaluation. Extra turns/test feedback change the inference
budget, so scores are not a compute-matched comparison to the one-shot baseline.
