# Frozen multi-file repository repair

Checkpoint: `mlx-community/Qwen3.5-4B-4bit`, revision `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`. No training or new download.
Three authored miniature repositories, each with two initial bugs in separate modules.
Limits frozen before inference: 16 turns, 512 generated tokens per turn,
greedy non-thinking decoding. No shell or external dependencies.

Hidden-graded completion: **3/3**.

| Repository | Completion | Turns | Edited modules | Failed visible calls | Protocol-invalid turns |
|---|---|---:|---:|---:|---:|
| shipping | PASS | 13 | 2 | 2 | 3 |
| pagination | PASS | 11 | 2 | 2 | 1 |
| ledger | PASS | 11 | 2 | 1 | 1 |

No failed tasks.

Generated tokens: 1646; generation time including full chat reprocessing: 378.98s;
**4.34 tokens/s**. Peak MLX allocation: **3.55GiB**, not total system memory.
Saved `audit.json` re-parses every raw call, reconstructs real temporary files, reruns visible
and hidden tests, and checks final file contents and grades. `run.log` is reconstructed
from these saved actions, not a separate original stdout capture.

Files are real and local imports execute; tests and README are read-only. Hidden integration
cases stay outside the repair workspace and never enter tool feedback. Edits invalidate a
previous pass. References pass all 24 tests, and reintroducing each of the six individual
bugs is detected before inference. Only controlled AST-limited Python is supported;
this is not a general security sandbox or arbitrary real-project environment.
The profile accepts exactly one tool call per turn. Several valid native calls in a response are rejected. The protocol-invalid count is NOT necessarily a broken-XML count.
Tool-profile restrictions are harness limitations, not a general parallel-tool capability measure.

This is local authored qualification, not SWE-bench, a third-party benchmark, or production
readiness. Pretraining contamination is unknown. There is no tuned training comparison,
MoE quality claim, or 30B scaling evidence. Cross-turn KV-cache reuse is not optimized.
