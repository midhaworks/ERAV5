# Generated M2 pretrained baseline report

Model: `mlx-community/Qwen3.5-4B-4bit`; revision `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`.
Greedy, non-thinking, community 4-bit quantization; authored development validation tasks.

| Task mode | Passed |
|---|---:|
| coding_direct | 5/6 |
| coding_reworded | 2/6 |
| repair_direct | 5/6 |
| repair_reworded | 5/6 |

Overall: 17/24 (70.8%).
Generation throughput including prompt processing: 25.25 tokens/s.
Peak MLX allocation: 2.49 GiB (does not include total system RAM).
Aggregate generation time: 55.0 seconds; loading/download excluded.

Failure counts:

* ValueError: Only plain solve function allowed: 3
* wrong_answer: 4

Raw responses remain unchanged. Audit re-executes them against their tests and also checks rotation input preservation.
The initial smoke checker rejected valid set.add calls; initial_smoke_restricted.json preserves that diagnostic.
The corrected checker permits ordinary collection methods and still excludes imports and introspection.
A wrong function name remains a failed API contract; the report does not rename it or substitute a reference solution.
This is narrow generation/direct-repair validation, not multi-step agent evaluation or a locked final benchmark.
No training, MoE conversion, expert quality gain or production readiness is demonstrated by this run.
