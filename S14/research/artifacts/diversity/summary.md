# Generated instruction-diversity ablation

Same architectures, task-pair splits, LR search and approximate FLOP budgets.
The held-out wording family is absent from training and validation.

| Model | Training wording | Standard coding | Standard repair | Held-out coding | Held-out repair |
|---|---|---:|---:|---:|---:|
| dense_continuation | Original only | 52.8% | 30.6% | 0.0% | 0.0% |
| dense_continuation | Three families | 16.7% | 16.7% | 0.0% | 0.0% |
| unrestricted_moe | Original only | 38.9% | 33.3% | 0.0% | 0.0% |
| unrestricted_moe | Three families | 5.6% | 8.3% | 0.0% | 0.0% |
| grouped_moe | Original only | 63.9% | 38.9% | 0.0% | 0.0% |
| grouped_moe | Three families | 13.9% | 5.6% | 0.0% | 0.0% |

All percentages are averages across seeds 14, 27 and 42 on 12 task pairs per role.
New wording contains no unseen training bytes. Grouped routing still uses oracle roles.
This is a synthetic fixed read/edit/test environment, with no retries or repository execution.
Compute parity uses an analytical estimate; dispatch, optimizer and hardware effects are excluded.
Instruction diversification triples prompt variants, not underlying task pairs or unique solutions.
Previous control checkpoints are reused, with their hashes recorded, and tested on the same new prompts.
Results do not establish statistical significance or a 30B scaling recipe.
