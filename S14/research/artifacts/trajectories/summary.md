# Generated read/edit/test pilot report

Synthetic arithmetic code tasks, oracle role labels, approximate FLOP budgets.
Learning rates 0.001 and 0.002 selected by held-out validation NLL only.

| Seed | Model | LR | Steps | FLOP budget used | Coding pass | Three-action repair pass |
|---:|---|---:|---:|---:|---:|---:|
| 14 | dense_continuation | 0.001 | 300 | 100.00% | 5/12 | 2/12 |
| 14 | unrestricted_moe | 0.001 | 235 | 99.57% | 3/12 | 1/12 |
| 14 | grouped_moe | 0.002 | 235 | 99.57% | 8/12 | 3/12 |
| 27 | dense_continuation | 0.002 | 300 | 100.00% | 6/12 | 5/12 |
| 27 | unrestricted_moe | 0.002 | 235 | 99.65% | 6/12 | 8/12 |
| 27 | grouped_moe | 0.002 | 235 | 99.65% | 9/12 | 6/12 |
| 42 | dense_continuation | 0.002 | 300 | 100.00% | 8/12 | 4/12 |
| 42 | unrestricted_moe | 0.002 | 235 | 99.92% | 5/12 | 3/12 |
| 42 | grouped_moe | 0.002 | 235 | 99.92% | 6/12 | 5/12 |

| Model | Mean coding success | Mean repair success |
|---|---:|---:|
| dense_continuation | 52.8% | 30.6% |
| unrestricted_moe | 38.9% | 33.3% |
| grouped_moe | 63.9% | 38.9% |

Evidence scope:

* A repair succeeds only after generated read, edit and test actions and passing arithmetic tests.
* Actual generated patches determine test results; no reference patch is substituted.
* Shared pretraining cost is recorded separately. Both LR trials are charged in total_search_flops_proxy.
* FLOP proxy covers attention, active FFNs, routers and vocabulary projection with backward estimated at twice forward.
* It excludes optimizer, activation and dispatch costs; wall time is separate. This is not measured GPU compute parity.
* Each role has only 12 correlated synthetic test cases; three seeds do not establish statistical significance.
* Exports match role-masked parent logits; role selection is an oracle and generalization to repositories is untested.
