# Generated prompt-selector results

Same frozen grouped parents as the trajectory pilot. The selector receives prompt text only.
No final-test or shifted-template examples enter selector training or checkpoint selection.

| Evaluation | Coding success | Three-action repair success |
|---|---:|---:|
| oracle | 63.9% | 38.9% |
| learned | 63.9% | 38.9% |
| shifted_oracle | 0.0% | 0.0% |
| shifted_learned | 0.0% | 0.0% |

| Seed | Standard selector accuracy | Unseen wording accuracy | Shifted balanced accuracy |
|---:|---:|---:|---:|
| 14 | 48/48 | 48/48 | 100.0% |
| 27 | 48/48 | 48/48 | 100.0% |
| 42 | 48/48 | 48/48 | 100.0% |

This small classifier uses normalized byte counts; it cannot represent token order.
Group choice is fixed for each generation, then recomputed from observed state at the next action.
Matched-template classification does not establish general routing capability or expert specialization.
Overall accuracy is sensitive to the 3:1 action/coding class balance; balanced accuracy weights both roles equally.
Shifted prompts introduce no byte values absent from parent training data; wording, not byte coverage, changes.
Unseen wording changes the parent prompts too; shifted oracle separates parent brittleness from selector error.
Only 12 test cases per role are evaluated. Reported success is not production readiness.
