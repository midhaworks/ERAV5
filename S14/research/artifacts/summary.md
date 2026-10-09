# Generated synthetic pilot results

Each role has 12 held-out operation/constant pairs per seed. These are
arithmetic generation and one edit-action tasks, not repository-level agent tasks.

| Seed | Model | Parameters | Coding successes | Edit-action successes | Test NLL |
|---:|---|---:|---:|---:|---:|
| 14 | dense_continuation | 58,080 | 8/12 | 6/12 | 0.02343 |
| 14 | unrestricted_moe | 114,632 | 10/12 | 8/12 | 0.01957 |
| 14 | grouped_moe | 114,632 | 9/12 | 6/12 | 0.02857 |
| 27 | dense_continuation | 58,080 | 12/12 | 11/12 | 0.00752 |
| 27 | unrestricted_moe | 114,632 | 11/12 | 11/12 | 0.01061 |
| 27 | grouped_moe | 114,632 | 12/12 | 10/12 | 0.01425 |
| 42 | dense_continuation | 58,080 | 12/12 | 9/12 | 0.01602 |
| 42 | unrestricted_moe | 114,632 | 12/12 | 8/12 | 0.02186 |
| 42 | grouped_moe | 114,632 | 12/12 | 8/12 | 0.02302 |

Mean success across seeds (small correlated task set; no significance claim):

* dense_continuation: coding 88.9%, edit action 72.2%.
* unrestricted_moe: coding 91.7%, edit action 75.0%.
* grouped_moe: coding 91.7%, edit action 66.7%.

Comparisons are token-matched; MoE activates two FFN experts per token versus
one FFN in the dense control. Training compute is therefore not matched.
Group choice is supplied by the task role. Export agreement establishes
equivalence to the masked parent, not quality retention from the unrestricted parent.
Shared attention, embeddings and head are duplicated in each independent export.
