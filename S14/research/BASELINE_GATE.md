# Baseline qualification before more MoE training

Status: a pretrained Qwen3.5-4B 4-bit candidate has been downloaded and measured on the
8GiB M2. The [bounded validation report](artifacts/pretrained/summary.md) records 17/24
passes, about 25.3 tokens/s and 2.49GiB peak MLX allocation. General coding/agentic
qualification remains incomplete. Prior synthetic results are diagnostics, not scaling evidence.

The [native tool-feedback evaluation](artifacts/tool_feedback/tasks_6/summary.md) now
records 5/6 hidden-graded single-file repairs and one successful recovery after actual
test failure. All 26 calls parse correctly; execution traces and grades independently
replay. Rotation's large-k failure remains recorded. This is bounded development-data
evidence, not an independent or repository-level qualification. Freeze a new task set
and evaluation rules before further model adaptation; do not train to these six cases.

That freeze-and-run step has now completed on six new authored families:
[qualification report](artifacts/qualification_v1/model_run/summary.md). Predeclared
completion is 5/6; all 27 native calls parse, and state/test/grade replay matches. The
interval adjacency failure also exposes an ambiguous domain wording. Preserve the
score, but do not interpret it as a clean model defect or production qualification.
Next: unambiguous repository-level repair contracts and hidden integration tests.
No pretrained MoE conversion/training has been launched based on these narrow scores.

The [miniature repository run](artifacts/repository_v1/model_run/summary.md) now completes
3/3 two-module repairs with hidden integration checks and independent file/action/test
replay. It measures 4.34 generated tokens/s and 3.55GiB peak MLX allocation on the M2.
Five bundled-read turns violate the serialized profile and remain in the evidence.
This passes a bounded harness-mechanics gate, not real-world repository qualification.
Address observed serialization/prefill overhead with equivalence checks; then expand
independent task coverage before committing to pretrained adaptation or scale-up.

The [efficiency probe](artifacts/efficiency_v1/summary.md) records bundled-read replay
equality on 5/5 historical turns. A separate read-only bundling runner is implemented.
Canonical-prefix reuse fails the exact-output gate on 2/9
fixed prompts despite parsed actions matching 9/9; it remains disabled in repair runs.
Do not promote the matched-subset timing ratio into a full-workload speedup claim.

The separate [live comparison](artifacts/repository_v1/comparison.md) now retains 3/3
completion, with 35→24 model turns and 381.55→262.39 seconds of episode time. Both
profiles independently replay. This is one observation per profile, with changed prompts
and generated token counts, no randomization or confidence interval; peak MLX allocation
increases slightly. It is bounded harness-efficiency evidence, not an architectural
speedup or a reason to skip independent task coverage and dense-control checks.

The [external HumanEval subset](artifacts/external_v1/summary.md) is now source-pinned,
licensed and hash-frozen (IDs 0–7, no performance filtering). Six OS-isolation probes
pass, but the requested memory-cap operation is rejected on this Mac even outside the
child sandbox. Reference/candidate execution remains blocked and the model score is
null. Select an approved compatible memory-limited backend before evaluating generated
code; do not remove the guard to produce a result. This old public benchmark also cannot
establish uncontaminated or current-SOTA quality.

## What the completed experiments establish

The 58K-parameter byte model can learn narrow arithmetic templates. The role selector
can identify two explicit prompt classes. Neither establishes general code understanding.
Instruction diversification at fixed compute reduced performance and did not repair
unseen-wording failures. Do not interpret this as evidence for or against grouping at 30B.

The synthetic agent has a fixed read/edit/test sequence and cannot plan an unknown tool
sequence, recover from failure, or explore a repository. The current group names denote
permissions, not demonstrated capabilities. Exact export equality proves preservation of
the masked computation; it does not prove preservation of the full parent's capabilities.

## Candidate selection

Use a pretrained instruction model with published coding evidence and accessible weights.
Check its exact architecture, tokenizer/chat template, license, release, and checkpoint
revision before adapting it. A recent release or small parameter count is insufficient.

Recent references:

* [Qwen3-Coder-Next](https://arxiv.org/abs/2603.00729) provides a coding/agentic training
  reference, but its 80B total parameter count makes it unsuitable as a casual Mac download.
* [Laguna XS.2](https://arxiv.org/abs/2605.27605) is relevant to the intended 30B sparse
  scale, not a small local baseline.
* [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) is a recent small general
  model. Its existence is not evidence that it qualifies for our coding benchmark.

The initial local preflight found approximately 3.8GiB free disk. The user subsequently
freed space, confirmed 8GiB unified memory, and authorized the M2 run. MLX 0.32.3,
MLX-LM 0.32.0 and Transformers 5.19.0 were installed in `.eraenv`; the pinned quantized
snapshot totals 3.06GB and sits in the ignored workspace model cache. Inference ran on
Metal. Quantized inference fit is not evidence that expert duplication or training fits.

## Data and evaluation gate

1. Develop training and validation tasks separately from locked final evaluation.
   Track source/license, repository or generator family, deduplication and content hashes.
2. Include executable algorithmic tests beyond arithmetic templates; cover branches,
   collections, edge cases and several implementation patterns. Require the reference
   solution to pass and deliberately wrong solutions to fail.
3. Include genuine repair environments with observation-dependent actions, failing test
   feedback and bounded retries. The new native-tool single-file harness implements
   these mechanics, records raw actions and independently replays execution. Its reused
   development tasks do not replace repository-level evaluation or independent splits.
   A fixed three-action script is only a harness smoke test.
4. Separate task identity, instruction wording and repository provenance in the split.
   Validate tokenizer/chat formatting, response extraction and hidden-test isolation.
5. Run the untouched pretrained candidate on validation first. Record raw generations,
   failures by category, truncation/EOS, latency, memory and calls per completed task.
   Do not train around a formatting or evaluator defect.

A numeric qualification threshold must follow the target capability and baseline
repeatability, and be frozen before final testing. Do not invent a convenient pass rate
after seeing results. If candidates fail, correct the diagnosed problem or choose a
stronger candidate; another routing experiment is not the remedy.

## Experiment order after qualification

| Gate | Work | What qualifies progression |
|---|---|---|
| 1 | Untouched pretrained validation | Demonstrated useful coding and repair behavior; inspected failures |
| 2 | Dense continuation control | Stable optimization, accurate target masks, measured validation change |
| 3 | Exact MoE conversion | Preserved logits within dtype tolerance on representative inputs |
| 4 | Ordinary MoE continuation | Comparison against a tuned dense control with token and compute accounting |
| 5 | Grouped routing and export | Measured task benefit or export retention, including mixed tasks |
| 6 | Larger pilot | Repeatable benefit plus feasible memory/dispatch/communication costs |

Advance one gate at a time. Change one variable per ablation. Publish negative results.
No method can guarantee improvement; a good experiment should still resolve a specific
question if the proposed improvement fails. The immediate objective is baseline competence,
not a novel router or a 30B training launch.
