# Research track: a joint sparse model with coding and agentic exports

**Current decision:** qualify a pretrained coding baseline before more MoE training.
The [baseline gate](BASELINE_GATE.md) records completed diagnostics, execution constraints,
data/evaluation requirements and the order of experiments. The pretrained candidate has
passed bounded local repairs, not broad coding/agentic qualification; no 30B training recipe is established.

Status: conversion/export invariants and a three-seed synthetic transformer pilot
implemented. Three authored multi-file repairs now pass; real-world repository coding,
broad multi-step agent performance and 30B scaling remain unmeasured.
See the [generated pilot report](artifacts/summary.md).

## Executable synthetic pilot

### External evaluation preparation (newest step; blocked preflight)

The [generated external evidence](artifacts/external_v1/summary.md) records an externally
authored evaluation source: [official HumanEval](https://github.com/openai/human-eval),
MIT-licensed, revision `6d43fb980f9fee3c892a914eda09951f772ad10d`. The 44,877-byte source
archive contains 164 tasks; the deterministic subset is IDs 0–7, without model-based
filtering or task substitution. Original tests and canonical solutions are retained;
prompt construction must never include those evaluation-side fields. The loader
rejects `for_training=True`, but this local API is not a global training-system firewall.

**Problem and solution:** our own examples cannot establish externally authored task
coverage. Freeze a third-party subset and its upstream byte hashes before model inference,
and preserve unsupported/failed cases. This does not make the tasks semantically disjoint
from previous programming primitives or unseen during pretraining. HumanEval dates to
2021 and is not a latest-SOTA or uncontaminated benchmark. No full-HumanEval claim applies.

| Preflight component | Measured result |
|---|---|
| Source/license retention and byte/content hashes | PASS |
| First eight numeric IDs frozen before inference | PASS |
| Fake outside-file content read | Blocked |
| Fake outside/workspace writes | Blocked |
| Local network connection | Blocked |
| Child process fork | Blocked |
| Fake parent environment variable | Not inherited |
| Required data-segment memory cap | FAIL on this host |
| Reference execution | NOT RUN |
| Model inference and score | NOT RUN; no score |

The fail-closed executor uses macOS `sandbox-exec`, a sanitized environment, and a
default-deny profile with trusted interpreter/library reads. Root-directory lookup was
necessary for Python startup; it does not allow descendant-file content reads. Generated
Python also faces AST/import restrictions. Trusted isolation probes deliberately bypass
AST gates to test the OS policy itself. These probes are not a comprehensive security audit.

The 64MiB data-cap attempt fails with `current limit exceeds maximum limit`, including
in a trusted Python process outside the child sandbox; earlier 512MiB attempts also fail.
We did **not** disable that guard or run candidates anyway. `preflight.json` records the
failed memory gate and `ready=false`; `evidence.json` has a null model score. Historical
startup diagnostics are retained rather than disguised as model failures. No Docker
installation or external-server coordination has been performed.

```bash
.eraenv/bin/python -u S14/research/external_data.py
.eraenv/bin/python -u S14/research/external_sandbox.py
.eraenv/bin/python S14/research/external_report.py
```

The preparation command reuses pinned cached source bytes and refuses changed frozen
files. On this Mac, the preflight command records the blocker rather than a successful
evaluation. The next requirement is an approved compatible memory-limited backend, such
as a container/Linux executor; generation can stay on the M2 while tests execute there.
Do not advance pretrained adaptation using an external score that has not been measured.

### Live bundled-read comparison (completed harness step)

The [generated comparison](artifacts/repository_v1/comparison.md) and
[machine-readable metrics](artifacts/repository_v1/comparison.json) compare the old
serialized run with a separate complete bundled-read run. Both pass **3/3 hidden-graded
repairs**, and both independently replay all raw actions, files and tests. The model,
packages, task manifest, 16-turn/512-token limits and greedy decoding are unchanged;
the allowed read-call policy and its system instruction differ. Prefix caching is off.

| Quantity | Serialized | Bundled reads |
|---|---:|---:|
| Hidden-graded completed repositories | 3/3 | 3/3 |
| Model turns | 35 | 24 |
| Accepted tool calls | 30 | 32 |
| Protocol-invalid turns | 5 | 1 |
| Generation time incl. full-history processing | 378.98s | 259.43s |
| Episode time incl. tools | 381.55s | 262.39s |
| Generated tokens/s | 4.34 | 5.49 |
| Peak MLX allocation | 3.55GiB | 3.63GiB |

**Problem and solution:** requesting several files required separate model turns and
previously caused rejected responses. Now up to eight read-only calls may share a turn;
all calls are validated before execution. Each task reads its three source modules and
visible tests in one turn. Reads themselves remain sequential, and every mutation/test/
finish stays single. `max_calls=16` limits model turns, not the number of individual
reads inside bundles. Accepted tool-call count increases because the model also reads
visible-test files; fewer model turns does not mean fewer filesystem operations.

Shipping drops from 13 to 6 turns, settlement from 11 to 7, and pagination stays at 11
because it needs several corrections. The model attempts bundled edits once; those are
rejected and remain counted. One complete run per profile shows 31.2% lower episode
time (1.45x ratio), **not** a repeatable or causal kernel-speedup claim. Generated tokens
and subsequent prompts differ; execution conditions are not randomized and there is no
confidence interval. Peak allocation increases slightly. Model loading is excluded.

```bash
.eraenv/bin/python -u S14/research/repository_benchmark.py --bundled-reads
.eraenv/bin/python S14/research/repository_report.py --bundled-reads
.eraenv/bin/python S14/research/repository_comparison.py
```

The completed runner refuses overwrites; the report/comparison commands can regenerate
the audits and summaries from saved traces. Neither run was selectively reset. Results
live under `repository_v1/bundled_run/`, while the serialized files stay intact. This is
still three reused authored miniature repositories, not an independent production
benchmark, an MoE-training result, or evidence that a 30B sparse model will benefit.

### Harness efficiency and preservation gates (fixed-history probe)

The [paired efficiency probe](artifacts/efficiency_v1/summary.md) tests nine fixed
historical prompts, selected before inference: the initial, a mid-history edit, and
final turn from each miniature repository. Fresh and cached arms use the same prompt
tokens, checkpoint and greedy decoding; arm order alternates, with a shared warmup.
Timing includes cache preparation/copying. This is not another quality benchmark.

| Change | Measured check | Result | Decision |
|---|---|---|---|
| Bundled read-only calls | Reexecute all five historical bundles; compare individual read responses and files | 5/5 match, no mutation | Live profile measured separately above |
| Canonical-history prefix reuse | Exact output tokens, text and stop reason on nine fixed prompts | 7/9 match | Gate FAIL; not enabled in repair runner |
| Cached parsed actions | Native tool calls extracted from both arms | 9/9 match | Diagnostic only; does not replace the failed exact gate |

For example, shipping turn 9 changes explanatory text between arms but emits the same
ceiling-division patch. Settlement turn 7 also differs in output. Preserve these failures
rather than weaken the predeclared gate. Numerical effects from prefill segmentation
are a hypothesis, not an isolated cause. Cache reuse needs further diagnosis before use.

On the seven exact-matching pairs alone, fresh generation totals 56.15s versus 28.62s
cached (1.96x ratio). This is a **postselected diagnostic subset**, not an accepted overall
speedup; the other pairs generate unequal outputs. There is one observation per arm,
no confidence interval, and shared persistent cache allocations. Peak MLX allocation
across the probe is 3.64GiB, not an isolated per-method memory comparison.

```bash
.eraenv/bin/python -u S14/research/efficiency_probe.py
.eraenv/bin/python S14/research/efficiency_report.py
# Separate live read-bundling profile; no prefix-cache reuse:
.eraenv/bin/python -u S14/research/repository_benchmark.py --bundled-reads
.eraenv/bin/python S14/research/repository_report.py --bundled-reads
```

Completed evidence is not overwritten. Bundling accepts at most eight `read_file` calls
and validates the entire bundle before execution; mutations/tests/finish remain single.
Reads execute sequentially within one model turn, not concurrently. The separate live
profile writes `repository_v1/bundled_run/`; its completed live result is shown above. Old serialized
results and their frozen protocol remain intact.

The experimental cache retains only the canonical history before the generation header,
checks exact token-prefix equality, resets on rewrites/shortening, and copies state before
generation. It never trims the hybrid model's recurrent state or blindly reuses decoded
XML after message normalization. Unit tests cover these prefix guards and read-only
atomicity; GPU evidence exposes the remaining exact-output limitation. No MoE training,
larger model download or model-quality improvement is part of this step.

### Multi-file repository repairs (serialized baseline)

The [repository environment](repository_environment.py) replaces the one-file fixture
with real temporary files and imports between local modules. The frozen authored suite
covers shipping quotes, pagination, and settlement totals. Each starts with two bugs in
different modules. The [generated report](artifacts/repository_v1/model_run/summary.md)
records completion, edited modules, failed tests, protocol-invalid turns, throughput and memory.

**Completed result:** 3/3 hidden-graded repairs pass. Each edits both broken modules;
all final files and test results independently replay. There are 35 turns: 30 accepted
calls and five protocol-invalid bundled-read turns. Five visible-test failures occur
before the successful repairs. Throughput is 4.34 generated tokens/s including full
prompt reprocessing, and peak MLX allocation is 3.55GiB. This slower rate is measured,
not hidden or compared as a model-architecture result. There was no training.

The next efficiency work should target the observed overhead: support bundled read-only
calls in a separately declared profile and reuse prefix KV state with exact-prefix
checks. Preserve this run unchanged. Require unchanged tool results/greedy outputs
where claimed; do not promise improved model quality from a harness optimization.

| Repository | Cross-module problem | Intended repair | Hidden checks |
|---|---|---|---|
| Shipping | Rounds units down and adds an unwanted base fee | Ceiling division in weights, remove fee in tariffs | Zero units, multiple shipments, large units, zero rate |
| Pagination | Wrong page origin and exclusive-end slicing | One-based page offsets, retain full slice end | Empty input, last page, beyond end, narrow and oversized pages |
| Settlement | Accepts negative paid amounts and ignores fees | Validate nonnegative amount, subtract each accepted fee | Mixed statuses, zero/negative net, exact status match |

For example, shipping's `api.py` calls `tariffs.charge`, which calls `weights.billable`.
For one unit at rate five the correct charge is five; fixing only the rounding bug
leaves the unwanted fee, charging ten. The model must inspect and repair the dependency
chain rather than output one unrelated function. Public helper and integration checks
exercise the contracts; hidden integration cases are not placed in the repository.

```bash
.eraenv/bin/python S14/research/repository_suite.py
.eraenv/bin/python -u S14/research/repository_benchmark.py
.eraenv/bin/python S14/research/repository_report.py
```

The suite and rules are hash-frozen before inference: all three tasks, at most 16 turns
and 512 generated tokens per turn, one native call per turn, greedy non-thinking decoding.
All three reference repairs pass; independently restoring each of six bugs is detected.
The runner refuses to overwrite a completed result. The report independently parses raw
calls, replays disk edits, reexecutes tests, and compares final file contents and grades.
Protocol-invalid turns consume budget, including responses with multiple otherwise valid
native calls. This serialized profile does not measure parallel-tool capability. `run.log`
is a reconstructed trace, clearly distinguished
from original stdout capture. Reference code and hidden examples never enter prompts.

`list_files`, `read_file`, `edit_file`, `run_tests`, and `finish` operate on an explicit
file allowlist. Tests and README are read-only, path traversal is rejected, and fresh
tests are required after every edit. Generated modules execute in an isolated child
with a CPU/time limit and AST restrictions: basic functions and collections, and imports
of functions from existing local modules. This is **not an OS security sandbox** and is
not intended for arbitrary untrusted code. There is no shell, network, package installation,
git manipulation, large codebase, or realistic dependency ecosystem.

These miniature repositories establish harness mechanics and bounded local repair
behavior, not SWE-bench quality, production readiness, or 30B scalability. Pretraining
contamination is unknown. No dense/MoE training or quality comparison is triggered.

### Frozen qualification families (single-file gate)

The [frozen suite](frozen_suite.py) separates the next qualification measurement from
the six previously inspected development tasks. Before inference, it fixes six new task
families, visible/hidden examples, checkpoint revision, decoding limits and scoring rules
in [qualification artifacts](artifacts/qualification_v1/manifest.json). This is not a
training dataset, and its reference code and hidden tests never enter model prompts.

The [completed model run](artifacts/qualification_v1/model_run/summary.md) scores **5/6**
under the predeclared rules: 27 valid tool calls, one successful failed-test recovery,
8.05 generated tokens/s including complete chat reprocessing, and 3.49GiB peak MLX
allocation. Raw calls, final code, test feedback and final hidden grading independently
replay. The model weights, prompts and budgets were not tuned after seeing these tasks.

**Evaluation limitation:** interval merging uses the phrase “closed integer intervals”.
The tests use the conventional interval-endpoint interpretation (only overlap/equal
endpoints merge), while the model merges adjacency too: `start <= current_end + 1`.
For `[-4,-2]` and `[-1,1]`, that produces `[-4,1]`, failing the frozen expected result.
Discrete-integer-set interpretation makes adjacency more plausible, so this is also a
specification risk, not clean evidence of a model reasoning defect. Keep the original
5/6 score and outputs unchanged; clarify the domain explicitly in a future suite rather
than retroactively fix this task or selectively rerun it. No quality improvement or
production-readiness claim follows from this result.

| Family | Visible checks | Hidden checks | Edge cases |
|---|---:|---:|---|
| Interval merging | 3 | 5 | Touching endpoints, containment, negatives, input mutation |
| Matrix transpose | 3 | 4 | Rectangles, zero columns, one row/column |
| Pair sum | 3 | 5 | Distinct indices, duplicates, lexicographic tie-breaking |
| Overlapping string matches | 3 | 5 | Overlap, final offset, longer pattern, Unicode |
| Record grouping | 3 | 4 | Repeated values, order, empty and Unicode keys |
| Grid neighbors | 3 | 4 | Corners, narrow grids, exclusion of diagonals |

All six references pass; 18 deliberately wrong implementations fail. This tests whether
the evaluator catches known errors, not whether the model solves the tasks. Hash checks
reject altered task/protocol files, the runner requires the frozen limits, and existing
model evidence is never silently overwritten. Exact IDs, instructions and reference/faulty
code are compared against the prior development suite. Family separation is manually
authored, not an automatic proof of semantic independence. Common primitives remain
shared; pretraining contamination is unknown.

```bash
.eraenv/bin/python S14/research/frozen_suite.py
.eraenv/bin/python -u S14/research/tool_feedback.py --frozen
.eraenv/bin/python S14/research/tool_feedback_report.py --directory S14/research/artifacts/qualification_v1/model_run
```

The scoring rule is complete-task success after a fresh visible-test pass and all hidden
checks passing, with at most ten turns and 384 tokens per turn. One full run is retained
without selective reruns or prompt adjustments. No convenient production pass threshold
is selected after seeing results. This is a small local qualification set, not locked
final production evaluation, a third-party benchmark, or repository-level agent testing.
The next gate is a separate repository-level repair environment with unambiguous task
contracts and hidden integration tests, not another router-training experiment.

### Native tool-feedback repair (development baseline)

The bounded [tool-feedback harness](tool_feedback.py) measures whether the unchanged
pretrained checkpoint can choose read/edit/test/finish actions using its **native tool
chat template**, rather than emit a fixed synthetic action sequence. It does not train
a router or convert the checkpoint to MoE. The [generated report](artifacts/tool_feedback/tasks_6/summary.md)
contains per-task completion, failed test calls, malformed calls, throughput and memory.

The completed six-task run passes **5/6** hidden-graded repairs in 26 turns, with no
malformed calls, 6.97 generated tokens/s including prompt reprocessing, and 3.23GiB peak
MLX allocation. Parentheses repair includes a genuine failed-test/edit/retest recovery.
The first-index task passes here, but rotation still fails: `xs[-k:] + xs[:-k]` omits
reducing `k` modulo list length. For `[1,2,3], k=4`, it returns `[1,2,3]`, not `[3,1,2]`.
Visible tests did not expose that case; hidden grading correctly prevents a false pass.
No claim of score improvement over one-shot repair is warranted: both score 5/6, on
reused tasks and different inference budgets, with different failed tasks.

```bash
.eraenv/bin/python -u S14/research/tool_feedback.py --limit-tasks 1
.eraenv/bin/python -u S14/research/tool_feedback.py
.eraenv/bin/python S14/research/tool_feedback_report.py
.eraenv/bin/python -m unittest discover -s S14/research -p 'test_*.py'
```

The cached pinned snapshot is required; there is no additional model download. Completed
results are not silently overwritten. Limits are ten turns, 384 generated tokens per
turn, greedy non-thinking decoding. Every raw generation, parsed call, tool response,
final code and timing is saved. The independent report re-parses raw calls, replays
state changes and reexecutes tests, rather than trusting the reported score.

**Problem and solution:** a one-shot corrected function does not demonstrate using tools.
Here `read_file("solution.py")` reveals the actual faulty implementation, `edit_file`
replaces the single virtual file, and `run_tests` executes visible examples in an isolated
Python child. For example, ordered deduplication starts with `sorted(set(xs))`, which
incorrectly changes `[3,1,3,2,1]` to `[1,2,3]`; feedback can reveal that `[3,1,2]` is required.
The model can retry after failure. An edit invalidates previous passing tests, and
`finish` is refused until the current code has passed. Hidden examples grade the final
code but never enter tool feedback. Hidden failure still counts as task failure.

This reuses the six **development** task families from the one-shot run. Hidden means
withheld during the episode, not independent unseen final evaluation. There is one file,
no repository exploration, no arbitrary shell, and no claim of a secure general-purpose
sandbox. Additional turns and feedback change the inference budget, so raw success
rates are not a matched-compute architecture comparison. Repository-level agentic
evaluation and a locked independent task set remain gates before training experiments.

### Pretrained M2 validation (one-shot baseline)

The [generated M2 report](artifacts/pretrained/summary.md) records the completed 24-prompt
run: 17 passes, roughly 25.3 tokens/s including prompt processing and 2.49GiB peak MLX
allocation. Direct coding and both repair forms score 5/6; reworded coding scores 2/6.
Four failures violate first-index semantics, and three violate the required function name.
Saved generations are independently rechecked in `audit.json`, including input preservation
for rotation. The initial checker failure on valid `set.add` is retained as a diagnostic.
These are real inference results without training; broader agentic capability remains unqualified.

The bounded MLX harness uses [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
through a [community 4-bit conversion](https://huggingface.co/mlx-community/Qwen3.5-4B-4bit).
The verified snapshot totals approximately 3.06 GB. This is a recent general instruction
model, not an already-qualified coding specialist. The checkpoint revision, package
versions, task hashes, raw output, failures, token limits, wall time and peak MLX allocation
are written under `artifacts/pretrained/`. Disk space is checked before download.

```bash
.eraenv/bin/python -m pip install mlx-lm
.eraenv/bin/python -u S14/research/pretrained_baseline.py --preflight
.eraenv/bin/python -u S14/research/pretrained_baseline.py --limit-tasks 1
.eraenv/bin/python -u S14/research/pretrained_baseline.py
.eraenv/bin/python S14/research/pretrained_report.py
```

Use `--revision SHA` to reproduce the recorded checkpoint. The model cache stays under
`research/model_cache/` and is ignored by git. Generation is greedy, non-thinking, capped
at 192 tokens, with the model's chat template. MLX has a 5GiB allocation limit and a 256MiB
cache limit; peak MLX allocation excludes other applications and does not measure total
system memory. Quantized inference is not proof that MoE training fits 8GiB.

Six authored validation tasks cover ordered deduplication, balanced parentheses,
first-match indexing, run encoding, rotation and positive summation. Each is tested as
generation and repair, with two wordings. Reference implementations must pass and faulty
implementations must fail before inference. Outputs run in a restricted, isolated Python
child with a CPU limit and parent timeout. Imports/introspection are excluded; this is
not a general secure sandbox or an unrestricted coding evaluator. Complete code fences
are removed, but explanations or broken code are not silently repaired.

These are development validation tasks, not locked final evaluation. This one-shot run
asks for a corrected function directly; its scores do not measure an agent loop. Completing them establishes
a narrow baseline only. No automatic qualification threshold, fine-tuning or MoE conversion
is triggered by the harness.

### Instruction diversity (newest experiment)

```bash
python3 S14/research/diversity_benchmark.py
```

The [diversity report](artifacts/diversity/summary.md) compares original-only checkpoints
with models trained on three instruction families. Architectures and task pairs remain
fixed. Shared pretraining and continuation each use the existing controls' analytical
FLOP budgets, so longer prompts may yield fewer optimizer steps. Both sides use the same
two learning rates and original validation tasks for selection. The final wording family
is distinct from all three training families and validation wording, and its bytes are
covered by training. Previously evaluated selector stress wording is not used here.

Underlying task diversity is unchanged: paraphrasing adds three prompt forms per solution,
not new algorithmic tasks. Standard and held-out-wording generation/repair success are
reported separately for dense, unrestricted MoE and oracle-grouped MoE, across three
seeds. Raw outputs and both training/search costs remain available in per-seed JSON.
Original checkpoint hashes identify the reused controls. No export or selector quality
improvement should be inferred until those stages are separately re-evaluated.

Observed result: all variants still scored zero on held-out wording. Standard grouped
MoE success fell from 63.9% coding / 38.9% repair to 13.9% / 5.6% after diversification
under the fixed budget. Dense and unrestricted controls also regressed. This is a failed
ablation, not evidence against instruction diversity in general: the increased prompt
variety changes the optimization difficulty and exposure per variant. It provides no
basis for scaling the current model. The next research gate is a competent coding
baseline with measured generalization before interpreting further MoE changes.

### Learned prompt selector

After generating the trajectory checkpoints below, run:

```bash
python3 S14/research/selector_benchmark.py
```

The [generated selector report](artifacts/selectors/summary.md) compares learned and
oracle routing on the same frozen parent checkpoints for seeds 14, 27 and 42. The selector
has normalized byte-frequency features and a 256→32→2 classifier. It trains on training
prompts with role labels, uses validation loss to select its checkpoint, and receives
only observable prompt text during inference. It has no access to test answers or hidden
arithmetic tests. Its choice remains fixed while an action is generated, then is recomputed
from the next observed state. Selection is request/action-level; token routing stays inside
the selected group. This baseline has no "both" group and cannot represent word order.

An unseen-wording stress test is evaluated for both oracle and learned routing. This
separates parent sensitivity to wording from selector errors. Stress templates are withheld
from selector training and validation. They still describe the same synthetic arithmetic
tasks; success does not establish transfer to natural repositories. Raw routing decisions,
generated actions, checkpoint hashes and classifier traces are saved. Selector training
and evaluation time are separate additional costs, not included in the parent's FLOP budget.

Observed across all three seeds: both standard and unseen-wording classification are
48/48 correct, with 100% balanced accuracy. Learned routing preserves standard-template
oracle task success (63.9% coding, 38.9% repair). Both shifted oracle and learned task
success are zero. The stress wording is checked to contain only bytes seen in parent
training, isolating a parent wording-generalization failure rather than unseen-byte
coverage. Richer training and independently held-out wording are prerequisites for
further capability claims; adding a more elaborate router alone cannot fix this result.

### Compute-budgeted read/edit/test pilot

The newer [read/edit/test experiment](artifacts/trajectories/summary.md) adds three
generated actions and an approximate training-compute budget:

```bash
python3 S14/research/trajectory_benchmark.py --seed 14
python3 S14/research/trajectory_benchmark.py --seed 27
python3 S14/research/trajectory_benchmark.py --seed 42
python3 S14/research/trajectory_report.py
```

The environment advances only when a generated tool action is valid. Generated edit
content becomes the file tested by the arithmetic interpreter; incorrect content fails.
There is no automatic correction, retry training or reference-answer substitution.
Prompts provide compact state summaries rather than full accumulated history. This is
a fixed three-action synthetic environment, not a repository or shell agent benchmark.

The split contains 21 training, 3 validation and 12 test operation/constant pairs,
kept disjoint across all roles and trajectory stages. Both dense and MoE receive the
same two-LR search (0.001, 0.002). Validation response NLL selects the LR; final task
success is measured only afterward. All search cost is recorded. No significance claim
is justified by such a small validation/test set.

Continuation training stops before exceeding a shared analytical FLOP budget. The
formula counts padded attention, vocabulary projection, full router scores and selected
FFNs, then estimates backward at twice forward cost. It excludes optimizer, nonlinearities
and dispatch overhead. The MoE executes fewer steps because two full FFNs activate per
token. Actual CPU time is reported separately; this does not establish GPU efficiency.
Independent exports and exact logit-agreement checks are repeated for each selected model.

The earlier token-matched pilot can still be reproduced with:

```bash
python3 S14/research/benchmark.py --seed 14
python3 S14/research/benchmark.py --seed 27
python3 S14/research/benchmark.py --seed 42
python3 S14/research/report.py
```

The 58,080-parameter dense causal transformer has two blocks, width 48, four attention
heads, byte tokens with explicit PAD/EOS, tied input/output weights, and context 160.
It receives 300 initial training steps, followed by 300 continuation steps per variant.
Every continuation variant uses the same sampled batches. Upcycling clones complete FFNs
into four experts per block with top-2 dispatch; initial logits are preserved.

Tasks ask for `solve(x)` implementing addition, subtraction or multiplication by a small
constant. The edit-action role asks for a JSON action repairing a faulty function.
This measures one action, not a complete agent loop. Responses are greedily generated,
parsed as Python AST, and interpreted on five test inputs through a restricted arithmetic
interpreter. Test labels never enter generation. Prompt and padding targets are masked.

Operation/constant pairs are held out across both roles. Every digit appears in training;
this prevents the earlier diagnostic split's unseen-byte issue. The test set is small
(12 tasks per role), synthetic and drawn from the same template family as training.
It cannot establish generalization to real repositories. Results include failed responses.

The pilot compares dense continuation, unrestricted MoE, grouped MoE, and standalone
group exports. Role masks are **oracle labels**, not a learned group selector. Comparisons
are token-matched, not compute-matched: two activated experts cost more than one dense FFN.
The unrestricted and grouped models share router initialization. No hyperparameter sweep
or pruning/recovery training is included yet. Each seed writes tasks with a content hash,
raw generations, NLL, pass counts, training time, and independent checkpoint state dictionaries.
Checkpoint weights are generated locally and gitignored; JSON/Markdown evidence is retained.

The export has 76,996 parameters versus the MoE parent's 114,632: shared components
prevent a 50% total reduction when half the experts are removed. The next decision must
come from compute-matched, tuned controls and richer held-out tasks, not training loss.

## Question and scope

Can training a joint MoE with occasional restricted expert masks improve two smaller
exports while preserving the parent's coding and agentic capabilities?

S14's regression demonstration remains separate. Its synthetic loss reduction is not
evidence for this question. Group names describe routing permissions; they do not prove
expert specialization. The initial implementation uses explicit group masks and a learned
token router. A learned group selector is a later ablation, not already implemented.

## Implemented first step

`grouped_moe.py` converts a trained linear layer into exact expert copies. There is no
target-dependent router initialization. Sparse dispatch evaluates selected rows only.
Both groups can be available to the parent. Export physically removes the other group
and copies retained expert and router weights into an independent model.

Run the invariant checks from the repository root:

```bash
PYTHONPATH=S14/research python3 -m unittest discover -s S14/research -p 'test_*.py' -v
```

Checks cover dense-output preservation, export equivalence after expert weights diverge,
independent exported storage, smaller parameter counts, and gradient isolation. Export
equivalence means matching the **restricted parent**, not matching its unrestricted output.
The Python dispatch is a correctness reference; GPU efficiency requires a grouped kernel.

## Recent research anchors

* [Expert Upcycling, April 2026](https://arxiv.org/abs/2604.19835): expert growth during training.
* [Cluster-Aware Upcycling, April 2026](https://arxiv.org/abs/2604.13508): specialization at initialization.
* [Qwen3-Coder-Next, February 2026](https://arxiv.org/abs/2603.00729): executable coding environments and feedback.
* [Laguna, May 2026](https://arxiv.org/abs/2605.27605): a 33.4B-total, 3B-active coding model reference.
* [Higher-order Expert Pruning, September 2026](https://arxiv.org/abs/2609.18916): cooperative expert selection.
* [Dynamic Expert Pruning, October 2026](https://arxiv.org/abs/2610.02951): task/role-dependent masks.

These are related methods, not evidence that our proposed training objective works.
Do not claim novelty before reading full methods and comparing their objectives.

## Experimental protocol before expensive training

1. Build a small causal transformer and an executable benchmark with generation, repair,
   tool selection, and repeated edit/test trajectories. Use licensed data with source
   hashes. Split by repository and task family before generating trajectories; keep final
   evaluation inaccessible to training and selection.
2. Tune a dense continuation control and an unrestricted upcycled MoE. Account for the
   initial dense training cost separately and in total. Hold data order and tokenizer
   fixed, and report both token-matched and compute-matched comparisons.
3. Compare unrestricted routing with explicit group restrictions. Treat group identity
   supplied by task labels as an oracle baseline; inference must either use an explicit
   user role or a learned selector whose mistakes are measured.
4. Extract equal-budget subsets using random selection and measured expert importance
   as controls. Compare a recent interaction-aware method when a verified implementation
   is available. Evaluate without recovery first; account for any recovery tokens/time.
5. Test occasional export-mask training versus ordinary parent training with identical
   budgets. Tune the mask frequency on validation only. Report parent degradation as well
   as export improvements. Mixed repair tasks must permit coding/agentic cooperation.
6. Repeat shortlisted comparisons with seeds 14, 27, and 42. Publish individual results
   and task-level uncertainty; do not select a winner on one seed or training loss alone.

| Measurement | Required evidence |
|---|---|
| Code quality | Hidden executable tests; pass@1 with fixed decoding |
| Agent behavior | End-to-end task success, tool validity, retries and timeouts |
| Efficiency | Wall time, peak device memory, active/total params, tokens/s |
| Task cost | Model tokens, tool calls and elapsed time per successful task |
| Specialization | Usage by task/layer plus expert-removal effect on success |
| Export quality | Full parent, masked parent, exported checkpoint and recovered export |
| Reproducibility | Config/data hashes, checkpoints, exact commands, raw traces |

Selection uses quality-versus-cost curves. Acceptance margins must be set before final
evaluation after baseline repeatability is measured. Small-model results justify a larger
pilot only; distributed all-to-all, expert load imbalance and long-context memory require
separate hardware measurements.

## Parameter budget

Let S be shared parameters and C/A be coding/agentic expert parameters:
parent = S + C + A; independent exports = S + C_retained and S + A_retained.
With S=4B and C=A=13B, the parent has 30B, but retaining every expert yields two 17B
exports. Two 15B exports each retain 11B specialists and duplicate the 4B shared part.
Pruning, merging or distillation must account for that removed capacity. Output heads,
embeddings, routers and shared experts all count. Active parameters are a separate metric.
