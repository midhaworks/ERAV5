# ERA V5 Model — Data Mixture & Curriculum

**Status:** proposal, not a result. Every number below is a hypothesis until the
1B and 3B proxy runs in [§9](#9-proxy-experiments) pass their gates.

This specification carries forward the S3 target: a 40B dense, coding-first,
agentic, India-first model trained on **20T token exposures**
(2,00,00,00,00,00,000 tokens) with a 128K context ceiling. The 20T total is also
a hypothesis; if the compute envelope changes, the percentages stay fixed and
the token counts rescale.

The companion [`V5 Mixture & Curriculum`](index.html) makes the hypothesis
interactive: moving one lane renormalises the others, displays benchmark impact,
and compares required exposure with published supply evidence.

## 1. Decisions in one page

- Eight mutually exclusive capability lanes sum to 100%; a token is charged to
  exactly one lane even when it could fit several.
- General web is the largest single lane at 26% because it has the deepest real
  supply, but Code + Agentic jointly receive 36%; with Reasoning + Long-context,
  **58% of the budget directly supports a Codex-style operating model**.
- Indic receives 12% and is the differentiator, with a protected 8% floor.
- A protected **61% always-on floor** prevents the online selector from starving
  any lane. Only the remaining 39% is adaptive.
- The final **10% / 2T tokens** is an inaccessible anneal reserve until cooldown.
- Agentic receives 12%, with a protected 8% floor; 60% of that lane is
  repository-level coding agents rather than generic function-calling chat.
- Full-scale approval is blocked if accepted unique data cannot meet the stated
  effective-epoch caps. A percentage is not permission to fabricate supply.

## 2. Accounting rules

1. `token budget` means tokens emitted by the frozen V5 tokenizer, not bytes,
   rows, words, or a publisher's tokenizer.
2. `unique supply` is measured only **after** license filtering, exact and near
   deduplication across all sources, benchmark decontamination, and tokenisation.
3. Published sizes below are discovery evidence, not accepted V5 inventory.
   Overlapping corpora such as Samanantar inside BPCC are never added twice.
4. Capability tags use this precedence so totals cannot overlap:
   `safety → agentic → long-context → India-domain → Indic → reasoning → code → general`.
5. Exposure beyond unique supply is reported as an effective epoch. No tier may
   exceed its cap merely to hit a headline share.

The machine-readable values are in [`mixture-plan.json`](mixture-plan.json) and
their sums are checked by [`validate_plan.py`](validate_plan.py).

## 3. Main pretraining mixture

The default composer preset governs the first **18T tokens**. The final 2T is
held outside this selector as the annealing reserve in §6.

| Capability lane | Share | Token exposure | Protected floor | Inventory assigned to it | Benchmark it is meant to move |
|---|---:|---:|---:|---|---|
| General web | **26%** | 4.68T | 12% | FineWeb/FineWeb-Edu, Dolma, Wikipedia, open books and professional documentation | MMLU-Pro, ARC-C, HellaSwag, factuality held-out set |
| Code | **24%** | 4.32T | 18% | The Stack v2 / Software Heritage, permissive repositories, tests, issues, PRs, package and API documentation | LiveCodeBench, HumanEval+, MBPP+, MultiPL-E, SWE-bench Verified |
| Agentic | **12%** | 2.16T | **8%** | Issue-to-patch/test trajectories, ToolACE, public API sandboxes, browser/retrieval traces and terminal workflows | BFCL, τ-bench/τ², executable tool success, recovery rate, SWE-bench Verified |
| Reasoning, mathematics & science | **14%** | 2.52T | 8% | FineWeb-Edu, OpenWebMath, scientific papers/textbooks, Proof-Pile-style text, OpenMathInstruct-2, verifiable problems and the cleaned S4 reasoning summaries | GSM8K, MATH-500, GPQA, BBH, reasoning calibration |
| Indic | **12%** | 2.16T | **8%** | Sangraha, IndicCorpV2, BPCC, Bhashini/BhashaDaan streams, licensed literature/news; tiered in [§5](#5-indic-is-four-ledgers-not-one) | IndicXTREME, IndicGenBench, IN22-Gen/Conv, FLORES, worst-language generation |
| Long-context | **8%** | 1.44T | 4% | Complete repositories, books, papers, judgments, acts and multi-document packs; ProLong-style 64K packing | RULER at 32K/64K/128K, LongBench, repo and judgment QA |
| India-domain | **3%** | 0.54T | 2% | India Code, Gazette, data.gov.in, PIB/MyGov, RBI/SEBI/TRAI, court judgments, NCERT/NPTEL/SWAYAM and state portals | IndQA, MILU, dated jurisdiction QA, UPI/GST/ONDC tasks |
| Safety & grounding | **1%** | 0.18T | 1% | Secure-code data, multilingual red-team cases, policy-grounded refusal/correction and finance/legal/medical boundaries | Indic red-team suite, XSTest/HarmBench-style tests, over-refusal rate |
| **Total** | **100%** | **18.00T** | **61%** | 39% remains selector-adjustable | Weighted proxy score in §9 |

Why these shares:

- **General web stays largest by two points** because it is the only lane with
  enough high-quality supply to provide broad world knowledge below one pass.
- **Code + Agentic is the primary capability block.** Its 36% direct share grows
  to 58% when the supporting Reasoning and Long-context lanes are included.
- **Reasoning is separate from general text.** Educational prose is useful, but
  verifiable solutions and proofs need their own sampling and metrics.
- **Long-context is a construction lane, not a duplicate copy.** A full repository
  assigned to this lane is removed from Code's count; the same applies to books,
  papers and judgments.
- **Agentic is a major, supply-constrained lane.** ToolACE publishes only 11,300
  rows (37.2 MB), so the 2.16T main-phase target is a generation and replay obligation, not
  permission to repeat a small conversation set.
- **Indic receives 12%** because native Indic capability is the reason V5 should
  exist. Its 2.16T main-phase target is aggressive and remains conditional on the explicit
  tier gaps below.

Because the anneal uses a different distribution, the realised end-of-run mix
is not identical to the default preset:

| Lane | Main 18T | Anneal 2T | Final exposure | Realised share of 20T |
|---|---:|---:|---:|---:|
| General web | 4.68T | 0.16T | 4.84T | 24.2% |
| Code | 4.32T | 0.44T | 4.76T | 23.8% |
| Agentic | 2.16T | 0.40T | 2.56T | 12.8% |
| Reasoning, mathematics & science | 2.52T | 0.40T | 2.92T | 14.6% |
| Indic | 2.16T | 0.34T | 2.50T | 12.5% |
| Long-context | 1.44T | 0.20T | 1.64T | 8.2% |
| India-domain | 0.54T | 0.04T | 0.58T | 2.9% |
| Safety & grounding | 0.18T | 0.02T | 0.20T | 1.0% |
| **Total** | **18.00T** | **2.00T** | **20.00T** | **100%** |

## 4. Supply reality

| Lane | Published discovery-scale evidence | Demand | Honest consequence |
|---|---|---:|---|
| General | FineWeb reports more than 18.5T tokens; strict FineWeb-Edu reports 1.3T and its relaxed tier 5.4T | 4.68T | Supply is sufficient below one pass; quality, rights and decontamination are the constraint |
| Code | The Stack v2 deduplicated files are reported as 67.53 TB; StarCoder2 used 3.3–4.3T code tokens | 4.32T | Near one prior-training-scale pass before our stricter licence/dedup losses; add issues, tests, commits and Indian APIs |
| Agentic | ToolACE: 11,300 rows; S4: 57,528 cleaned reasoning-summary rows, only ≈1,80,38,602 estimated tokens and not execution-verified traces | 2.16T | **Most starved.** Build grounded repo histories and generate replayable trajectories; real, replayed and generated counts remain separate |
| Reasoning | FineWeb-Edu has 1.3T strict tokens; OpenWebMath has 14.7B; OpenMathInstruct-2 has 14M generated pairs | 2.52T | Fill mainly with strict + relaxed educational text; cap generated problem families and decontaminate by template and semantic similarity |
| Indic | Sangraha publishes 251.321B tokens across all components; IndicCorpV2 publishes 20.9B and may overlap; Bhashini publishes record/hour counts across several modalities, not comparable token totals | 2.16T | About 8.6× Sangraha's headline total even before dedup; Bhashini assets must be classified by modality and tier before tokenisation |
| Long-context | ProLong's public 64K book split is 6.4B tokens; large full-document supply must be reconstructed from other inventory | 1.44T | Pack source documents once under a long-context tag; do not count their tokens again in the source lane |
| India-domain | S3 names authoritative sources, but no tokenised, cross-deduplicated inventory exists yet | 0.54T | **Starved.** Full-scale approval requires a manifest and at least 0.135T unique accepted tokens at a 4-epoch cap |
| Safety | Public inventory is fragmented and small | 0.18T | Use diverse grounded transformations, not verbatim replay; keep refusal and safe-completion balance measurable |

Published evidence used here: [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup),
[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb),
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
[OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math),
[OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2),
[Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha),
[IndicCorpV2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2),
[ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE), and
[ProLong-64K](https://huggingface.co/datasets/princeton-nlp/prolong-data-64K).

### Bhashini is a multimodal repository, not one translated corpus

The Government of India's published Bhashini inventory reports the following
discovery-scale assets. They are not added to the token ledger until each
dataset's licence, provenance, language balance, quality and overlap are checked.

| Bhashini asset | Published scale | V5 classification |
|---|---:|---|
| Parallel sentences | 24,60,00,000 pairs | **Translated Indic** after pair-quality filtering and cross-source deduplication |
| Monolingual text | 37,00,000 entries | **Verified or unverified Indic**, depending on authorship and document provenance |
| ASR speech | 14,000 hours | Audio is outside text-only pretraining; accepted transcripts are native text with speech provenance |
| TTS speech | 476 hours | Audio is outside text-only pretraining; accepted prompts/transcripts may enter native text |
| OCR images | 25,00,000 samples | Only corrected extracted text is eligible; neither images nor raw OCR output count as text tokens |
| Transliteration | 2,05,60,000 entries | **Synthetic/romanised Indic**, separate from native-script text |

Therefore, the answer is **partly**: Bhashini contains a very large parallel
translation collection, but “Bhashini” cannot be assigned wholesale to the
translated tier. The [official parliamentary inventory](https://sansad.in/getFile/loksabhaquestions/annex/186/AU486_qctfVt.pdf?source=pqals)
reports all six asset classes, while the [BhashaDaan definitions](https://bhashini.gov.in/bhashadaan/en/terms-and-conditions)
separately describe voice recordings, transcriptions, translations and labelled
images. Record counts and hours must first be converted into accepted V5-token
counts; parallel pairs may also overlap BPCC/Samanantar-family sources.

### Agentic is an execution ledger, not chat with tool names

The 2.16T main-phase agentic exposure is divided by operating capability:

| Agentic subslot | Share of agentic | Main-phase exposure | Required behaviour |
|---|---:|---:|---|
| Repository coding agents | **60%** | 1.296T | Inspect a repository, plan, edit, run tests, diagnose failure and verify the patch |
| Tool and API workflows | 15% | 0.324T | Select a tool, form valid arguments and ground the answer in its observation |
| Browser and retrieval | 10% | 0.216T | Search, open, compare and cite retrieved evidence across steps |
| Terminal and environment | 10% | 0.216T | Operate stateful shells, builds and package environments safely |
| Planning and recovery | 5% | 0.108T | Maintain checkpoints, notice a failed action, revise the plan and continue |

Provenance is also locked: **35% real or environment-replayed**, **55% generated
and execution-verified**, and at most **10% simulated and human-audited**. Every
accepted trajectory stores the goal, current plan/checkpoint, action and tool
arguments, the real observation or error, the revised plan after failure, and
final verification. A model-written “tool result” that was never produced by an
environment is rejected, not counted as synthetic agentic data.

## 5. Indic is four ledgers, not one

The 2.16T main-pretraining Indic target is split before language sampling:

| Tier | Share of Indic | Share of total run | Exposure | Maximum effective epochs | What qualifies |
|---|---:|---:|---:|---:|---|
| **Verified** | **35%** | 4.2% | 0.756T | 4× | Human-authored trusted domains, reviewed OCR/transcripts and sources with document provenance |
| **Unverified** | **15%** | 1.8% | 0.324T | 2× | Filtered multilingual/web corpora that pass language, quality and dedup gates but lack document-level human verification |
| **Translated** | **30%** | 3.6% | 0.648T | 2× | Native-script translation from licensed source text, including labelled Sangraha Wiki translation and accepted BPCC target sides |
| **Synthetic** | **20%** | 2.4% | 0.432T | 2× | Romanised/transliterated text and model-generated, source-grounded tasks, explanations and code-mix |
| **Indic total** | **100%** | **12%** | **2.160T** | — | The four counters are never merged in reporting |

The [Sangraha card](https://huggingface.co/datasets/ai4bharat/sangraha/blob/main/README.md)
publishes 64.306B verified tokens, including 12.760B English, leaving about
**51.546B Indic verified**; 24.308B unverified; and 162.708B in a combined
translated/romanised synthetic component. It does **not** publish the latter as
the two separate ledgers required here, so native-script translations and
Latin-script transliterations must be retokenised and counted separately.

The anneal adds 0.34T Indic tokens at verified/translated/synthetic = 50/30/20
and admits no unverified data. Therefore, the final 20T run realises 2.50T Indic
exposure, and its feasibility arithmetic is intentionally uncomfortable:

| Tier | Full-run exposure | Unique supply required at cap | Publicly quantified starting point | Minimum visible gap before new sources |
|---|---:|---:|---:|---:|
| Verified | 926B | 231.5B | 51.546B Sangraha Indic verified | **≥179.954B** |
| Unverified | 324B | 162B | 24.308B Sangraha unverified | **≥137.692B** |
| Translated | 750B | 375B | Not separately published after V5 tokenisation/dedup | Unknown; must measure |
| Synthetic | 500B | 250B | Not separately published after V5 tokenisation/dedup | Unknown; must measure/generate |

[BPCC](https://huggingface.co/datasets/ai4bharat/BPCC) contributes roughly 230M
parallel pairs, including Samanantar-family data, but pairs are not tokens and
are not added until cross-source deduplication. If any tier misses its unique
supply requirement, the realised 12.5% Indic hypothesis fails; the selector may not
quietly compensate by exceeding the epoch cap.

Within each tier, 80% goes to the 22 scheduled-language pool with temperature
sampling (`alpha = 0.5`), 10% to S3's regional/P1 languages, and 10% to
romanised/code-mixed usage. Every scheduled language has a 1% floor of the
Indic lane; the remaining 78% is temperature sampled. Macro and worst-language
scores, not Hindi-weighted averages, decide success.

## 6. Protected selector and anneal reserve

### Online selector

For the first 18T tokens, every 10B-token window is built in two steps:

1. Allocate the fixed lane floors in §3 (61% of the window).
2. Allocate the remaining 39% from smoothed validation-loss improvement per
   training FLOP, subject to a maximum movement of 2 percentage points per lane
   per window and the effective-epoch caps.

The selector cannot touch benchmark hold-outs, cannot use raw loss across
different tokenisers as a quality score, and must reconcile cumulative lane
shares to within ±0.25 percentage points of the main-pretraining targets.

### Anneal reserve

The last **2T tokens (10%)** are reserved at job start and are invisible to the
online selector. Therefore its locked high-quality mixture is not constrained
by the main selector's 61% floor; it retains explicit 8% Indic, 8% agentic and
1% safety guardrails if edited as a counterfactual in the composer.

| Lane | Anneal share | Anneal tokens |
|---|---:|---:|
| General | 8% | 160B |
| Code | 22% | 440B |
| Agentic | 20% | 400B |
| Reasoning | 20% | 400B |
| Indic | 17% | 340B |
| Long-context | 10% | 200B |
| India-domain | 2% | 40B |
| Safety & grounding | 1% | 20B |

Anneal admits only the top quality decile per lane, executable/verifiable
solutions where applicable, and no unverified Indic. The anneal Indic 340B is
50% verified, 30% translated and 20% synthetic. Learning rate decays from 10%
of peak to zero over this reserve; data order is shuffled within 20B-token
blocks so the cooldown is not a benchmark-shaped fine-tune.

## 7. Curriculum

Difficulty is assigned by deterministic signals first (test pass, dependency
count, document count, proof depth, tool count) and calibrated on a 1,000-item
human-reviewed sample per lane.

| Band | Share | Operational definition | Concrete example |
|---|---:|---|---|
| D0 · foundation | 20% | One concept or one local operation; no hidden dependency | Explain GST input tax credit from one dated government paragraph; implement `sum_even(xs)` with tests |
| D1 · applied | 35% | Two–three constraints or a short derivation | Fix a typed API call after reading its schema; solve a two-step percentage problem in Marathi |
| D2 · compositional | 30% | Four–eight reasoning steps, multiple files/docs, recovery allowed | Repair a failing UPI client across three files and tests; reconcile two RBI circular sections |
| D3 · expert/adversarial | 15% | Long dependency chain, ambiguity, distractors or conflicting evidence | Resolve a SWE-bench-style issue in a full repository; answer a jurisdiction question across an Act, amendment and later judgment |

Schedule: tokens 0–20% use D0/D1 at 80%; 20–60% linearly approach the target
table; 60–90% use the target table; the anneal uses D0/D1/D2/D3 =
10/25/40/25 but only from verified high-quality sources.

Reasoning length is the **visible solution, rationale or tool trajectory**, not
private hidden chain-of-thought:

| Band | Share | Visible reasoning tokens | Concrete example |
|---|---:|---:|---|
| R0 | 35% | 0–128 | Direct factual answer with a source line or a one-line code edit |
| R1 | 35% | 129–512 | Short algebra derivation or one tool call plus grounded response |
| R2 | 20% | 513–2,048 | Multi-file debugging explanation with test evidence |
| R3 | 8% | 2,049–8,192 | Multi-document legal comparison or a recoverable tool workflow |
| R4 | 2% | 8,193–32,768 | Repository-scale migration plan with executed checkpoints |

R3+R4 is capped at 10%: long rationales are expensive, often padded and easy to
distil incorrectly. Each generated R2+ item needs a final-answer verifier; code
needs execution, math needs symbolic/numeric checking, and agent traces need
environment replay.

Context grows separately: 8K through 40% of tokens, 32K through 70%, 64K
through 90%, and 128K only in the anneal. RULER must pass at each length before
the next transition; a claimed context window is not an effective one.

## 8. Data gates and the next cleaning tranche

No proxy starts until all of these hold for its sampled data:

- 100% source manifest coverage: source URL/snapshot, licence decision,
  language/script, lane, tier, tokenizer count and SHA-256;
- zero known evaluation items after exact and semantic decontamination;
- 99.9% schema-valid records; PII/secrets and malware scans completed;
- exact + MinHash/SimHash dedup within and across lanes;
- enough accepted unique tokens to respect each lane/tier epoch cap;
- a 1,000-record stratified manual audit with ≥98% lane correctness and ≥95%
  quality acceptance (Wilson lower bound reported, not only point estimate).

The next **100B accepted-token cleaning tranche** follows the model's actual
priority and targets the visibly starved lanes: **35B executable agentic
trajectories**, **35B verified native Indic**, 15B long-document eligible, 10B
India-domain and 5B multilingual safety/grounding. The
S4 corpus is useful as a cleaning-pipeline proof, but ≈18M estimated output
tokens is only 0.018% of this tranche and cannot be presented as material V5
supply.

Full-scale data review remains blocked until the Indic gaps in §5 are closed,
India-domain has at least 135B unique accepted tokens, and accepted agentic
supply can realise the 35/55/10 provenance split without violating source-family
or effective-epoch caps.

## 9. Proxy experiments

### 1B screening

- **Model:** identical 1B dense architecture and frozen V5 tokenizer.
- **Budget:** 30B tokens per arm; same optimiser, steps, batch tokens, context
  schedule, checkpoints and evaluation harness.
- **Arms:** A = S3 exactly mapped as Code 30 / General 22 / Reasoning 12 /
  Indic 18 / India 10 / Agentic 8 / Long 0 / Safety 0; B = this proposal;
  C = floor-bound supply-conservative ablation (Code 24 / General 31 /
  Reasoning 17 / Indic 8 / India 3 / Agentic 8 / Long 8 / Safety 1); D =
  proposal without difficulty curriculum or anneal.
- **Process:** one seed for all four arms; promote the top two, then run two more
  seeds for those arms.

The pre-registered score is:

`0.25 Code + 0.15 Agentic + 0.15 Reasoning + 0.15 Indic + 0.10 Long + 0.10 India + 0.05 General + 0.05 Safety`

Each component is the macro-average of z-normalised metrics against Arm A.
Arm B is confirmed at 1B only if its mean composite improves by **≥1.5 points**,
Code/Indic/Agentic each improve, and no lane regresses by more than **1.0 point**.
Indic additionally requires ≥2 points on the 22-language macro and no scheduled
language losing >1 point. Report mean, seed spread and paired bootstrap 95% CIs.

### 3B confirmation

Train Arm A and the promoted mixture at 3B parameters for 90B tokens, two seeds
each. Confirmation requires:

- composite improvement ≥1.0 point with paired-bootstrap 95% CI above zero;
- LiveCodeBench and BFCL non-inferior within 0.5 point;
- Indic macro and worst-language score both improve;
- RULER improves ≥2 points at 32K with no >0.5-point 8K regression;
- validation loss per lane shows no protected lane diverging.

Failure means revise the responsible share and rerun a proxy; it does **not**
mean rationalise the number at 40B. No 1B or 3B result has been run yet, so this
README deliberately contains acceptance criteria rather than invented scores.

## 10. Review checklist

- [x] Every capability lane has a share, token budget, floor, inventory and benchmark.
- [x] Agentic, reasoning and long-context are explicit lanes.
- [x] Indic is split into verified, unverified, translated and synthetic ledgers.
- [x] Published supply, overlap risk, repetition caps and gaps are visible.
- [x] A 61% always-on floor and 10% anneal reserve are fixed.
- [x] Difficulty and reasoning-length bands include concrete examples.
- [x] 1B and 3B experiments have falsifiable metrics and rejection rules.
- [ ] Proxy runs completed and results linked.
- [ ] Full-scale data gates passed.

Run the arithmetic check:

```bash
python3 validate_plan.py
```
