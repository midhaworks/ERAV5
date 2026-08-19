# RKE-Head: a reversible, vocabulary-independent Kronecker output

**Session 7 · selected problem 5:** “How do I make a reverse of this, so the same embedding gives the same Kronecker? Can we remove the final vocabulary head?”

This submission proposes **RKE-Head (Reversible Kronecker Embedding Head)**. It replaces a `d_model × |V|` token classifier with an ordered position × byte-symbol code. The same small byte codebook is used forward for input and transposed for output. There is no vocabulary-sized classifier, and its parameter count does not depend on the number of token strings. The strongest causal model does contain a `d_hidden × d_code` structured output adapter; it is counted explicitly and must not be described as “zero output-head parameters.”

The submission contains analytic, controlled-composition, natural-language and learned-continuation evidence. It does **not** claim to be production ready.

The evidence-driven improvement plan and primary-source literature map are in
[`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md). Its leading next experiment is a
UTF-8-constrained structured Kronecker head that models dependencies between output bytes
without restoring a vocabulary-sized classifier.

## Original paper and the five V2 directions

The starting point is [*Kronecker Embeddings: Byte-Level Structured Token Representations for Parameter-Efficient Language Models*](https://arxiv.org/abs/2605.29459). The assignment proposed five **independent** research problems that could contribute toward a future Kronecker Embeddings V2. They should not be interpreted as five claims solved by this repository.

| Problem | Research question | Direction considered | Status here |
|---|---|---|---|
| 1. Mathematical structure | Can an embedding carry exact numerical structure so operations such as `9+9` and `9×9` map to the representations of `18` and `81`? | Append a protected typed algebra register and test closure outside the training range | Idea only |
| 2. Text, image and audio | Can one Kronecker family represent all three modalities? | Factor content × local coordinate × modality after patch/spectrogram preprocessing | Idea only |
| 3. Dynamic length | Can short strings avoid reserving 32 positions while long strings avoid cropping? | Use ragged blocks with explicit CONT/EOS boundaries | Supporting mechanics implemented; not the selected standalone claim |
| 4. Fourier alternative | Can ordered character waves be added and still reconstruct the word? | Encode position as Fourier phase; full frequencies remain invertible through an inverse DFT | Idea only |
| **5. Reverse Kronecker** | **Can the same structured representation decode back to the same string and remove the vocabulary head?** | **Tie the input byte codebook to structured output, add EOS/CONT and decode under UTF-8 constraints** | **Selected and implemented** |

The four unselected directions are developed as falsifiable proposals in [`OTHER_IDEAS.md`](OTHER_IDEAS.md). The remainder of this README reports only the implementation and evidence for the selected reverse-decoding direction, plus the continuation mechanics it requires.

## Headline result: matched next-token experiment

The strongest experiment compares three output mechanisms on the same controlled compositional language. Every stem, suffix and character occurs in training, but all 250 evaluated whole-token combinations are absent from the training vocabulary:

```text
context: [JOIN, suffix, stem] → next token: stem + suffix
example: [JOIN, "3", "fa"] → "fa3"
```

All arms use the same structured inputs, hidden width, one-layer single-head causal transformer and batch size.

| Output arm | Seen exact | Held-out OOV exact | Held-out NLL per byte/EOS | Total parameters |
|---|---:|---:|---:|---:|
| Vocabulary softmax | 98.0% | 0% | Undefined: target has no class | 75,168 |
| Autoregressive byte fallback | 99.2% | 94.4% | 0.034743 | 21,888 |
| **Parallel RKE-Head** | **100%** | **100%** | **0.000522** | **21,096** |

For RKE, the current deterministic run also measures 10-bin ECE `0.002075`, Brier score `0.000025`, and zero separate vocabulary-classifier parameters. CPU decode throughput is recorded in `artifacts/lm_v2/results.json`; it is intentionally not treated as hardware-independent.

What this establishes:

- RKE performs genuine next-token composition rather than merely copying a complete payload.
- It emits held-out bounded strings that have no vocabulary-softmax output row.
- It predicts all positions and EOS in one parallel model call; fallback needs four autoregressive byte/EOS calls.
- EOS is loss-bearing, post-EOS PAD is masked, and UTF-8 decoding rejects invalid transitions.

The authoritative, generated evidence is in `artifacts/lm_v2/results.json`, with the exact split, predictions, models and hashes beside it.

## Multilingual full-byte result

The trained pathway is also exercised with the actual 258-state codec—PAD, EOS and all 256 byte values—on four UTF-8 scripts. It trains on 128 compositional forms and evaluates 32 unseen whole forms, eight per script. Every held-out stem and suffix occurs in that script's training split; only the complete combination is new.

| Script | Held-out forms | Exact match | NLL per byte/EOS | Invalid UTF-8 after constraint |
|---|---:|---:|---:|---:|
| Hindi / Devanagari | 8 | 100% | 0.000156 | 0% |
| Telugu | 8 | 100% | 0.000149 | 0% |
| Tamil | 8 | 100% | 0.000733 | 0% |
| Arabic | 8 | 100% | 0.021682 | 0% |

Overall multilingual held-out NLL is `0.004449` per byte/EOS, with 100% exact match. Indic stems are consonants combined with dependent vowel marks; Arabic stems are letters combined with harakat. The saved predictions include the Unicode text and its UTF-8 bytes. This proves full-byte compositional decoding across these scripts in the controlled task; it does not prove multilingual semantics or natural-corpus perplexity.

## V2.1 natural multilingual pilot

This section is newest-first. It replaces synthetic copying with natural next-word and cross-block continuation prediction from Wikipedia-derived English, Hindi, Telugu and Sindhi text. Text is NFC-normalized and case-folded. The earlier single-block pilot hash-splits whole paragraphs; the newer cross-block benchmark isolates complete source documents.

| Version | Specific problem | Implemented solution | Current result |
|---|---|---|---|
| V2.1.11 | One topical neighborhood per language could make apparent language gains a subject-matter artifact | 400 revision-pinned documents across 4 languages × 10 topic strata, with balanced document and target splits | Construction and quota gates PASS |
| V2.1.10 | Global sampling let high-volume languages dominate | Equal source-language quotas plus per-language, macro and micro metrics | PASS; content-language validation remains open |
| V2.1.9 | A >24-byte *word* filter selected scripts by UTF-8 width | Shortest 25–96-byte complete-token continuation spans | Script exclusion fixed; balance completed in V2.1.10 |
| V2.1.8 | Decision-sampled model had 7.2% exact continuation and lost to retrieval | Full-sequence causal PyTorch RKE, matched initialization/batches, validation-only calibration | Efficiency PASS; strict quality parity FAIL |
| V2.1.7 | One article/language; no revision or document-isolation proof | Resumable, revision-pinned, hashed 40-document corpus | PASS; later balanced by V2.1.10 |
| V2.1.6 | Relative wins could be confused with deployment readiness | Generated required-gate matrix with no silent aggregation | Candidate NO, honestly gated |
| V2.1.5 | Continuation was an oracle, not learned language modelling | Generated-prefix cross-block RKE plus frozen-block continuation protocol | Learning exists; production quality FAIL |
| V2.1.4 | Fixed 32 slots waste space and truncate long words | Ragged 24-byte CONT/EOS blocks plus neural mask oracle | Mechanics PASS |
| V2.1.3 | Synthetic/single-seed evidence did not establish natural LM quality | Natural four-language next-word task and three-seed causal comparison | Mean parity PASS |
| V2.1.2 | Independent parallel slots miss within-word dependencies | Masked causal and two-pass refinement alternatives | Negative result retained |
| V2.1.1 | Hand-written NumPy could hide implementation defects | Parameter-identical PyTorch forward/backward/Adam parity | CPU parity PASS |

### V2.1.11 Topic-stratified manifested corpus — PASS

**Problem.** Forty documents selected from one `India` link neighborhood supplied multiple documents, but not broad subject coverage. A head could appear stronger or weaker because one Wikipedia edition's country pages use different names, dates or formatting—not because its representation generalizes across language. A document-level split alone does not remove that topic confound.

**Solution.** `acquire_multidoc.py` now builds schema-v2 data from ten declared roots—country, science, history, geography, literature, mathematics, technology, biology, music and sports—in each of English, Hindi, Telugu and Sindhi. It accepts exactly ten qualifying documents per language/topic stratum, for 400 total. Direct namespace-0 links are considered in sorted order; sparse roots use declared, validated fallbacks and bounded deterministic depth-two expansion. Root-exhaustion records make an interrupted crawl resumable. Exact-domain Sindhi music and sports fallbacks are direct-only, preventing broad landing pages from leaking linked biographies or geography into those strata. Every accepted page retains its topic/root, page and revision IDs, revision timestamp, canonical URL, payload hash, byte count and license in `data/multidoc/manifest.json`.

Documents are hash-ordered independently inside every language/topic stratum and split `8 train / 1 validation / 1 test`. Target spans are then quota-sampled inside those same strata: 100/20/12-or-13 train/validation/test targets per topic and language, producing exactly 4,000/800/500 examples while preventing a large topic from backfilling a sparse one. The builder fails closed if any stratum cannot supply its quota after cross-split target de-duplication.

**Example.** English mathematics and Sindhi music each contribute eight source documents to training, one unseen document to validation and one different unseen document to test. Each contributes 100/20/12-or-13 selected targets; neither can dominate the aggregate merely because it has longer articles. A downloaded page cannot silently migrate between topics because its declared root and topic are validated against the manifest schema.

**Proof and result.** Automated acquisition tests cover exact language/topic counts, interrupted resume, cache reuse, payload tampering, duplicate content, malformed provenance and path traversal. The final corpus manifest hash is `3ac6ab20cd47f8993aa923fc2c34f503ebb1d860dc2bfc3d5189226947063b20`. The generated cross-block audit records the full `document_inventory_by_language_topic_split`, `available_by_language_topic_split` and `selected_by_language_topic_split`; the production-readiness gate requires all 400 documents, all ten topics and every topic quota before it passes. Topic balance addresses topical concentration, not source-family dependence or target-language purity; a second licensed source and target-level language checks remain separate steps.

The one-command run rebuilt the causal head comparison on this corpus. RKE now generates
22/500 exact held-out suffixes versus 18/500 for the byte fallback and retains 94.69% of its
median CPU decode throughput with 10.30% fewer total parameters. This is useful evidence,
but **not quality parity**: raw micro NLL is 3.0492 versus 2.6137 (16.66% worse), and
calibrated micro NLL is 1.7411 versus 1.6949 (2.73% worse). The candidate verdict therefore
remains `NO`.

### V2.1.10 Balanced sampling and macro evaluation — PASS

**Problem.** V2.1.9 made every source language eligible, but a single global hash sampler still selected `99/274/94/33` English/Hindi/Telugu/Sindhi test examples. Aggregate scores therefore over-weighted Hindi and under-weighted Sindhi. Merely reporting that every language appeared did not establish a balanced comparison.

**Solution.** Selection now applies deterministic quotas independently inside each source-language and split: 1,000 train, 200 validation and 125 test spans per language. Cross-split target strings are removed before quota selection, and the builder fails rather than silently backfilling from another language when a quota cannot be met. The selected rows are then hash-interleaved without changing counts. Generation and NLL reports expose each language separately, a micro average over all examples or loss-bearing decisions, and a macro average that gives each language equal weight.

**Example.** At the V2.1.10 revision, exact match had equal denominators, so causal RKE's four per-language rates—2.4%, 3.2%, 7.2% and 3.2%—produced both 4.0% macro and 4.0% micro exact. Byte accuracy differed: micro was 21.72%, while macro was 17.69%, because the micro calculation gives more weight to languages whose targets contain more suffix bytes. Both values were retained.

**Proof and result.** That revision generated dataset hash `3b7f8834d26ca89a53048b8aacfd8ffe822d72fa83be34425a29d1dd74d185f5` and passed both `cross_block_language_balance` and `cross_block_macro_micro_reporting`. V2.1.11 preserves those invariants but supersedes the following historical scores with the topic-stratified run above.

| Split | English | Hindi | Telugu | Sindhi | Total |
|---|---:|---:|---:|---:|---:|
| Train | 1,000 | 1,000 | 1,000 | 1,000 | 4,000 |
| Validation | 200 | 200 | 200 | 200 | 800 |
| Test | 125 | 125 | 125 | 125 | 500 |

| Causal arm | Exact micro / macro | Suffix-byte accuracy micro / macro | Raw NLL micro / macro |
|---|---:|---:|---:|
| RKE | 4.00% / 4.00% | 21.72% / 17.69% | 3.1027 / 3.6779 |
| Byte fallback | 4.60% / 4.60% | 22.51% / 18.08% | **2.7040 / 3.1490** |

These are source-language quotas based on the manifested Wikipedia edition. They do not yet prove that every selected span is written predominantly in that language: for example, the Sindhi edition contains some Latin-script passages. Target-level language/script validation remains an explicit data-quality limitation for a later step.

### V2.1.9 Script-neutral cross-block targets — construction PASS, sampler superseded

**Problem.** V2.1.8 required one word to exceed 24 UTF-8 bytes. That is not a script-neutral definition of a difficult target: many Indic characters occupy three UTF-8 bytes while ordinary English characters occupy one. The old held-out pool therefore contained 269 Hindi and 231 Telugu examples, but zero English and Sindhi examples. The model was being tested on a property of UTF-8 storage width as much as language continuation.

**Solution.** `build_continuation_span` now chooses the shortest sequence of complete normalized tokens whose joined representation occupies 25–96 bytes. Tokens are joined with one ASCII space; no byte, Unicode code point or token is cropped. The 24-byte decoder block remains unchanged, so every selected target still requires at least two CONT/EOS blocks. Each record stores its start/end token offsets, token count, byte count, document identity and document-content hash.

**Example.** An English target such as `the model predicts the next token` crosses the 24-byte boundary as a complete-token span. A Telugu target may cross it in one or two longer UTF-8 tokens. Both exercise the same byte-block decoder, but neither language is admitted or rejected merely because of bytes per character.

**Proof and result.** The builder validates NFC normalization, UTF-8 round trips, complete-token spacing and the 25–96-byte range for every selected record. Unit tests include an ASCII span that must cross the boundary without slicing a token, and reject a single token that cannot fit inside the four-block limit.

| Language | Available train | Available validation | Available test | V2.1.9 global-sampler test |
|---|---:|---:|---:|---:|
| English | 34,925 | 9,483 | 1,384 | 99 |
| Hindi | 8,166 | 347 | 3,603 | 274 |
| Telugu | 14,429 | 5,812 | 1,138 | 94 |
| Sindhi | 44,486 | 621 | 529 | 33 |

All four languages gained held-out cross-block coverage, and document/content/target/sample leakage remained zero. The final column records the V2.1.9 imbalance rather than rewriting history; V2.1.10 supersedes that sampler with 125 held-out examples per source language.

### V2.1.8 Causal full-sequence continuation — efficiency PASS, strict quality FAIL

**Problem.** The decision-sampled V2.1.5 model proved that learned continuation was non-zero, but 36/500 exact suffixes (7.2%) was below a fixed 20% production-test threshold and worse than an exact-prefix retrieval control. It also trained RKE and fallback with different seeds/data streams, weakening causal attribution.

**Solution.** `torch_continuation_lm.py` encodes the complete first block with a GRU and trains on each complete remaining byte/CONT/EOS sequence. RKE projects decoder state into a 64-dimensional code space and scores with the transposed input codebook; fallback uses an otherwise matched 259-class matrix. Both arms now share the same initial body hash, batch-stream hash, 192-wide recurrent body, 3,000 updates and decoding constraints. A scalar temperature is selected independently for each arm on validation only; raw NLL is retained, and scaling cannot change exact outputs.

**Example.** Given the frozen first 24 bytes of the held-out Hindi span `नाम से प्रसिद्ध`, the model receives the real block-ending `CONT` state, then must causally emit every remaining UTF-8 byte and `EOS`. It never receives the gold suffix. The retrieval control may only return the most frequent training span with the identical 24-byte prefix.

**Proof and result.** The initial development floor is ≥20% exact and better than retrieval. The stricter aligned gate additionally requires exact ≥99% of fallback, raw and calibrated NLL ≤1.01× fallback, ≥10% active-parameter reduction and throughput ≥90% of fallback. Raw NLL remains primary; calibration cannot hide a training-quality gap.

| Arm | Parameters | Separate vocabulary classifier | Validation exact | Test exact | Raw test NLL | Validation-calibrated test NLL |
|---|---:|---:|---:|---:|---:|---:|
| Causal sequence RKE | **326,080** | **0** | 5.00% | **4.40% (22/500)** | 3.0492 | 1.7411 (`scale=0.4`) |
| Matched byte fallback | 363,520 | 49,728 | 5.00% | 3.60% (18/500) | **2.6137** | **1.6949** (`scale=0.4`) |
| Exact-prefix retrieval | — | — | — | 0% (0/500) | — | — |

RKE beats retrieval and fallback exact in this single run, removes the 259-class output matrix, reduces active parameters by 10.30%, and retains 94.69% of fallback median decode throughput across five alternating measurements. Its 12,288-parameter structured adapter remains included in the 326,080 total. It still fails strict quality: 4.4% exact is below the fixed 20% floor; raw micro/macro NLL are 16.66%/16.82% worse, and calibrated micro/macro NLL are 2.73%/3.43% worse. Exact match and NLL disagree, so the result cannot be presented as a win. Because the current test split was observed during development, confirmatory evidence also requires three seeds and a newly frozen source-held-out split.

### V2.1.7 Versioned multi-document corpus — historical precursor

**Problem.** V2.1.5 originally used one article per language and paragraph-level splits. That was enough for a pilot, but it could not demonstrate multi-document generalization, document isolation, revision reproducibility, or a 500-example cross-block test.

**Solution.** The V2.1.7 implementation introduced a revision-pinned corpus of 40 Wikipedia documents: 10 each for English, Hindi, Telugu and Sindhi. It records the page/revision identity, revision timestamp, canonical URL, license, original/stored character counts, byte count and SHA-256 of every UTF-8 payload. It also hashes the canonical manifest. V2.1.11 supersedes this original one-root manifest with the 400-document schema-v2 corpus above.

**Example.** If acquisition stops after English and Hindi, the next invocation validates those 20 payload hashes, skips their network calls, and continues with Telugu. If one cached byte changes—or a manifest path is changed to `../outside.txt`—validation fails before the corpus can enter training.

**Proof and result.** Automated tests exercise clean acquisition, interrupted resume, zero-network reuse, payload tampering, duplicate content, malformed provenance and path traversal. The learned benchmark assigns whole documents—not paragraphs—to splits, then removes cross-split target duplication and verifies document identity, payload hash, target string and exact-sample isolation.

| Corpus invariant | Result |
|---|---:|
| Languages | 4 |
| Documents per language | 10 |
| Total manifested documents | 40 |
| Document-level train/validation/test overlap | 0 |
| Document-content SHA-256 overlap | 0 |
| Cross-split target-string overlap | 0 |
| Cross-split exact-sample overlap | 0 |
| Selected cross-block spans | 4,000 / 800 / 500 |
| Test decisions checked by future-byte firewall | 17,212 |
| Payload and manifest hashes verified | PASS |
| Interrupted acquisition resume tested | PASS |
| Tampered payload/path traversal rejected | PASS |

The acquisition is resumable: a completed language is reused only after its files and hashes validate, and the final manifest is published atomically. A valid final corpus is a zero-network cache boundary. The benchmark reads only manifest-listed files; stale or unlisted files cannot enter training.

This was the first proof of document isolation. V2.1.9 later superseded its long-word target filter, V2.1.10 balanced languages, and V2.1.11 replaced its single topical neighborhood. The table remains as historical evidence rather than a description of the current manifest.

### V2.1.6 Production-scale qualification — current verdict

**Problem.** A collection of passing unit tests or a relative NLL win can be mistaken for production readiness even when absolute generation, data coverage, hardware, or recovery evidence is missing.

**Solution.** `run_experiment.py` compiles a machine-readable gate matrix in `artifacts/production_readiness.json`. A costly-test candidate is `YES` only when every named required gate is `PASS`; `NOT_RUN`, `PARTIAL`, and relative-quality successes cannot hide a failed absolute requirement.

**Example.** On the current balanced span stream, the sampled V2.1.5 model generates 5/500 exact RKE suffixes after a frozen first block, but 0/500 complete spans from only two context words. V2.1.6 therefore records non-zero learned mechanics separately from the failed 20% continuation-quality gate and leaves 128-token conditioning as `NOT_RUN` instead of collapsing three different claims into one score.

**Proof and result.** The table below is regenerated from measured artifacts by the one-command run. Its present answer is intentionally `NO`.

**Candidate for costly production-scale testing: NO.** The verdict is generated from `artifacts/production_readiness.json`, not manually asserted here.

| Qualification level | Result | Evidence |
|---|---|---|
| Reversible EOS codec | PASS | Exhaustive/random round trips |
| Full 256-byte UTF-8 pathway | PASS | Controlled four-script experiment |
| Natural single-block next-token LM | PASS | Matched causal RKE/fallback evaluation |
| Three-seed quality parity | PASS | Means, sample SD and 95% confidence intervals |
| Dynamic continuation codec | PASS | Exact payloads through 10,000 bytes |
| Neural multi-block mechanics | PASS | CONT/EOS loss, PAD mask and 100% oracle reconstruction |
| Learned cross-block sampled model | **FAIL** | Teacher-forced NLL improves, but open-ended exact is zero and 6.4% of RKE chains miss EOS |
| Learned cross-block mechanism | **PASS** | Frozen block 0 → generated suffix: RKE 5/500, fallback 0/500 |
| Production continuation quality | **FAIL** | Causal exact 22 vs 18 but below 20%; raw micro/macro NLL are 16.66%/16.82% worse |
| Open-ended full-span exact | **FAIL** | Both two-word-context arms have zero full-span exact; diagnostic, not isolated head gate |
| 128-token context conditioning | NOT RUN | Current open-ended arm sees two preceding words |
| Matched tokenizer/BPE fallback | NOT RUN | Present only in the synthetic task, not the natural causal comparison |
| Cross-block three-seed stability | NOT RUN | Current learned continuation pilot uses one seed |
| NumPy ↔ PyTorch CPU parity | PASS | Logits, loss, gradients and Adam update |
| Accelerator and mixed precision | NOT RUN | No CUDA or MPS device is exposed |
| Topic-stratified multilingual corpus | PASS | 400 revision-pinned documents; 4 languages × 10 topics × 10 documents, with manifest and payload hashes verified |
| Cross-block coverage in every language | **PASS** | Complete-token spans provide eligible held-out targets in all four languages |
| Equal cross-block source-language quotas | **PASS** | 1,000/200/125 per language for train/validation/test |
| Per-language + macro/micro reporting | **PASS** | Generation, retrieval and NLL reports expose all aggregation levels |
| Unicode security suite | PARTIAL | Valid UTF-8 constraints pass; broader attacks remain |
| Distributed recovery | NOT RUN | Required before production training |

Parameters required before the costly-test label may become `YES`:

| Area | Minimum production-test parameter |
|---|---|
| Corpus | Versioned, licensed, content-hashed, at least 100 documents per language across 10 declared topic strata |
| Languages | Current four plus representative Latin, Indic, Arabic-derived and CJK coverage |
| Cross-block test | At least 500 natural complete-token spans longer than one block, with equal language quotas |
| Split | Document-level isolation with zero content-hash overlap |
| Context | At least 128 preceding tokens; current two-word context is insufficient |
| Blocks | 24 bytes per block, configurable maximum, explicit CONT/EOS |
| Seeds | At least 3 paired seeds per arm with Student-t confidence intervals |
| Controls | Identical body, data order, optimizer budget and hardware for RKE and fallback |
| Quality | Raw and calibrated RKE NLL ≤ 1.01× fallback and exact ≥ 0.99× fallback |
| Efficiency | At least 10% total parameter reduction and decode throughput ≥90% of fallback |
| Generation | 100% valid chains, zero missing EOS, zero premature EOS and zero truncation |
| Runtime | Mixed-precision throughput, latency, memory and utilization on the target accelerator |
| Reliability | Deterministic checkpoint/resume and distributed failure recovery |

The rationale for these thresholds and the separation between matched baselines and external Sarvam/Kimi ceilings is documented in [`BENCHMARK_ALIGNMENT.md`](BENCHMARK_ALIGNMENT.md).

### V2.1.5 Learned cross-block natural LM — mechanism PASS, production quality FAIL

**Problem.** V2.1.4 proved that continuation blocks can be represented and decoded, but its neural model was an oracle: it did not prove that a language model could *learn* the next bytes of natural text after crossing a block boundary without seeing future target bytes.

**Solution.** Train matched RKE and byte-softmax models on 4,000 train, 800 validation and 500 document-isolated test targets of 25–96 UTF-8 bytes. In the original V2.1.5 run each target was one long word; V2.1.9 now constructs complete-token continuation spans without changing the decoder protocol. For block zero the model sees two preceding context words and the within-block prefix. For later blocks it sees the generated previous block and current generated prefix. Teacher-forced training exposes only the ground-truth prefix up to the current decision—never the unseen suffix or complete target block.

**Example.** The actual held-out 41-byte Hindi target `नाम से प्रसिद्ध` becomes a 24-byte block ending in `CONT`, followed by the remaining 17 bytes and `EOS`. At test time the second block is conditioned on the model's first block, not the correct hidden target. A mistake in block zero therefore propagates exactly as it would in deployment.

```text
context → generate block 0 + CONT → feed generated block 0
        → generate block 1 + EOS  → reconstruct complete word
```

**Proof and result.** A future-byte firewall checks all 17,212 test decisions, and saved predictions expose every target/predicted byte span and terminator. Two protocols prevent the model body and output head from being confused: (A) open-ended generation starts from two context words; (B) continuation generation freezes the real first block and requires the model to generate every remaining byte and EOS. Protocol B directly tests the cross-block claim.

| Arm | Parameters | Separate vocabulary classifier | Teacher-forced NLL | Open-ended byte accuracy | Full-span exact | Gold-block continuation exact |
|---|---:|---:|---:|---:|---:|---:|
| Continuation RKE | **162,672** | **0** | **1.8334** | **20.16%** | 0% | **1.0% (5/500)** |
| Byte fallback | 214,272 | 51,600 | 1.9923 | 18.53% | 0% | 0% (0/500) |
| Exact-prefix retrieval control | — | — | — | — | — | 0% (0/500) |

Both matched arms use `d_slot=8`, `d_model=200`, 4,000 optimizer steps and the same 13/5/46 block-start/terminator/interior sampling mix. RKE passes document/target/sample isolation, the future-byte firewall, micro and macro NLL parity, relative byte accuracy and zero structurally premature block-zero EOS. Its 5/500 exact continuations beat both fallback and retrieval in this run, so the narrow mechanism-comparison gate passes. The aggregate still fails because open-ended byte accuracy is below 25%, 6.4% of RKE chains miss EOS, open-ended exact is zero, and continuation exact is far below the fixed 20% gate. Predicting a valid span shorter than the reference is reported separately as a language error; it is not mislabeled as malformed EOS.

The open-ended failure is retained as a two-word-context/tiny-body limitation. It is not used as the isolated RKE-head mechanism check; V2.1.6 separately requires a future 128-token conditioning experiment.

The original long-word pool was imbalanced because the >24-byte condition selected scripts by UTF-8 width. V2.1.9 fixes that construction defect and gives every source language held-out coverage; V2.1.10 adds equal quotas and explicit macro/micro reports.

Artifacts: `artifacts/cross_block_lm/results.json`, `predictions.json`, `split.json` and both saved models.

### V2.1.4 Neural continuation mechanics — PASS

**Problem.** The original Kronecker layout reserves 32 positions for every token, wastes most positions on short words, and truncates anything longer than the fixed limit. A codec-only dynamic-block proposal would still leave batching, loss masks and neural decoding unproved.

**Solution.** `ContinuationByteCodec` uses 24-byte ragged blocks and 259 states: PAD, EOS, CONT and 256 byte values. A short payload allocates one block; a longer payload adds blocks only as needed. `CONT` says another block follows, `EOS` closes the final block, and only slots after either terminator are PAD/loss-masked.

**Example.** `apple` is encoded as five byte states plus `EOS` in one active block. The 51-byte `దక్షిణాఫ్రికాలోని` uses three blocks: 24 bytes + `CONT`, 24 bytes + `CONT`, then 3 bytes + `EOS`. No byte is cropped, and unused future blocks are never allocated.

**Proof and result.** The discrete codec round-trips payloads through 10,000 bytes. A tied-prototype neural oracle reconstructs 150 disjoint held-out payloads, 336 blocks and 6,850 bytes through 96-byte payloads with 100% exact chains. Tests verify that CONT, EOS and all bytes bear loss while post-terminator PAD does not. This proves neural batching and decoding mechanics, not learned language quality; V2.1.5 addresses that next.

### V2.1.3 Natural single-block and multi-seed quality — PASS

**Problem.** The earlier copy/composition tasks proved reversibility and unseen-string emission, but they did not establish next-token quality on natural multilingual text. A single lucky seed could also make a small experiment look stronger than it is.

**Solution.** Run next-word models on natural English, Hindi, Telugu and Sindhi paragraphs. The single-block benchmark samples 4,000 training, 800 validation and 800 test transitions, balanced across languages and isolated by paragraph. It compares vocabulary softmax, autoregressive byte fallback, fully parallel RKE, two parallel refinements, and causal RKE under recorded parameter/training budgets. The causal RKE/fallback comparison uses paired seeds, identical shared-body initialization, identical sampled decision streams and 5,000 updates per arm. With only three pairs, 95% intervals use Student's t (`df=2`), not the asymptotic `1.96` multiplier.

**Example.** Given only preceding natural words, every byte arm must emit the next word byte-by-byte or slot-by-slot and terminate with EOS. A held-out Hindi or Sindhi word remains representable even when it has no vocabulary row; the vocabulary arm reports its coverage rather than pretending an unknown-token loss is the word's NLL.

**Proof and result.** Whole paragraphs are assigned to exactly one split, UTF-8 decoding is constrained, PAD after EOS is loss-masked, and matched arms report exact match plus byte/EOS NLL:

| Output arm | Test exact | Test NLL | Target coverage | Parameters |
|---|---:|---:|---:|---:|
| Vocabulary softmax (512 words) | 41.875% | 2.3359 per representable word | 66% | 92,632 |
| Autoregressive byte fallback | **19.000%** | **1.7640 per byte/EOS** | 100% | 67,032 |
| Parallel RKE-Head | 17.250% | 2.2483 per byte/EOS | 100% | **41,332** |
| Masked parallel RKE | **18.625%** | 2.1395 per byte/EOS | 100% | 41,560 |
| Two-pass refined RKE | 17.250% | 2.2553 per byte/EOS | 100% | 41,560 |
| **Causal RKE-Head** | **17.875%** | **1.8042 per byte/EOS** | **100%** | **41,332** |

The vocabulary NLL uses a different unit and excludes OOV targets, so it is not numerically comparable to byte-normalized NLL. The earlier unpaired run is superseded: its `1.96 × SE` intervals were too narrow for three seeds. Regenerated artifacts report paired-seed results with Student-t intervals. Causal RKE keeps zero separate vocabulary-classifier parameters but gives up parallel decoding.

### V2.1.2 Parallel alternatives — informative negative results

**Problem.** The original parallel RKE predicts every byte slot independently from the same hidden token state. Natural spelling has strong left-to-right dependencies, so parallel speed may be purchased with worse NLL than an autoregressive byte fallback.

**Solution.** Test two bounded-compute alternatives. `MaskedSlotRKE` adds an eight-lag causal slot convolution in one fixed-depth pass. `RefinedSlotRKE` first proposes every slot, then performs one temperature-sharpened refinement pass conditioned on those proposals.

**Example.** After proposing the UTF-8 prefix bytes of a word, a causal/refinement path can alter a later byte using earlier-slot evidence; slot 7 can depend on slots 0–6, while slot 0 cannot leak information from slot 7. Both alternatives retain a fixed one- or two-pass decode rather than one model call per byte.

**Proof and result.** Finite-difference tests cover refinement weights, the tied codebook and transformer body, and a causality test perturbs a later source slot to prove it cannot affect an earlier output. Both variants improve or illuminate the baseline but miss the predefined byte-fallback NLL parity gate. This section is a recorded negative result, not a claimed fix; V2.1.3 shows that causal RKE closes the gap by accepting sequential decoding.

### V2.1.1 Framework correctness — PASS

**Problem.** The initial research model used hand-written NumPy forward/backpropagation. Strong toy results are not worth scaling if a gradient, mask or optimizer defect is responsible, and NumPy alone is not the intended production training framework.

**Solution.** Port the same RKE computation to PyTorch without changing architecture or parameters, then load an identical state and compare both implementations on the same masked minibatch.

**Example.** One test computes logits, EOS/PAD-masked loss, every parameter gradient and one Adam update in NumPy and PyTorch. It compares corresponding tensors directly rather than merely checking that both losses decrease.

**Proof and result.** Maximum observed differences are `3.5e-18` for logits, `2.6e-10` for masked loss, `1.1e-17` for gradients and `1.4e-17` for one Adam update. CPU framework parity passes. Accelerator kernels and mixed precision remain untested because this environment exposes neither CUDA nor MPS.

## Run everything

Install dependencies once:

```bash
python3 -m pip install -r S7/requirements.txt
```

Then the complete deterministic demonstration is one command:

```bash
python3 S7/run_experiment.py
```

It validates the checked-in corpus hashes, regenerates `S7/artifacts/`, trains every model, evaluates the held-out sets, creates the reports, records dataset/model hashes and exits non-zero if any functional claim fails. The generated production-readiness verdict may still be `NO`; unresolved research gates are evidence, not demo-process failures.

To deliberately refresh the revision-pinned source corpus (network required), run `python3 S7/acquire_multidoc.py`. With the current valid manifest it performs a zero-network validation/reuse path.

Tests:

```bash
python3 -m unittest discover -s S7/tests -v
```

Open `S7/artifacts/report.html` for the visual result.

The four other assignment directions are developed as independent proposals in [`OTHER_IDEAS.md`](OTHER_IDEAS.md): algebra registers, multimodal tensor Kronecker, ragged dynamic blocks, and phase-shifted Fourier characters.

## Starting point: what the paper does

The referenced [Kronecker Embeddings paper](https://arxiv.org/abs/2605.29459) represents a token byte sequence with a sum of byte-position basis vectors:

```text
κ(b₁…bₗ) = (1/√L) Σₚ one_hot(byte=bₚ) ⊗ one_hot(position=p)
```

The paper uses 256 byte values and `d_p=32`, z-normalizes this fixed code, then learns a `D → d_model` input projection. It retains the conventional vocabulary output matrix. The paper explicitly identifies output-side Kronecker decoding as an untested future hypothesis and notes that bytes after `d_p` are truncated.

RKE-Head tests a concrete version of that output-side hypothesis.

## Construction

For a maximum of `P` bytes, allocate `P+1` ordered slots. Each slot has 258 states:

```text
{ PAD, EOS, byte_0, byte_1, …, byte_255 }
```

For a byte string `s` of length `L ≤ P`:

1. slots `0 … L-1` contain their corresponding byte states;
2. slot `L` contains EOS;
3. later slots contain PAD.

This is still a Kronecker code: each occupied feature is `one_hot(symbol) ⊗ one_hot(position)`. EOS is the necessary addition that separates prefixes such as `a` and `aa` and lets decoding recover length.

The model partitions its hidden state into `P+1` position slots of width `d_slot`. A shared codebook `E ∈ R^(258 × d_slot)` maps byte states into every input slot. Output uses the same matrix in reverse:

```text
input slot:    h_p = one_hot(symbol_p) E
output logits: z_p = h_p Eᵀ
decode:        symbol_p = argmax(z_p)
```

There is no token embedding lookup and no vocabulary-sized classifier. There is also no separate byte output matrix: `E` is tied in both directions.

## Why it is reversible

**Theorem.** The discrete codec `C` is injective over all byte strings of length at most `P`.

**Proof.** Assume `C(s)=C(t)`. The unique EOS state occurs at the same slot in both codes, so `|s|=|t|`. Every earlier slot is one-hot and equal, so every corresponding byte is equal. Therefore `s=t`. Decoding scans slots up to EOS and returns those bytes, hence `decode(C(s))=s`. ∎

For the tied continuous codebook, normalize each distinct prototype row. A prototype’s cosine similarity with itself is 1; its similarity with every distinct row is strictly below 1. Nearest-prototype decoding therefore returns the original state. The generated full-byte proof checks this numerically for all 258 states.

This proof is about a correctly emitted code. A neural model can still emit the wrong code; that is why the experiment separately measures exact held-out decoding.

## Evidence design

The experiment is falsifiable in four independent ways.

### 1. Full-byte analytic round trip

`FullByteCodec` tests the empty string, every one-byte value from 0 through 255, and 1,000 deterministic random binary strings. It also tests nearest-prototype reversal for all PAD/EOS/byte states.

Pass condition: every payload and every prototype round-trips exactly.

### 2. Truncation counterexample

With the original paper codec restricted to four positions, `abcd00` and `abcd11` have the same information-bearing code because only `abcd` remains. RKE configured for six bytes assigns different codes and exactly decodes both.

Pass condition: the original supports collide and the RKE codes differ.

### 3. Tiny-transformer OOV experiment

The pure-NumPy model is a genuine one-layer causal, single-head transformer. It receives three structured tokens:

```text
[ COPY_COMMAND, DISTRACTOR, PAYLOAD ] → PAYLOAD
```

It trains on 700 unique strings and evaluates on 250 disjoint strings. The split is hash-deterministic and its hash is written to `results.json`. No held-out target occurs in the training vocabulary. The model predicts all symbol-position slots in parallel through the tied byte codebook.

Pass condition: at least 90% exact token match on held-out OOV strings. The current generated run reaches **100%**.

The first dense-projection prototype of this experiment reached only 77.2% OOV exact match and failed the threshold. Constraining the projection to share the same symbol map across positions removed whole-token memorization and produced the passing result. This failure informed the final architecture; it was not hidden by lowering the threshold.

### 4. Fixed-vocabulary baseline

A normal vocabulary softmax can only select a row that exists in its output matrix. Because all 250 test strings are excluded from the training vocabulary, its exact representational coverage is 0/250 without byte fallback or retokenization. RKE-Head can emit their byte composition directly.

This is a capability comparison, not a claim that RKE has better language-model perplexity.

## Generated results

After `python3 S7/run_experiment.py`:

- `artifacts/results.json` — machine-readable claims, metrics, hashes and limitations;
- `artifacts/proof_records.json` — every discrete analytic round trip;
- `artifacts/split.json` — exact train and held-out OOV tokens;
- `artifacts/training_curve.csv` and `training_curve.svg` — measured optimization history;
- `artifacts/model.npz` — trained NumPy parameters;
- `artifacts/report.html` — self-contained visual summary;
- `artifacts/lm_v2/results.json` — matched next-token NLL, calibration, parameters and decode speed;
- `artifacts/lm_v2/comparison.svg` — next-token held-out comparison;
- `artifacts/lm_v2/predictions.json` — every held-out target and reconstructed output;
- `artifacts/lm_v2/split.json` — the exact training, seen-control and held-out composition split.
- `artifacts/multilingual/results.json` — full-byte aggregate, per-script and byte-length metrics;
- `artifacts/multilingual/predictions.json` — Unicode targets, UTF-8 bytes and decoded outputs;
- `artifacts/multilingual/split.json` — exact multilingual train and whole-word holdout split.
- `artifacts/natural_corpus/results.json` — real-corpus NLL, exact match, calibration, coverage, speed and hashes;
- `artifacts/natural_corpus/split.json` and `predictions.json` — paragraph-isolated examples and decoded outputs;
- `data/multidoc/manifest.json` — revision, license, path, byte count and SHA-256 for all 40 source documents;
- `artifacts/cross_block_lm/results.json` — document-isolated cross-block metrics, firewall and quality gates;
- `artifacts/cross_block_lm/split.json` and `predictions.json` — exact continuation-span stream and generated byte chains;
- `artifacts/torch_continuation_lm/results.json` — full-sequence causal continuation, matched controls and calibration;
- `artifacts/production_readiness.json` and `.md` — generated quality and deployment gates.

At the demo width, a one-million-token vocabulary head would have `d_model × 1,000,000` parameters. The bounded parallel RKE experiment adds no output-specific adapter and reuses the fixed-size input byte codebook. The causal continuation model separately reports and counts its structured adapter while still eliminating the vocabulary-sized classifier. `results.json` computes the exact comparison rather than hardcoding it.

## V2 methodology and audit details

There are 1,000 possible two-character-stem/one-character-suffix words. The experiment trains on 750 and evaluates on the other 250. All components are covered in training, while the whole-target overlap between train and held-out sets is exactly zero. `artifacts/lm_v2/split.json` records every example and `dataset_hash` binds that split into the results.

RKE and the vocabulary arm receive 1,200 optimizer steps. Fallback receives 4,800 steps because every three-byte word requires four target decisions including EOS. EOS bears loss; every PAD slot after EOS is masked. A test verifies that changing masked targets cannot change either loss or gradients.

Generated metrics include:

- word and byte/EOS-normalized NLL;
- exact next-token accuracy;
- 10-bin expected calibration error and Brier score;
- output and total parameter counts;
- median and p95 CPU decode time;
- completed words per second;
- UTF-8-constrained decoding evidence.

Closed-vocabulary OOV NLL is reported as undefined, not replaced by UNK NLL: a softmax with no row for the target assigns no probability to that string. Its confidence in a wrong known class is still included in calibration.

For text decoding, the full-byte path applies an incremental UTF-8 mask. Invalid continuation bytes are forbidden, and EOS is legal only at a complete code-point boundary. The generated evidence deliberately makes invalid byte `0xFF` the highest unconstrained logit and proves valid recovery of both `é` and `अ`.

This remains a controlled compositional language, not a natural-corpus perplexity claim. The V2.1 pilot above performs the corresponding frozen-split natural multilingual comparison and reports its weaker, more realistic metrics separately.

## Files

- `rke.py` — codecs, tiny transformer, manual backpropagation and Adam;
- `lm_compare.py` — matched vocabulary, byte-fallback and RKE next-token arms;
- `multilingual.py` — full-byte Hindi, Telugu, Tamil and Arabic experiment;
- `natural_corpus.py` — natural English, Hindi, Telugu and Sindhi matched-arm pilot;
- `continuation_neural.py` — multi-block neural batching, mask and tied-decoding oracle;
- `cross_block_lm.py` — learned natural continuation RKE versus matched fallback;
- `torch_continuation_lm.py` — causal full-sequence PyTorch continuation and matched fallback;
- `acquire_multidoc.py` — resumable revision-pinned acquisition and strict corpus validation;
- `torch_port.py` — parameter-identical PyTorch port and NumPy parity oracle;
- `run_experiment.py` — deterministic experiment and evidence generator;
- `tests/test_rke.py` — codec, gradients, causality, UTF-8 and split invariants;
- `tests/test_multidoc.py` — corpus resume, reuse, tamper and path-safety invariants;
- `BENCHMARK_ALIGNMENT.md` — matched-baseline, external-ceiling and threshold rationale;
- `OTHER_IDEAS.md` — four separate Kronecker V2 research proposals;
- `requirements.txt` — NumPy and PyTorch.

## Honest limitations and next experiments

1. The controlled matched experiment uses a ten-character subset for speed; both multilingual experiments use the full 258-state byte codec.
2. The natural pilot is small next-word prediction, not document-scale pretraining, semantic evaluation or generation-fluency evidence.
3. Output compute is `O(P × 258)`. It is vocabulary-independent, not free.
4. Learned cross-block conditioning is evaluated and has valid termination, but open-ended full-span exact remains a weak diagnostic and only one continuation seed is trained.
5. The raw byte codec can represent invalid UTF-8; the provided constrained decoder prevents it for text output, but production implementations must preserve that constraint.
6. Both tested fixed-depth parallel refiners miss byte-fallback NLL parity; causal RKE passes the pilot parity gate but gives up parallel decoding.
7. PyTorch CPU parity is proven; accelerator kernels, mixed precision, distributed checkpointing and Unicode security stress tests remain unverified.

The next paper-worthy alternative is blockwise causal decoding: emit small groups of slots in parallel while conditioning each group on completed preceding groups. It offers a measurable speed/quality continuum instead of assuming that one or two fully parallel passes can match the byte chain rule. Before costly testing, cross-block training needs non-zero exact generation over at least three seeds, context must grow from two to at least 128 tokens, source labels must be upgraded to target-level language/script validation, and PyTorch must be benchmarked on an accelerator.
