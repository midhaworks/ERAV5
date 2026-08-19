# RKE-Head: a reversible, vocabulary-independent Kronecker output

**Session 7 · selected problem 5:** “How do I make a reverse of this, so the same embedding gives the same Kronecker? Can we remove the final vocabulary head?”

This submission proposes **RKE-Head (Reversible Kronecker Embedding Head)**. It replaces a `d_model × |V|` token classifier with an ordered position × byte-symbol code. The same small byte codebook is used forward for input and transposed for output. There is no separately learned final head, and its parameter count does not depend on the number of token strings.

The submission has two layers of evidence: an exact reversibility proof and a stronger **V2 matched next-token experiment**. It does **not** claim to have solved open-domain language modelling or unbounded token length.

## Headline result: V2 matched next-token experiment

The strongest experiment compares three output mechanisms on the same controlled compositional language. Every stem, suffix and character occurs in training, but all 250 evaluated whole-token combinations are absent from the training vocabulary:

```text
context: [JOIN, suffix, stem] → next token: stem + suffix
example: [JOIN, "3", "fa"] → "fa3"
```

All arms use the same structured inputs, hidden width, one-layer single-head causal transformer and batch size.

| Output arm | Seen exact | Held-out OOV exact | Held-out NLL per byte/EOS | Total parameters |
|---|---:|---:|---:|---:|
| Vocabulary softmax | 100% | 0% | Undefined: target has no class | 75,168 |
| Autoregressive byte fallback | 100% | 100% | 0.002269 | 21,888 |
| **Parallel RKE-Head** | **100%** | **100%** | **0.000522** | **21,096** |

For RKE, the current deterministic run also measures 10-bin ECE `0.002075`, Brier score `0.000025`, and zero separate output-head parameters. CPU decode throughput is recorded in `artifacts/lm_v2/results.json`; it is intentionally not treated as hardware-independent.

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

The first production-readiness step replaces synthetic compositions with next-word prediction from the repository's Wikipedia-derived English, Hindi, Telugu and Sindhi corpora. Unicode text is NFC-normalized and case-folded. Entire paragraphs—not individual word windows—are assigned by a deterministic hash to 80/10/10 train, validation and test partitions, preventing context leakage. The run samples 4,000 training, 800 validation and 800 test transitions, balanced across languages.

| Output arm | Test exact | Test NLL | Target coverage | Parameters |
|---|---:|---:|---:|---:|
| Vocabulary softmax (512 words) | 41.375% | 2.1965 per representable word | 66% | 92,632 |
| Autoregressive byte fallback | **17.875%** | **1.7961 per byte/EOS** | 100% | 67,032 |
| Parallel RKE-Head | 17.250% | 2.2483 per byte/EOS | 100% | **41,332** |
| Masked parallel RKE | **18.625%** | 2.1395 per byte/EOS | 100% | 41,560 |
| Two-pass refined RKE | 17.250% | 2.2553 per byte/EOS | 100% | 41,560 |
| **Causal RKE-Head** | **17.875%** | **1.8042 per byte/EOS** | **100%** | **41,332** |

The vocabulary NLL has a different unit and excludes OOV targets, so it must not be compared numerically with the byte-normalized rows. All RKE and fallback variants cover every selected target and produce 100% valid UTF-8. Parallel RKE is much faster in this small CPU decode benchmark, but its byte/EOS NLL is about 25.2% worse than fallback. The eight-lag masked slot convolution improves NLL to 2.1395 and exact match to 18.625% in one fixed-depth parallel pass, but still misses the 5% NLL gate. At the corrected 5,000-step convergence budget, causal RKE reaches 1.8042 NLL and 17.875% exact match, compared with fallback's 1.7961 and 17.875%. It retains the same 258-state tied codebook, zero separate output parameters and 41,332 total parameters, at the cost of autoregressive decoding.

The three-seed result is the qualification metric, not the best single run. Causal RKE mean NLL is `1.8101 ± 0.0246` (95% CI half-width) and mean exact match is `17.625% ± 0.374%`; fallback is `1.7987 ± 0.0097` and `18.042% ± 0.726%`. The causal means remain within the predefined 5% thresholds, so the multi-seed gate passes. Every seed, model hash, duration and metric is generated in `seed_stability`.

The fixed neural arm's 24-byte bound covers 99.75% of English, 96.17% of Hindi, 91.33% of Telugu and 99.95% of Sindhi words in these sources. `ContinuationByteCodec` now removes truncation at the representation layer with explicit `CONT` and `EOS` states; generated proofs round-trip payloads through 10,000 bytes. Neural batching across those blocks is not implemented yet, so that gate remains partial. See `artifacts/natural_corpus/results.json` for corpus hashes, splits, per-language and byte-length metrics, calibration, predictions, decode speed, model hashes and training curves; see `artifacts/production_readiness.md` for the gate assessment.

The measured NLL gap was addressed with a causal tied-codebook arm. During training it teacher-forces the preceding byte slots; during decoding it feeds the valid UTF-8 prefix back into the same transformer and scores the next RKE slot through `Ein.T`. This adds no classifier matrix and demonstrates that the quality loss came mainly from conditional independence between parallel slots.

`MaskedSlotRKE` tests the parallel alternative directly. A causal eight-lag convolution mixes only earlier latent slots into each later slot, but all positions are evaluated in vectorized fixed depth. Its finite-difference gradient and causal direction are tested. It improves the original parallel arm but does not observe actual previously chosen bytes, so it cannot reproduce the full autoregressive advantage. This negative result is retained in the evidence rather than labelled a fix.

`RefinedSlotRKE` then tests a fixed two-pass alternative. Pass one proposes every slot; its temperature-sharpened distributions are mapped back through the same `Ein` codebook; pass two uses a masked eight-lag refiner and emits every corrected slot together. The refiner is zero-initialized, so its initial output is exactly the proposal rather than a random perturbation. End-to-end finite-difference checks cover `Wref`, the tied `Ein`, and the transformer body. It reaches 2.2553 test NLL and therefore fails to improve on the simpler masked arm. This rules out this small refiner, not proposal/refinement methods in general.

Passing this pilot quality gate does not make the system production-ready. Long-token coverage, GPU execution, large-scale pretraining, distributed recovery and a broader Unicode security suite remain open gates in the generated readiness report.

## Production-test qualification

The PyTorch port loads the exact NumPy parameters and compares float64 logits, masked loss, every parameter gradient and one Adam update. Maximum errors are `3.5e-18` for logits, `2.6e-10` for loss, `1.1e-17` for gradients and `1.4e-17` for the optimizer step. CPU framework parity therefore passes. This environment exposes neither CUDA nor MPS, so accelerator and mixed-precision gates are not inferred from CPU behavior.

The generated verdict is currently **not a candidate for costly production-scale testing**. The remaining required blockers are neural continuation-block integration, accelerator benchmarks, a versioned multi-document corpus and the full Unicode security suite. This verdict is computed from named gates in `artifacts/production_readiness.json`; it is not a README assertion.

## Run everything

```bash
python3 -m pip install -r S7/requirements.txt
python3 S7/run_experiment.py
```

The second command deterministically regenerates `S7/artifacts/`, trains the model, evaluates the held-out vocabulary, creates a plot and visual report, and exits non-zero if any claim fails.

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
- `artifacts/production_readiness.json` and `.md` — generated quality and deployment gates.

At the demo width, a one-million-token vocabulary head would have `d_model × 1,000,000` parameters. The RKE output adds **zero** separate head parameters; it reuses the fixed-size input byte codebook. `results.json` computes the exact comparison rather than hardcoding it.

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
- `torch_port.py` — parameter-identical PyTorch port and NumPy parity oracle;
- `run_experiment.py` — deterministic experiment and evidence generator;
- `tests/test_rke.py` — codec and split invariants;
- `OTHER_IDEAS.md` — four separate Kronecker V2 research proposals;
- `requirements.txt` — NumPy and PyTorch.

## Honest limitations and next experiments

1. The controlled matched experiment uses a ten-character subset for speed; both multilingual experiments use the full 258-state byte codec.
2. The natural pilot is small next-word prediction, not document-scale pretraining, semantic evaluation or generation-fluency evidence.
3. Output compute is `O(P × 258)`. It is vocabulary-independent, not free.
4. Continuation blocks remove codec truncation, but the neural trainer still accepts only one fixed block per token.
5. The raw byte codec can represent invalid UTF-8; the provided constrained decoder prevents it for text output, but production implementations must preserve that constraint.
6. Both tested fixed-depth parallel refiners miss byte-fallback NLL parity; causal RKE passes the pilot parity gate but gives up parallel decoding.
7. PyTorch CPU parity is proven; accelerator kernels, mixed precision, distributed checkpointing and Unicode security stress tests remain unverified.

The next paper-worthy alternative is blockwise causal decoding: emit small groups of slots in parallel while conditioning each group on completed preceding groups. It offers a measurable speed/quality continuum instead of assuming that one or two fully parallel passes can match the byte chain rule. Before costly testing, continuation blocks must be integrated into neural loss/batching, PyTorch must be benchmarked on an accelerator, and substantially larger document-separated corpora must replace the four single-article sources.
