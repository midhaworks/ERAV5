# Research roadmap: improving RKE-Head without hiding the trade-offs

This roadmap asks one narrow question: **how can a reversible, vocabulary-independent
Kronecker output head close the natural-language quality gap while retaining a material
parameter or runtime advantage?** It does not claim that any untested proposal works.
Every proposal below has a matched ablation and a rejection criterion.

The literature review was refreshed on 20 August 2026. “New proposal” means a proposed
combination for this project; it is not a formal claim of publication-level novelty. A
broader prior-art and patent search would still be required before making such a claim.

## 1. What the present evidence says

The controlled synthetic experiments establish reversibility, explicit EOS, full-byte
UTF-8 support, output-vocabulary independence and unseen-string composition. The natural
continuation experiment is more important for deciding what to build next: the causal RKE
head is efficient and learns, but its held-out sequence NLL and exact generation do not yet
match the byte fallback. Calibration cannot repair the raw-NLL gap.

The likely bottlenecks are separable:

| Bottleneck | Observable symptom | Experiment that isolates it |
|---|---|---|
| Independent output slots | Locally inconsistent bytes; parallel head trails causal bytes | Hold the transformer body fixed and add only structured transitions |
| Fixed 24-byte blocks | Waste on easy text, repeated boundaries on difficult text | Compare fixed blocks with entropy- or boundary-adaptive blocks at matched FLOPs |
| Weak context body | Both RKE and fallback have low exact continuation | Replace the two-word context with 128+ tokens before judging the head |
| Optimization of a tied codebook | Raw NLL gap remains even when decoding is valid | Codebook geometry and distillation ablations with identical inference |
| Multilingual/data noise | Large macro/micro or per-language gaps | Target-level language/script checks and source-held-out evaluation |

These must not be conflated. A larger body may lift every arm while leaving the output-head
gap unchanged; a better dataset may improve exact match without validating the factorization.

## 2. Strongest relevant methods in prior work

| Research line | Primary source | Result relevant to this project | What it does **not** prove here |
|---|---|---|---|
| Kronecker input embeddings | [Kronecker Embeddings](https://arxiv.org/abs/2605.29459) | Deterministic byte-position factorization removed most input embedding parameters and improved a controlled 124M-model validation loss | It did not test a reversible output distribution |
| Input/output weight tying | [Press & Wolf](https://arxiv.org/abs/1608.05859), [Inan et al.](https://arxiv.org/abs/1611.01462) | Tying can reduce parameters and improve language-model quality | Tying alone does not make arbitrary strings decodable or remove the vocabulary normalization |
| Token-free multilingual models | [CANINE](https://arxiv.org/abs/2103.06874), [ByT5](https://arxiv.org/abs/2105.13626), [Charformer](https://arxiv.org/abs/2106.12672) | Character/byte models can be competitive; downsampling or learned latent subwords reduce sequence cost | They do not provide this tied reversible output construction |
| Multiscale bytes | [MEGABYTE](https://arxiv.org/abs/2305.07185) | A global model between patches plus a local model within patches can model long byte sequences efficiently | It is still autoregressive locally and is not an output-head-only ablation |
| Selective byte computation | [SpaceByte](https://arxiv.org/abs/2404.14408), [BLT](https://arxiv.org/abs/2412.09871) | Boundary- or entropy-selected patches allocate expensive computation where bytes are difficult | Dynamic patches do not automatically preserve exact reversibility or fair multilingual costs |
| Parallel future prediction | [Multi-token Prediction](https://arxiv.org/abs/2404.19737) | Auxiliary prediction of several future tokens improved sample efficiency and enabled faster inference in larger models | Independent future heads are not a joint normalized string probability |
| Structured non-autoregressive decoding | [Fast Structured Decoding](https://arxiv.org/abs/1910.11555), [Blockwise Parallel Decoding](https://arxiv.org/abs/1811.03115) | Local structured dependence or validation can recover quality lost by independent parallel predictions | Results in translation do not establish RKE language-model NLL |
| Output-distribution expressivity | [Breaking the Softmax Bottleneck](https://arxiv.org/abs/1711.03953) | A single low-rank softmax distribution can be too restrictive for language | A mixture is useful only if its added parameters and compute remain fairly counted |
| Large-vocabulary baseline | [Adaptive Softmax](https://arxiv.org/abs/1609.04309) | Frequency-based clusters reduce expected large-vocabulary computation | It remains vocabulary-dependent and is a baseline, not a solution to unseen strings |
| Calibration | [Guo et al.](https://arxiv.org/abs/1706.04599) | Validation-only temperature scaling is a strong simple calibration baseline | Calibrated NLL must not replace raw NLL when judging learning quality |
| Multilingual representation cost | [Petrov et al.](https://arxiv.org/abs/2305.15425) | Encoding length and cost can differ greatly by language, including for bytes/characters | “Tokenizer-free” does not mean linguistically or computationally fair |
| Multilingual data curation | [CulturaX](https://arxiv.org/abs/2309.09400), [Dolma](https://arxiv.org/abs/2402.00159), [IndicLLMSuite/Sangraha](https://aclanthology.org/2024.acl-long.843/) | Language ID, filtering, decontamination, deduplication and source diversity are necessary at scale | Corpus size or a “clean” label is not evidence that this head is better |

## 3. New proposal A: K-CRF, a structured Kronecker output head

### Problem

The parallel head currently approximates a string distribution as independent slot
decisions conditioned on one context vector. That is cheap, but `q`, the following UTF-8
continuation byte, EOS and CONT are not independent. Autoregressive fallback has access to
the previous emitted bytes and therefore solves an easier statistical problem.

### Proposed solution

Keep the tied Kronecker codebook as the **unary** score for each byte/EOS/CONT state, then
add a small chain-structured model over output positions:

```text
score(y | h) = Σᵢ unary(yᵢ, h, i) + Σᵢ transition(yᵢ₋₁, yᵢ, h, i)
```

Hard transition masks encode the UTF-8 finite-state machine, legal EOS placement and CONT
rules. Learned transitions use a positive low-rank factorization, so sum-product training
can be implemented in roughly `O(block_length × states × rank)` rather than enumerating a
vocabulary. Exact normalization yields a real sequence NLL. Viterbi or a small exact/beam
decoder returns the best valid byte string. Parameters depend on 259 states and the chosen
rank—not on one million possible token strings.

Example: if a unary draft assigns high scores to the leading byte of a three-byte Telugu
character, the UTF-8 automaton removes EOS and ASCII from the next transition; learned
factors distinguish plausible continuation bytes. The model remains able to emit a string
never observed as a whole.

### Why this may be a real contribution

CRFs have been used to restore dependencies in non-autoregressive sequence models, and
Kronecker factorization provides vocabulary-free unary codes. The proposed contribution is
their combination with a tied reversible byte codebook, exact EOS/CONT semantics and UTF-8
automaton constraints. The search performed for this roadmap did not find that exact
combination; that is a hypothesis for a fuller novelty review, not proof of novelty.

### Proof plan

Use one frozen transformer body, shared initialization and the exact same batch stream for:

1. independent parallel RKE;
2. K-CRF ranks 4, 8, 16 and 32;
3. autoregressive byte fallback;
4. tokenizer softmax with byte fallback;
5. exact-prefix retrieval control.

Report raw and calibrated NLL per target byte/EOS, sequence exact, byte accuracy, valid UTF-8,
ECE, parameter counts, peak memory, training tokens/s and greedy decode strings/s. Reject K-CRF
if rank 16 does not recover at least half of the raw-NLL gap, if UTF-8 validity is below 100%,
or if its measured decode throughput falls below 90% of the matched byte fallback.

## 4. New proposal B: entropy-gated causal repair

### Problem

Fully parallel slots are fast but weak; fully autoregressive bytes are expressive but require
one model step per byte. Fixed grouping chooses the same compromise for `the`, a rare Sindhi
name and a malformed byte sequence.

### Proposed solution

Produce a one-pass RKE draft and its per-slot entropy. Accept a longest UTF-8-valid prefix
whose confidence exceeds threshold `τ`. A tiny tied local decoder repairs only the uncertain
suffix or uncertain groups of 4–8 bytes. Training mixes gold and sampled drafts and charges
the repair path for its actual FLOPs.

This combines BLT's entropy allocation with blockwise validation, but uses entropy to choose
**output repair depth**, not input patches. The vocabulary-independent codebook is used in
both draft and repair paths.

Example: an ASCII punctuation ending may complete in one pass; an Arabic character whose
first byte is certain but continuation bytes conflict gets one local repair group rather
than forcing 24 sequential calls.

### Proof plan

Sweep `τ` on validation only and draw a quality/latency Pareto curve. Include fixed group
sizes 24, 12, 8, 4 and 1 so the gate must beat a simpler policy. Report average and p95 repair
steps separately per language and target byte length. Reject the gate if a language receives
more than 1.25× the macro-average compute at matched NLL; otherwise a fast aggregate could
hide a multilingual latency tax.

## 5. New proposal C: cycle-consistent codebook geometry

### Problem

Tying says the same codebook is used in both directions, but it does not guarantee separated,
well-conditioned states or that a contextual prediction maps back near the correct input
representation. Rare UTF-8 continuation bytes can receive weak gradients.

### Proposed solution

Add three training-only terms:

- a tight-frame penalty on the shared codebook to avoid collapsed dimensions;
- a frequency-aware margin between confusable byte/EOS/CONT states;
- an encode → predict distribution → expected embedding cycle loss.

The codebook remains the only state dictionary at inference, so vocabulary independence is
unchanged. A rare-state sampler ensures every byte appears in auxiliary batches without
altering the measured natural-data mixture.

### Proof plan

Run each regularizer alone and together across at least three paired seeds. Measure codebook
condition number, nearest-state margin, rare-byte NLL, macro language NLL and overall NLL.
Reject the combined regularizer if its 95% paired confidence interval does not improve raw
NLL or if it improves only synthetic byte coverage while hurting natural text.

## 6. New proposal D: boundary state carry plus multi-block training

### Problem

CONT is currently a structural marker; it carries no learned summary of what the preceding
block intended. Long strings therefore expose the next block to a hard boundary exactly
where language dependencies continue.

### Proposed solution

After each block, compute a low-rank boundary state from the predicted byte embeddings and
feed it to the next block. Train with:

- ordinary next-block likelihood;
- a multi-block auxiliary objective predicting blocks 1…`k` from the same context;
- scheduled replacement of gold boundary states with predicted states;
- a consistency loss between a whole-span encoding and the composition of block states.

This borrows global/local separation from MEGABYTE and the auxiliary signal from multi-token
prediction, but preserves a single tied state codebook and explicitly tests exposure at RKE
block boundaries.

### Proof plan

Stratify results by 1, 2, 3 and 4 blocks. Require improvement to increase with block count;
otherwise the proposed mechanism has not fixed a boundary problem. Also test shuffled
boundary states: quality should collapse relative to correct states, proving the model uses
the carry rather than merely benefiting from added parameters.

## 7. New proposal E: factor each byte by UTF-8 role and nibbles

### Problem

A flat 256-byte codebook shares little statistical strength between rare bytes, even though
bytes have deterministic high/low nibbles and specific UTF-8 roles.

### Proposed solution

Represent a byte with three tied factors: UTF-8 role, high nibble and low nibble. Score the
intersection, not three independent outputs, and retain dedicated EOS/CONT/PAD states. A
small residual code handles distinctions the factors miss. This is an error-correcting
product code rather than a vocabulary classifier.

### Proof plan

Compare flat-state RKE, factors without residual and factors plus residual at equal total
head parameters. Hold out selected legal bytes from natural training but include them in a
synthetic coverage evaluation. The method must improve rare-byte and multilingual macro NLL
without reducing seen-byte NLL or allowing an illegal UTF-8 sequence.

## 8. Data improvements that should happen before scaling

The new 400-document Wikipedia corpus is a mechanism benchmark, not a pretraining corpus.
Before spending on a 100M–150M run:

1. Add a second licensed source family, preferably verified/native text from Sangraha for
   Hindi/Telugu/Sindhi plus a matched English source. Keep source-held-out test splits.
2. Run target-level language identification and script-mixture audits. Source edition is
   provenance, not proof of target language.
3. Apply exact and near-duplicate removal across sources before splitting. Record normalized
   and raw hashes so Unicode normalization cannot hide leakage.
4. Stratify not just by language/topic, but by suffix entropy, byte length, block count,
   script, code-switch rate and Unicode risk class.
5. Add a frozen adversarial suite: malformed/truncated UTF-8, combining marks, bidi controls,
   confusables, ZWJ/ZWNJ, emoji sequences, normalization pairs and mixed scripts.
6. Preserve an untouched evaluation firewall. Filtering thresholds, entropy gates,
   temperatures and early stopping are chosen on validation only.

Synthetic translation should be a labelled lane, never silently pooled with native text.
Report it separately because it can improve volume while narrowing linguistic variety.

## 9. Ordered experiment ladder and stop rules

| Order | Experiment | Cost | Promotion rule |
|---:|---|---:|---|
| 1 | 128-token matched body on the 400-document corpus | Small | Both byte fallback and RKE must learn materially above retrieval |
| 2 | K-CRF ranks and fixed causal groups | Small | Recover ≥50% of raw-NLL gap with 100% valid UTF-8 |
| 3 | Entropy-gated repair | Small/medium | Dominate at least one fixed group on NLL and throughput, with no language compute tax |
| 4 | Codebook geometry and boundary carry, three paired seeds | Medium | RKE ≤1.01× fallback raw and calibrated macro/micro NLL |
| 5 | Second-source, source-held-out replication | Medium | Same conclusion across source families and no leakage |
| 6 | 100M–150M accelerator ablation | Expensive | ≥10% total parameter reduction, ≥90% decode throughput, exact ≥0.99× fallback |
| 7 | Approximately 1B replication | Very expensive | Only after confidence intervals pass at 100M–150M |

At every level, the tokenizer-softmax arm, byte fallback and retrieval control remain. A
proposal is not promoted because it raises exact match alone; it must meet NLL, calibration,
validity, parameter, memory and latency requirements under the same body and data stream.

## 10. Recommended next implementation

Implement **K-CRF rank 8 and 16 first**, alongside fixed causal groups of 4 and 8. This is
the cleanest diagnostic: it changes only output dependence, can use the existing 259-state
codec and exposes whether independence—not the transformer body or corpus—is the head's
main limitation. Entropy-gated repair should follow only if structured transitions close a
meaningful fraction of the NLL gap.
