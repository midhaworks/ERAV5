# RKE-Head: a reversible, vocabulary-independent Kronecker output

**Session 7 · selected problem 5:** “How do I make a reverse of this, so the same embedding gives the same Kronecker? Can we remove the final vocabulary head?”

This submission proposes **RKE-Head (Reversible Kronecker Embedding Head)**. It replaces a `d_model × |V|` token classifier with an ordered position × byte-symbol code. The same small byte codebook is used forward for input and transposed for output. There is no separately learned final head, and its parameter count does not depend on the number of token strings.

The result is deliberately scoped. It proves exact reversibility for bounded byte strings and demonstrates learned OOV reconstruction in a controlled tiny-transformer task. It does **not** claim to have solved open-domain language modelling or unbounded token length.

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
- `artifacts/report.html` — self-contained visual summary.

At the demo width, a one-million-token vocabulary head would have `d_model × 1,000,000` parameters. The RKE output adds **zero** separate head parameters; it reuses the fixed-size input byte codebook. `results.json` computes the exact comparison rather than hardcoding it.

## Files

- `rke.py` — codecs, tiny transformer, manual backpropagation and Adam;
- `run_experiment.py` — deterministic experiment and evidence generator;
- `tests/test_rke.py` — codec and split invariants;
- `requirements.txt` — NumPy only.

## Honest limitations and next experiments

1. The neural experiment uses a ten-character subset for speed, although the separate analytic proof executes all 256 bytes.
2. Copying isolates output reversibility; it does not establish language-model quality, semantic calibration or generation fluency.
3. Output compute is `O(P × 258)`. It is vocabulary-independent, not free.
4. Length is bounded by `P`; RKE fixes information loss within that bound but does not solve truly unbounded words.
5. UTF-8 validity is not automatic. Arbitrary bytes can be emitted, so production decoding should constrain or validate UTF-8 when text is required.
6. A serious follow-up should compare vocabulary softmax, paper Hypothesis A, distributional Hypothesis B, and RKE-Head on matched-compute language modelling, including calibration and decoding speed.

The strongest paper-worthy next step is a multilingual next-token experiment with held-out whole-word types: train both heads on identical transformer bodies, then measure NLL, exact generation, OOV composition, parameter count and wall-clock decode cost.
