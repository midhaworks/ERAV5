# Independent Kronecker V2 research ideas

These are intentionally **separate hypotheses**. They should not be combined into one experiment: each changes a different assumption and needs its own baseline, proof obligation and failure criteria.

## Problem 1 — Mathematical structure inside embeddings

### Hypothesis: a protected algebra register

Append an exact, typed mathematical register to ordinary lexical dimensions:

```text
[ learned semantic dimensions | exact algebra dimensions ]
```

The register should be operated on by typed deterministic maps selected by the transformer, rather than asking attention to rediscover arithmetic. Useful channels include sign, integer/rational value, and prime-exponent coordinates. In prime coordinates, multiplication becomes addition: `factor(ab)=factor(a)+factor(b)`. A protected residual path would keep LayerNorm from destroying exact scale.

### Proof and experiment

Require representational equality, not only answer classification:

```text
operate(E(9), multiply, E(9)).math == E(81).math
```

Train operation selection on small numbers and test exact arithmetic far outside the training range. Compare learned number embeddings, digit encoders, and the algebra register on addition, multiplication, fractions, units, and multi-step expressions.

### Boundary

A fixed finite vector cannot represent all mathematical objects and proofs exactly. The defensible claim is closure over a precisely defined numerical algebra, not “all mathematics.”

## Problem 2 — One Kronecker family for text, images and audio

### Hypothesis: modality-coordinate tensor encoding

Generalize byte×position into:

```text
content × local_coordinate × modality
```

- Text: byte × character position × TEXT
- Image: quantized patch coefficient × x × y × channel × IMAGE
- Audio: quantized spectrogram coefficient × time × frequency × AUDIO

The modality basis prevents accidental equality between the same numeric value in unrelated modalities, while a shared learned projection can align their semantics.

### Proof and experiment

Before projection, distinct quantized value-coordinate-modality tuples occupy distinct basis coordinates, giving injectivity on the configured patch domain. Test on a controlled corpus where concepts appear as words, dot-pattern images, and tone-count audio. Measure reconstruction, cross-modal retrieval, parameter count, translation/shift robustness, and held-out layouts against separate modality projections and raw-byte Kronecker.

### Boundary

Raw bytes are universal but not automatically a useful inductive bias. Patch preprocessing and coordinate factorization are the actual research questions.

## Problem 3 — Dynamic length without 32 reserved positions

### Hypothesis: ragged block Kronecker

Split a token into small fixed blocks, for example eight bytes:

```text
"a"                    → 1 active block
"internationalization" → 3 active blocks
100-byte identifier     → 13 active blocks
```

Each block encodes byte×position-within-block×block-index. A lightweight pooler may produce one transformer token, while retaining the ragged block sequence when exact reconstruction is required. Compute and storage become proportional to actual length, and no byte after position 32 is discarded.

### Proof and experiment

The ordered block sequence is reversible because block boundaries and total length are explicit. Evaluate lengths 1–128, long common-prefix collision pairs, URLs, code identifiers, Indic UTF-8, combining marks, and emoji. Report exact reconstruction, collision rate, discarded bytes, memory, encoding latency and downstream quality against fixed `d_p=32`, `d_p=64`, character CNN and byte encoders.

### Boundary

An arbitrary-length string cannot be reversibly compressed into a fixed number of finite-precision values. Exactness belongs to the ragged block sequence, not necessarily its pooled fixed-width summary.

## Problem 4 — A real Fourier alternative

### Hypothesis: phase-shifted Fourier characters

Naively summing character waves loses order because addition is commutative: `wave(a)+wave(b)=wave(b)+wave(a)`. Position must enter as phase:

```text
F_k(word) = Σ_p amplitude(byte_p) · exp(-2π i k p / P)
```

Equivalently, build the byte×position indicator matrix and apply a discrete Fourier transform along its position axis. With all `P` frequencies, inverse DFT reconstructs every position exactly. This is a genuine change of basis from one-hot Kronecker position to Fourier position—not free compression.

### Proof and experiment

Exactness follows from `IDFT(DFT(X))=X`, with EOS carrying length. Compare full Fourier, reduced-frequency Fourier, random Fourier features, learned frequency selection, and original Kronecker on reconstruction, anagram separation, typo geometry, prefixes and shifted suffix families such as `nation/creation`.

### Boundary

Keeping all frequencies preserves the same information scale as Kronecker. Keeping fewer frequencies creates compression but necessarily makes reconstruction lossy. The promising question is whether controlled phase or magnitude invariance improves suffix geometry enough to justify that trade-off.

## Suggested order

1. Fourier alternative: clearest theorem and an answer to absolute-position suffix weakness.
2. Ragged blocks: clearest practical V2 fix for truncation and wasted positions.
3. Algebra register: high upside, but explicitly neuro-symbolic.
4. Multimodal tensor encoding: broadest experiment and most sensitive to preprocessing choices.
