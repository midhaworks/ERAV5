# Benchmark alignment and threshold rationale

This document separates three kinds of evidence that answer different questions. Scores must not be moved between these layers.

## 1. Controlled head ablation — primary evidence

The primary question is whether RKE changes quality or efficiency when only the output mechanism changes. Every arm must use the same body, data, split, initialization for shared parameters, batch stream, optimizer budget and hardware.

Required arms:

1. tied reversible RKE codebook;
2. separate byte softmax;
3. tokenizer/BPE vocabulary softmax with byte fallback;
4. exact-prefix retrieval as a memorization control.

The current pre-production thresholds are deliberately stricter than the earlier pilot gates:

| Metric | Required threshold | Reason |
|---|---:|---|
| Raw test NLL | RKE ≤ 1.01× matched fallback | Validation loss is the original paper's principal controlled metric; calibration cannot hide a training-quality gap |
| Calibrated test NLL | RKE ≤ 1.01× fallback | Checks probability quality after validation-only scalar calibration |
| Exact continuation | RKE ≥ 0.99× fallback | Non-inferiority, rather than a corpus-dependent absolute percentage |
| Retrieval | RKE exact > retrieval exact | Requires learned generalization beyond prefix lookup |
| Parameters | ≥10% total reduction and zero separate RKE output parameters | Establishes a material efficiency benefit |
| Decode throughput | RKE ≥90% of fallback on the same device | Prevents parameter savings from hiding an excessive runtime cost |
| Stability | At least three seeds with confidence intervals | A single run is exploratory, not confirmatory |

The 20% exact floor remains a development sanity check, not the main scientific comparison. Exact percentages depend strongly on corpus ambiguity and prefix length.

## 2. Existing Kronecker paper — methodological anchor

The [original Kronecker Embeddings paper](https://arxiv.org/abs/2605.29459) reports a controlled three-seed 124M nanoGPT experiment over 2.5B FineWeb-Edu tokens. Only the input embedding scheme differs. It reports `0.083 ± 0.007` nats lower validation loss, `2.5 ± 0.2%` relative improvement and about `1.43×` fewer steps to reach the BPE arm's converged loss.

Those values are not thresholds for this output experiment, but they establish the standard of evidence: same-body controls, training-scale validation loss, multiple seeds and efficiency measurements.

The paper's six-model embedding probe uses Llama-3.2-1B, Qwen3-32B, Gemma-3-1B-pt, DeepSeek-V3-Base, GPT-OSS-120B and SmolLM2-135M. It is a representation-geometry probe, not a suffix-generation leaderboard, so its loose-morphology and anisotropy statistics cannot be compared numerically with continuation exact match or NLL.

## 3. External pretrained ceilings — contextual evidence

Sarvam and Kimi are useful external ceilings only after running the same frozen evaluation protocol. Their published benchmark scores are not substitutes for that run.

| Model | Published scale/capability | Why it is not a primary baseline |
|---|---|---|
| [Sarvam-30B](https://huggingface.co/sarvamai/sarvam-30b) | 32B total MoE, about 2.4B active non-embedding parameters; Indian-language focus | Vastly more pretraining and a different tokenizer/body |
| [Sarvam-105B](https://huggingface.co/sarvamai/sarvam-105b) | 106B total, 10.3B active; 128K context and Indian-language focus | Published MMLU, math, coding and agentic scores do not measure structured continuation |
| [Kimi K2](https://huggingface.co/moonshotai/Kimi-K2-Instruct) | 1T total MoE, 32B active, 160K vocabulary | General frontier/agentic ceiling, not an Indic or head-isolation control |

For an external run, every model should receive the same frozen Unicode-safe prefix and context. Report suffix exact match, byte accuracy, normalized edit distance, valid UTF-8, latency and—where token logits are available—NLL normalized by target UTF-8 bytes. Results must be labelled as potentially contaminated because large pretraining corpora may contain the source Wikipedia pages.

## Scale ladder

Passing the tiny controlled experiment permits a medium-scale test; it does not establish production readiness.

1. Current sub-million-parameter mechanism test.
2. Three-seed 100M–150M same-body ablation with a genuinely blind multilingual split.
3. Replication at approximately 1B parameters if the 100M-scale confidence intervals pass.
4. External Sarvam/Kimi ceiling evaluation on the frozen set.
5. Only then consider costly production-scale training.
