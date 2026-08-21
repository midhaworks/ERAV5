# The Attention Atlas — Session 8

An interactive, chronological field guide to attention mechanisms: what each one fixed, what it bought, what it gave up, and when to use it.

The site is deliberately plain HTML, CSS, and JavaScript. There is no build step, package manager, framework, server, or API key.

## Interactive features

- A live softmax explainer with selectable query tokens, temperature, and causal masking
- A KV-cache simulator comparing MHA, MQA, GQA, and MLA across 2K–1M-token contexts
- A searchable, family-filterable chronology of 22 mechanisms
- A six-act reading of what chronological order reveals, including the alternating compute and KV-cache crises
- A two-mechanism comparison tray for honest head-to-head trade-offs
- A context-cost explorer with workload-specific recommendations
- Source and historical-nuance dialogs on every mechanism
- A bonus ledger for Routing Transformer, FlashAttention, Parallel DeltaNet, and DeepSeek Sparse Attention

## Run locally

```bash
cd S8
python3 -m http.server 8080
```

Open `http://localhost:8080`.

## Deploy

### Netlify

Import the repository and use:

- Base directory: `S8`
- Build command: leave empty
- Publish directory: `.`

`netlify.toml` already contains the publish and security-header configuration.

### Cloudflare Pages

Import the repository and use:

- Framework preset: None
- Root directory: `S8`
- Build command: leave empty (or `exit 0` if the UI requires one)
- Build output directory: `.`

Cloudflare Pages reads the included `_headers` file.

## Dating method

Dates shown in the UI are the first public arXiv v1 submission date or, for release-first work, the official release date. They are not conference dates. This matters: conference publication often trails public introduction by months.

Two special cases are labeled in the UI:

- Learned absolute positions predate the Transformer. The timeline uses ConvS2S (8 May 2017) as the canonical modern sequence-model source, while *Attention Is All You Need* later compared learned and sinusoidal positions.
- NTK-aware RoPE scaling began as a community result rather than a paper. It is dated to bloc97's original LocalLLaMA post, not to a later library implementation.

## Primary sources for dates and claims

| Date | Mechanism | Primary source |
|---|---|---|
| 2017-05-08 | Learned absolute positions | [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122) |
| 2017-06-12 | Scaled dot-product attention and sinusoidal positions | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| 2019-04-23 | Sparse Transformer | [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) |
| 2019-11-06 | Multi-query attention | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) |
| 2020-04-10 | Sliding-window + global attention | [Longformer](https://arxiv.org/abs/2004.05150) |
| 2020-06-29 | Linear attention | [Transformers are RNNs](https://arxiv.org/abs/2006.16236) |
| 2020-03-12 | Content-routed sparse/top-k attention | [Routing Transformer](https://arxiv.org/abs/2003.05997) |
| 2021-02-22 | Modern delta-rule linear attention | [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) |
| 2021-04-20 | RoPE | [RoFormer](https://arxiv.org/abs/2104.09864) |
| 2021-08-27 | ALiBi | [Train Short, Test Long](https://arxiv.org/abs/2108.12409) |
| 2022-05-27 | FlashAttention | [FlashAttention](https://arxiv.org/abs/2205.14135) |
| 2023-05-22 | GQA | [Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) |
| 2023-06-28 | NTK-aware RoPE scaling | [Original bloc97 LocalLLaMA post](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) |
| 2023-08-31 | YaRN | [YaRN](https://arxiv.org/abs/2309.00071) |
| 2023-09-29 | Attention sinks / StreamingLLM | [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) |
| 2024-05-07 | MLA | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) |
| 2024-06-10 | Parallel DeltaNet | [Parallelizing Linear Transformers with the Delta Rule](https://arxiv.org/abs/2406.06484) |
| 2024-12-09 | Gated DeltaNet | [Gated Delta Networks](https://arxiv.org/abs/2412.06464) |
| 2025-02-16 | Native Sparse Attention | [NSA](https://arxiv.org/abs/2502.11089) |
| 2025-09-29 | DeepSeek Sparse Attention | [Official DeepSeek-V3.2-Exp release](https://api-docs.deepseek.com/news/news250929/) and [official repository](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) |
| 2025-12-13 | DroPE | [Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings](https://arxiv.org/abs/2512.12167) |

## Accuracy notes

- FlashAttention reduces memory traffic and avoids materializing the quadratic attention matrix; it does **not** remove dense attention's quadratic arithmetic.
- Attention sinks enable stable processing of an unbounded stream with a fixed cache; they do **not** preserve arbitrary old content.
- MLA compresses the KV cache; it does **not** make dense attention subquadratic.
- DeepSeek's NSA (February 2025) and DSA (September 2025) are distinct. NSA has compression, selection, and sliding-window branches. DSA uses a learned lightning indexer for token-level top-k selection over MLA cache entries.
- “Delta rule” has older roots in neural learning rules. The timeline dates its modern fast-weight/linear-attention formulation and separately dates the later parallel DeltaNet algorithm.
- DroPE's public date is December 2025. Any attribution to 2024 is incorrect.

## Before submission

After pushing `S8/`, include both the deployed URL and [`github.com/midhaworks/ERAV5/tree/main/S8`](https://github.com/midhaworks/ERAV5/tree/main/S8) in the submission.
