# Session 6 — Assignment 16: Training Data Execution System

This is a dependency-free, deterministic miniature of the full V5 training-data path:

`documents → tokenized shards → manifests → mixture → packing → batches → training → ledgers → checkpoint → crash → resume → replay → audit`

## Demonstration data

The system uses a real but intentionally tiny synthetic corpus defined in `source_documents()` in `tdes.py`:

- 9 training documents: 3 general-text, 2 Python/code, 1 agentic tool-use trajectory, and 3 protected-language instruction/response records;
- 1 validation document; and
- 1 evaluation document.

The agentic record contains system and user context, a model-generated tool call, an observed tool result, and a final model answer. Its loss mask supervises the tool call and final answer while masking the system prompt, user request, and tool result. The protected records contain simple Hindi, Telugu, and Sindhi prompts with romanized responses. The validation and evaluation records are deliberately present so the firewall can prove that they never enter a loss-bearing batch. On every run, all 11 source documents are encoded into actual token arrays and written as content-addressed JSONL shards under `submission_artifacts/shards/`. The training worker consumes only the three training shards; the tiny bigram model computes token-level losses and updates its learned transition counts from those tokens.

## Run it

From the repository root:

```bash
python3 S6/run_demo.py
```

The command deletes and regenerates only `S6/submission_artifacts/`. A worker trains through checkpoint 5 and is then terminated abruptly with exit code 86 before batch 6 is consumed. The parent observes that process death and launches a fresh recovery worker, which resumes from durable state, replays an earlier interval, forks an older checkpoint, audits the result, and exits non-zero if any invariant fails.

Run the tests with:

```bash
python3 -m unittest discover -s S6/tests -v
```

## Design

- **Immutable input:** documents are encoded with a frozen byte tokenizer. Every JSONL shard has its SHA-256 in its filename; manifests bind content, tokenizer, split, lane and token counts.
- **Firewalls:** the packer itself rejects any record not marked `train`. The audit also scans every consumed shard, while the demo separately records rejected validation and evaluation requests.
- **Deterministic planning:** curriculum stages compile weighted lanes into exact slot allocations. Protected data has a hard 20% floor.
- **Packing:** causal, prompt/response, and agentic tool-use records use distinct loss policies. For agentic data, model actions and final answers bear loss while environment observations do not. Two documents may share a sequence, but causal attention cannot cross document segments, position IDs reset, and padding is invisible and lossless.
- **OPUS:** candidate decisions include acceptance, validation-regression rejection, uncertainty deferral, and an accepted proposal whose protected share is overridden to its floor.
- **Training trace:** a real online Laplace-smoothed bigram model computes token losses and updates counts. The learning ledger links each token-level trace to sample hashes, source spans and the corresponding consumption entry, and records model-state hashes before and after each update.
- **Durability:** consumption and learning ledgers are append-only SHA-256 chains. Checkpoints bind model/packer state to exact ledger offsets and heads, plus tokenizer and schedule hashes.
- **Recovery and replay:** the crash happens after checkpoint 5 but before batch 6 is consumed. Resume reconstructs batch 6 exactly. Replay regenerates batches 2–5 from the initial state and checks batch IDs, token spans and hashes.
- **Forking:** a branch starts from checkpoint 2 with isolated ledgers and records parent/child checkpoint lineage.
- **Evidence:** all PASS/FAIL values are computed from generated artifacts. `evidence.json` also includes a checksum of its generated requirement map.

The generated output is under `S6/submission_artifacts/` and includes the requested log, evidence files, manifests, ledgers, checkpoints and performance report, plus content-addressed shards and detailed audit reports.
