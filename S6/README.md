# Session 6 — Assignment 16: Training Data Execution System

This is a dependency-free, deterministic miniature of the full V5 training-data path:

`documents → tokenized shards → manifests → mixture → packing → batches → training → ledgers → checkpoint → crash → resume → replay → audit`

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
- **Packing:** causal and prompt/response records use distinct loss policies. Two documents may share a sequence, but causal attention cannot cross document segments, position IDs reset, and padding is invisible and lossless.
- **OPUS:** candidate decisions include acceptance, validation-regression rejection, uncertainty deferral, and an accepted proposal whose protected share is overridden to its floor.
- **Training trace:** a real online Laplace-smoothed bigram model computes token losses and updates counts. The learning ledger links each token-level trace to sample hashes, source spans and the corresponding consumption entry, and records model-state hashes before and after each update.
- **Durability:** consumption and learning ledgers are append-only SHA-256 chains. Checkpoints bind model/packer state to exact ledger offsets and heads, plus tokenizer and schedule hashes.
- **Recovery and replay:** the crash happens after checkpoint 5 but before batch 6 is consumed. Resume reconstructs batch 6 exactly. Replay regenerates batches 2–5 from the initial state and checks batch IDs, token spans and hashes.
- **Forking:** a branch starts from checkpoint 2 with isolated ledgers and records parent/child checkpoint lineage.
- **Evidence:** all PASS/FAIL values are computed from generated artifacts. `evidence.json` also includes a checksum of its generated requirement map.

The generated output is under `S6/submission_artifacts/` and includes the requested log, evidence files, manifests, ledgers, checkpoints and performance report, plus content-addressed shards and detailed audit reports.
