# External evaluation preparation

Source: [official HumanEval](https://github.com/openai/human-eval), MIT license retained in
`upstream/LICENSE`, pinned revision `6d43fb980f9fee3c892a914eda09951f772ad10d`.
Downloaded source contains 164 tasks; selection is the first eight
numeric IDs, without filtering on model performance. Tasks, unchanged official tests,
canonical solutions and protocol are hash-frozen. Canonical solutions and tests are
evaluation-side only, never model prompt material. The loader rejects training use.

**Model inference: NOT RUN. Score: none. Ready: False.**
The current host rejects the requested data-segment cap even in a trusted process
outside the child sandbox. The candidate evaluator fails closed; the memory guard is
not removed to manufacture a benchmark result. Reference execution is also not run.

| Measured OS-isolation probe | Result |
|---|---|
| outside_read_blocked | PASS |
| outside_write_blocked | PASS |
| workspace_write_blocked | PASS |
| network_blocked | PASS |
| fork_blocked | PASS |
| environment_sanitized | PASS |
| memory_guard | FAIL |

The probes bypass candidate AST gates on purpose and use fake sentinel files, a local
listener, and a fake environment variable. They demonstrate these tested restrictions,
not a complete security audit. Metadata and trusted runtime-library reads are allowed;
user-file contents, writes, network, and forking were blocked in these probes. Root-directory
lookup was needed for interpreter startup; it does not permit reading descendant files.
Historical startup diagnostic results are retained separately.

Required next step: a compatible memory-limited execution backend, such as a separately
approved container/Linux executor. Candidate generation can remain on the M2, while
candidate tests run in that isolated backend. No Docker installation or external-server
coordination has been performed without direction.

HumanEval is an established 2021 benchmark, not latest SOTA or guaranteed uncontaminated
evaluation. This eight-task subset is not full HumanEval, a production gate, or an agentic
repository benchmark. Some underlying primitives overlap our local tasks; externally
authored does not mean semantically disjoint or unseen during pretraining.
