import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tdes import (  # noqa: E402
    BATCH_SIZE,
    SEQ_LEN,
    FrozenByteTokenizer,
    HashedLedger,
    TOKENIZER_SPEC,
    build_batch,
    build_shards,
    compile_schedule,
    digest,
    firewall,
    initial_packer_state,
    pack_sample,
    read_jsonl,
    run_demo,
    validate_manifests,
)


class ExecutionSystemTests(unittest.TestCase):
    def test_content_addressed_shards_and_frozen_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            tokenizer = FrozenByteTokenizer(TOKENIZER_SPEC)
            manifests, _ = build_shards(base, tokenizer)
            validate_manifests(base, tokenizer, manifests)
            for manifest in manifests:
                self.assertIn(manifest["content_hash"][:16], manifest["path"])
                self.assertEqual(tokenizer.hash, manifest["tokenizer_hash"])

    def test_firewall_blocks_validation_and_eval(self):
        for split in ("validation", "eval"):
            with self.assertRaises(PermissionError):
                firewall({"split": split, "shard_id": "sealed"})

    def test_packer_enforces_firewall_at_point_of_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, records = build_shards(Path(tmp), FrozenByteTokenizer(TOKENIZER_SPEC))
            poisoned = {lane: [dict(x) for x in lane_records] for lane, lane_records in records.items()}
            poisoned["general"][0]["split"] = "eval"
            with self.assertRaises(PermissionError):
                pack_sample("general", poisoned, initial_packer_state())

    def test_packing_masks_attention_and_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, records = build_shards(Path(tmp), FrozenByteTokenizer(TOKENIZER_SPEC))
            batch = build_batch(0, compile_schedule(), records, initial_packer_state())
            self.assertEqual(BATCH_SIZE, len(batch["samples"]))
            for sample in batch["samples"]:
                self.assertEqual(SEQ_LEN, len(sample["tokens"]))
                lane_records = {x["document_id"]: x for x in records[sample["lane"]]}
                for span in sample["source_spans"]:
                    self.assertEqual(len(lane_records[span["document_id"]]["tokens"]), span["source_token_end"])
                for i in range(SEQ_LEN):
                    for j in range(SEQ_LEN):
                        if sample["attention_mask"][i][j]:
                            self.assertLessEqual(j, i)
                            self.assertEqual(sample["segment_ids"][i], sample["segment_ids"][j])

    def test_agentic_policy_trains_actions_and_answers_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer = FrozenByteTokenizer(TOKENIZER_SPEC)
            _, records = build_shards(Path(tmp), tokenizer)
            agentic = next(x for x in records["code"] if x["kind"] == "agentic")
            self.assertEqual("agentic_action_and_answer", agentic["packing_policy"])
            self.assertEqual(2, len(agentic["supervised_regions"]))
            expected = [0] * len(agentic["tokens"])
            for start, end in agentic["supervised_regions"]:
                expected[start:end] = [1] * (end - start)
            self.assertEqual(expected, agentic["loss_mask"])
            # The tool result is context, not a model-generated training target.
            result_token = tokenizer.encode("5")[0]
            masked_fives = [i for i, token in enumerate(agentic["tokens"]) if token == result_token and not agentic["loss_mask"][i]]
            supervised_fives = [i for i, token in enumerate(agentic["tokens"]) if token == result_token and agentic["loss_mask"][i]]
            self.assertTrue(masked_fives)
            self.assertTrue(supervised_fives)

    def test_batch_reconstruction_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, records = build_shards(Path(tmp), FrozenByteTokenizer(TOKENIZER_SPEC))
            schedule = compile_schedule()
            first = build_batch(0, schedule, records, initial_packer_state())
            second = build_batch(0, schedule, records, initial_packer_state())
            self.assertEqual(first["batch_hash"], second["batch_hash"])
            self.assertEqual(first["samples"][0]["source_spans"], second["samples"][0]["source_spans"])

    def test_ledger_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            ledger = HashedLedger.open(path)
            ledger.append({"event": "one"})
            entry = read_jsonl(path)[0]
            entry["event"] = "tampered"
            path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                HashedLedger.open(path)

    def test_complete_demo_generates_computed_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_demo(Path(tmp) / "submission_artifacts")
            evidence = json.loads((Path(result["artifacts"]) / "evidence.json").read_text())
            self.assertTrue(evidence["overall_pass"])
            self.assertEqual(digest(evidence["requirements"]), evidence["bundle_hash"])
            self.assertTrue(all(x["passed"] for x in evidence["requirements"].values()))
            crash = json.loads((Path(result["artifacts"]) / "reports/crash_expectation.json").read_text())
            self.assertEqual(86, crash["observed_exit_code"])


if __name__ == "__main__":
    unittest.main()
