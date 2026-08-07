"""Deterministic miniature Training Data Execution System.

The implementation is intentionally dependency-free.  It is small enough to audit,
but every artifact is produced by the same code path used by the demonstration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "submission_artifacts"
SEQ_LEN = 48
BATCH_SIZE = 2
SPECIAL = {"pad": 0, "bos": 1, "eos": 2, "sep": 3, "unk": 4}
TOKENIZER_SPEC = {
    "name": "frozen-byte-v1",
    "version": 1,
    "normalization": "none",
    "special_tokens": SPECIAL,
    "byte_offset": 5,
    "vocab_size": 261,
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value: Any) -> str:
    data = value if isinstance(value, bytes) else canonical(value)
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class FrozenByteTokenizer:
    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.hash = digest(spec)

    def encode(self, text: str) -> list[int]:
        return [byte + self.spec["byte_offset"] for byte in text.encode("utf-8")]


class EventLog:
    def __init__(self, path: Path, append: bool = False):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not append:
            path.write_text("", encoding="utf-8")

    def event(self, message: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
        print(message, flush=True)

    def passed(self, name: str, detail: str = "") -> None:
        self.event(f"[PASS] {name}" + (f" | {detail}" if detail else ""))


@dataclass
class HashedLedger:
    path: Path
    offset: int = 0
    head: str = "0" * 64

    @classmethod
    def open(cls, path: Path) -> "HashedLedger":
        entries = read_jsonl(path)
        ledger = cls(path)
        for index, entry in enumerate(entries):
            payload = {k: v for k, v in entry.items() if k != "entry_hash"}
            if entry["offset"] != index or entry["prev_hash"] != ledger.head:
                raise ValueError(f"broken ledger chain at {path}:{index}")
            if digest(payload) != entry["entry_hash"]:
                raise ValueError(f"bad ledger hash at {path}:{index}")
            ledger.head = entry["entry_hash"]
            ledger.offset += 1
        return ledger

    def append(self, payload: dict[str, Any]) -> dict[str, Any]:
        entry = {"offset": self.offset, "prev_hash": self.head, **payload}
        entry["entry_hash"] = digest(entry)
        append_jsonl(self.path, entry)
        self.offset += 1
        self.head = entry["entry_hash"]
        return entry


def source_documents() -> list[dict[str, str]]:
    return [
        {"id": "g0", "split": "train", "lane": "general", "kind": "causal", "text": "Rain feeds rivers and fields."},
        {"id": "g1", "split": "train", "lane": "general", "kind": "causal", "text": "A compass points toward north."},
        {"id": "g2", "split": "train", "lane": "general", "kind": "causal", "text": "Small proofs make systems clear."},
        {"id": "c0", "split": "train", "lane": "code", "kind": "causal", "text": "def add(a,b): return a+b"},
        {"id": "c1", "split": "train", "lane": "code", "kind": "causal", "text": "items = sorted(set(items))"},
        {"id": "a0", "split": "train", "lane": "code", "kind": "agentic", "system": "tool", "user": "2+3?", "tool_call": "add(2,3)", "tool_result": "5", "response": "5"},
        {"id": "p0", "split": "train", "lane": "protected", "kind": "instruction", "prompt": "Translate sun to Hindi:", "response": "surya"},
        {"id": "p1", "split": "train", "lane": "protected", "kind": "instruction", "prompt": "Say water in Telugu:", "response": "neeru"},
        {"id": "p2", "split": "train", "lane": "protected", "kind": "instruction", "prompt": "Sindhi greeting:", "response": "salaam"},
        {"id": "v0", "split": "validation", "lane": "general", "kind": "causal", "text": "validation is never training"},
        {"id": "e0", "split": "eval", "lane": "general", "kind": "causal", "text": "evaluation remains sealed"},
    ]


def encode_document(doc: dict[str, str], tokenizer: FrozenByteTokenizer) -> dict[str, Any]:
    supervised_regions: list[list[int]]
    if doc["kind"] == "agentic":
        system = tokenizer.encode(doc["system"])
        user = tokenizer.encode(doc["user"])
        tool_call = tokenizer.encode(doc["tool_call"])
        tool_result = tokenizer.encode(doc["tool_result"])
        response = tokenizer.encode(doc["response"])
        prefix = [SPECIAL["bos"], *system, SPECIAL["sep"], *user, SPECIAL["sep"]]
        middle = [SPECIAL["sep"], *tool_result, SPECIAL["sep"]]
        tokens = [*prefix, *tool_call, *middle, *response, SPECIAL["eos"]]
        call_start = len(prefix)
        response_start = len(prefix) + len(tool_call) + len(middle)
        supervised_regions = [[call_start, call_start + len(tool_call)], [response_start, len(tokens)]]
        loss = [0] * len(tokens)
        for start, end in supervised_regions:
            loss[start:end] = [1] * (end - start)
        policy = "agentic_action_and_answer"
    elif doc["kind"] == "instruction":
        prompt = tokenizer.encode(doc["prompt"])
        response = tokenizer.encode(doc["response"])
        tokens = [SPECIAL["bos"], *prompt, SPECIAL["sep"], *response, SPECIAL["eos"]]
        # Response and EOS bear loss; prompt and separator do not.
        loss = [0] * (len(prompt) + 2) + [1] * (len(response) + 1)
        supervised_regions = [[len(prompt) + 2, len(tokens)]]
        policy = "prompt_response_masked"
    else:
        body = tokenizer.encode(doc["text"])
        tokens = [SPECIAL["bos"], *body, SPECIAL["eos"]]
        loss = [0] + [1] * (len(body) + 1)
        supervised_regions = [[1, len(tokens)]]
        policy = "causal_document"
    return {
        "document_id": doc["id"], "split": doc["split"], "lane": doc["lane"],
        "kind": doc["kind"], "packing_policy": policy, "tokens": tokens,
        "loss_mask": loss, "supervised_regions": supervised_regions, "content_hash": digest(doc),
    }


def build_shards(base: Path, tokenizer: FrozenByteTokenizer) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    shards_dir = base / "shards"
    manifest_dir = base / "manifests"
    write_json(manifest_dir / "tokenizer.json", TOKENIZER_SPEC)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for doc in source_documents():
        grouped[(doc["split"], doc["lane"])].append(encode_document(doc, tokenizer))

    manifests, train = [], defaultdict(list)
    for (split, lane), records in sorted(grouped.items()):
        payload = b"".join(canonical(record) + b"\n" for record in records)
        shard_hash = digest(payload)
        name = f"{split}-{lane}-{shard_hash[:16]}.jsonl"
        path = shards_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        manifest = {
            "schema_version": 1, "shard_id": name.removesuffix(".jsonl"), "path": f"shards/{name}",
            "split": split, "lane": lane, "record_count": len(records),
            "token_count": sum(len(x["tokens"]) for x in records), "content_hash": shard_hash,
            "tokenizer_hash": tokenizer.hash, "immutable": True,
        }
        write_json(manifest_dir / f"{name}.manifest.json", manifest)
        manifests.append(manifest)
        if split == "train":
            for record in records:
                train[lane].append({**record, "shard_id": manifest["shard_id"]})
    write_json(manifest_dir / "index.json", {"tokenizer_hash": tokenizer.hash, "shards": manifests})
    return manifests, dict(train)


def validate_manifests(base: Path, tokenizer: FrozenByteTokenizer, manifests: list[dict[str, Any]]) -> None:
    if digest(read_json(base / "manifests/tokenizer.json")) != tokenizer.hash:
        raise AssertionError("tokenizer specification changed")
    for manifest in manifests:
        payload = (base / manifest["path"]).read_bytes()
        if digest(payload) != manifest["content_hash"] or manifest["tokenizer_hash"] != tokenizer.hash:
            raise AssertionError(f"invalid manifest {manifest['shard_id']}")


def firewall(manifest: dict[str, Any]) -> None:
    if manifest["split"] != "train":
        raise PermissionError(f"{manifest['split']} shard blocked: {manifest['shard_id']}")


def opus_decisions(path: Path) -> list[dict[str, Any]]:
    candidates = [
        {"id": "opus-accept-code", "proposal": {"code": 0.50}, "score_delta": 0.04, "uncertainty": 0.01},
        {"id": "opus-reject-regression", "proposal": {"general": 0.05}, "score_delta": -0.08, "uncertainty": 0.01},
        {"id": "opus-defer-uncertain", "proposal": {"code": 0.60}, "score_delta": 0.02, "uncertainty": 0.09},
        {"id": "opus-floor-override", "proposal": {"protected": 0.05}, "score_delta": 0.03, "uncertainty": 0.01},
    ]
    decisions = []
    floor = 0.20
    for candidate in candidates:
        applied = dict(candidate["proposal"])
        override = None
        if applied.get("protected", floor) < floor:
            override = {"field": "protected", "proposed": applied["protected"], "applied": floor, "reason": "protected_floor"}
            applied["protected"] = floor
        if candidate["score_delta"] < 0:
            decision, reason = "reject", "validation_regression"
        elif candidate["uncertainty"] > 0.05:
            decision, reason = "defer", "insufficient_confidence"
        else:
            decision, reason = "accept", "positive_validation_delta"
        record = {**candidate, "decision": decision, "reason": reason, "applied": applied, "override": override}
        append_jsonl(path, record)
        decisions.append(record)
    return decisions


def allocate_slots(weights: dict[str, float], slots: int, floor: dict[str, float]) -> dict[str, int]:
    adjusted = dict(weights)
    for lane, value in floor.items():
        adjusted[lane] = max(adjusted.get(lane, 0), value)
    total = sum(adjusted.values())
    quotas = {lane: adjusted[lane] / total * slots for lane in adjusted}
    counts = {lane: int(math.floor(value)) for lane, value in quotas.items()}
    for lane in sorted(quotas, key=lambda x: (quotas[x] - counts[x], x), reverse=True)[:slots - sum(counts.values())]:
        counts[lane] += 1
    for lane, minimum in floor.items():
        required = math.ceil(minimum * slots)
        while counts[lane] < required:
            donor = max((x for x in counts if x != lane), key=lambda x: counts[x] - quotas[x])
            counts[donor] -= 1
            counts[lane] += 1
    return counts


def interleave(counts: dict[str, int]) -> list[str]:
    remaining = dict(counts)
    used = Counter()
    total = sum(counts.values())
    sequence = []
    for position in range(total):
        choices = [lane for lane, count in remaining.items() if count]
        lane = max(choices, key=lambda x: ((position + 1) * counts[x] / total - used[x], x))
        sequence.append(lane)
        remaining[lane] -= 1
        used[lane] += 1
    return sequence


def compile_schedule(decisions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    stages = [
        {"name": "foundation", "start": 0, "batches": 6, "weights": {"general": .50, "code": .25, "protected": .25}, "floors": {"protected": .20}},
        {"name": "specialize", "start": 6, "batches": 4, "weights": {"general": .25, "code": .25, "protected": .25}, "floors": {"protected": .20}},
    ]
    # Only accepted OPUS records can mutate the next curriculum stage.  The
    # protected-floor candidate carries its already-overridden applied value.
    for decision in decisions or []:
        if decision["decision"] == "accept":
            stages[1]["weights"].update(decision["applied"])
    lanes = []
    for stage in stages:
        counts = allocate_slots(stage["weights"], stage["batches"] * BATCH_SIZE, stage["floors"])
        stage["planned_slots"] = counts
        stage["lane_sequence"] = interleave(counts)
        lanes.extend(stage["lane_sequence"])
    batches = []
    for index in range(sum(x["batches"] for x in stages)):
        stage = next(x for x in stages if x["start"] <= index < x["start"] + x["batches"])
        batches.append({"batch_index": index, "stage": stage["name"], "lanes": lanes[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]})
    accepted_ids = [x["id"] for x in (decisions or []) if x["decision"] == "accept"]
    schedule = {
        "schema_version": 1, "batch_size": BATCH_SIZE, "sequence_length": SEQ_LEN,
        "stages": stages, "batches": batches,
        "opus_decisions_applied": accepted_ids,
        "derivation": "stage lane slots are largest-remainder allocations after protected floors",
    }
    schedule["schedule_hash"] = digest(schedule)
    return schedule


def initial_packer_state() -> dict[str, int]:
    return {"general": 0, "code": 0, "protected": 0}


def pack_sample(lane: str, records: dict[str, list[dict[str, Any]]], state: dict[str, int]) -> dict[str, Any]:
    tokens: list[int] = []
    loss_mask: list[int] = []
    position_ids: list[int] = []
    segment_ids: list[int] = []
    spans = []
    # Two complete documents per sample demonstrate real packing and isolation.
    for segment in range(2):
        record = records[lane][state[lane] % len(records[lane])]
        if record["split"] != "train":
            raise PermissionError(f"non-training record blocked inside packer: {record['document_id']}")
        available = SEQ_LEN - len(tokens)
        # Never silently drop a document tail.  If another complete document does
        # not fit, leave padding and consume that document in the next sample.
        if tokens and len(record["tokens"]) > available:
            break
        if len(record["tokens"]) > available:
            raise AssertionError(f"document {record['document_id']} exceeds sequence length")
        state[lane] += 1
        take = len(record["tokens"])
        start = len(tokens)
        tokens.extend(record["tokens"][:take])
        loss_mask.extend(record["loss_mask"][:take])
        position_ids.extend(range(take))
        segment_ids.extend([segment] * take)
        spans.append({
            "shard_id": record["shard_id"], "document_id": record["document_id"],
            "source_token_start": 0, "source_token_end": take,
            "packed_start": start, "packed_end": start + take,
            "content_hash": record["content_hash"], "packing_policy": record["packing_policy"],
        })
        if len(tokens) == SEQ_LEN:
            break
    real = len(tokens)
    padding = SEQ_LEN - real
    tokens.extend([SPECIAL["pad"]] * padding)
    loss_mask.extend([0] * padding)
    position_ids.extend([0] * padding)
    segment_ids.extend([-1] * padding)
    attention_mask = [
        [1 if i < real and j <= i and segment_ids[i] == segment_ids[j] else 0 for j in range(SEQ_LEN)]
        for i in range(SEQ_LEN)
    ]
    sample = {
        "lane": lane, "tokens": tokens, "loss_mask": loss_mask,
        "position_ids": position_ids, "segment_ids": segment_ids,
        "attention_mask": attention_mask, "source_spans": spans,
        "non_padding_tokens": real, "loss_bearing_tokens": sum(loss_mask),
    }
    validate_sample(sample)
    sample["sample_hash"] = digest(sample)
    return sample


def validate_sample(sample: dict[str, Any]) -> None:
    fields = ("tokens", "loss_mask", "position_ids", "segment_ids", "attention_mask")
    if any(len(sample[field]) != SEQ_LEN for field in fields):
        raise AssertionError("packed tensor has wrong length")
    for i in range(SEQ_LEN):
        segment = sample["segment_ids"][i]
        if segment < 0:
            if sample["loss_mask"][i] or any(sample["attention_mask"][i]):
                raise AssertionError("padding can neither attend nor bear loss")
            continue
        expected_pos = sum(1 for x in sample["segment_ids"][:i] if x == segment)
        if sample["position_ids"][i] != expected_pos:
            raise AssertionError("position id did not reset at packed boundary")
        for j, visible in enumerate(sample["attention_mask"][i]):
            if visible != int(j <= i and sample["segment_ids"][j] == segment):
                raise AssertionError("attention crossed a document boundary")
    for span in sample["source_spans"]:
        if span["packing_policy"] == "prompt_response_masked":
            start, end = span["packed_start"], span["packed_end"]
            segment_tokens = sample["tokens"][start:end]
            if SPECIAL["sep"] in segment_tokens:
                sep = start + segment_tokens.index(SPECIAL["sep"])
                if any(sample["loss_mask"][start:sep + 1]):
                    raise AssertionError("instruction prompt bears loss")


def build_batch(index: int, schedule: dict[str, Any], records: dict[str, list[dict[str, Any]]], state: dict[str, int]) -> dict[str, Any]:
    plan = schedule["batches"][index]
    samples = [pack_sample(lane, records, state) for lane in plan["lanes"]]
    batch = {
        "batch_index": index, "batch_id": f"batch-{index:04d}", "stage": plan["stage"],
        "schedule_hash": schedule["schedule_hash"], "samples": samples,
        "non_padding_tokens": sum(x["non_padding_tokens"] for x in samples),
        "loss_bearing_tokens": sum(x["loss_bearing_tokens"] for x in samples),
    }
    batch["batch_hash"] = digest(batch)
    return batch


class TinyBigramModel:
    def __init__(self, counts: dict[str, dict[str, int]] | None = None):
        self.counts = {prev: dict(nxt) for prev, nxt in (counts or {}).items()}

    def train(self, batch: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
        losses, traces = [], []
        vocab = TOKENIZER_SPEC["vocab_size"]
        for sample_index, sample in enumerate(batch["samples"]):
            sample_losses = []
            for token_index in range(1, SEQ_LEN):
                if not sample["loss_mask"][token_index]:
                    continue
                prev, target = str(sample["tokens"][token_index - 1]), str(sample["tokens"][token_index])
                row = self.counts.setdefault(prev, {})
                total = sum(row.values())
                probability = (row.get(target, 0) + 1) / (total + vocab)
                loss = -math.log(probability)
                row[target] = row.get(target, 0) + 1
                losses.append(loss)
                sample_losses.append({"token_index": token_index, "target": int(target), "loss": round(loss, 8)})
            traces.append({
                "sample_index": sample_index, "sample_hash": sample["sample_hash"], "lane": sample["lane"],
                "source_spans": sample["source_spans"], "token_losses": sample_losses,
                "mean_loss": round(sum(x["loss"] for x in sample_losses) / max(1, len(sample_losses)), 8),
            })
        return sum(losses) / max(1, len(losses)), traces

    def state(self) -> dict[str, dict[str, int]]:
        return self.counts


def consumption_payload(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": "batch_consumed", "batch_index": batch["batch_index"], "batch_id": batch["batch_id"],
        "batch_hash": batch["batch_hash"], "stage": batch["stage"],
        "sample_hashes": [x["sample_hash"] for x in batch["samples"]],
        "lanes": [x["lane"] for x in batch["samples"]],
        "token_spans": [x["source_spans"] for x in batch["samples"]],
        "non_padding_tokens": batch["non_padding_tokens"], "loss_bearing_tokens": batch["loss_bearing_tokens"],
    }


def save_checkpoint(path: Path, model: TinyBigramModel, packer_state: dict[str, int], next_batch: int,
                    consumption: HashedLedger, learning: HashedLedger, schedule: dict[str, Any], tokenizer_hash: str,
                    parent: str | None = None) -> dict[str, Any]:
    state = {
        "schema_version": 1, "next_batch_index": next_batch, "model": model.state(),
        "packer_state": dict(packer_state), "schedule_hash": schedule["schedule_hash"],
        "tokenizer_hash": tokenizer_hash,
        "ledger_offsets": {"consumption": consumption.offset, "learning": learning.offset},
        "ledger_heads": {"consumption": consumption.head, "learning": learning.head},
        "parent_checkpoint": parent,
    }
    wrapper = {"checkpoint_hash": digest(state), "state": state}
    write_json(path, wrapper)
    return wrapper


def load_checkpoint(path: Path, consumption: HashedLedger, learning: HashedLedger,
                    schedule: dict[str, Any], tokenizer_hash: str) -> tuple[TinyBigramModel, dict[str, int], int]:
    wrapper = read_json(path)
    state = wrapper["state"]
    if digest(state) != wrapper["checkpoint_hash"]:
        raise AssertionError("checkpoint checksum failed")
    if state["schedule_hash"] != schedule["schedule_hash"] or state["tokenizer_hash"] != tokenizer_hash:
        raise AssertionError("checkpoint configuration mismatch")
    offsets = state["ledger_offsets"]
    heads = state["ledger_heads"]
    if offsets != {"consumption": consumption.offset, "learning": learning.offset}:
        raise AssertionError("checkpoint ledger offset mismatch")
    if heads != {"consumption": consumption.head, "learning": learning.head}:
        raise AssertionError("checkpoint ledger head mismatch")
    return TinyBigramModel(state["model"]), dict(state["packer_state"]), state["next_batch_index"]


def train_batch(batch: dict[str, Any], model: TinyBigramModel, consumption: HashedLedger,
                learning: HashedLedger) -> float:
    model_before_hash = digest(model.state())
    loss, traces = model.train(batch)
    model_after_hash = digest(model.state())
    consumed = consumption.append(consumption_payload(batch))
    learning.append({
        "event": "learning_update", "batch_index": batch["batch_index"], "batch_id": batch["batch_id"],
        "batch_hash": batch["batch_hash"], "consumption_entry_hash": consumed["entry_hash"],
        "mean_loss": round(loss, 8), "sample_traces": traces,
        "model_before_hash": model_before_hash, "model_after_hash": model_after_hash,
        "updated_token_count": sum(len(x["token_losses"]) for x in traces),
    })
    return loss


def reconstruct_to(stop: int, schedule: dict[str, Any], records: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    state = initial_packer_state()
    return [build_batch(index, schedule, records, state) for index in range(stop)]


def verify_mixture(schedule: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for stage in schedule["stages"]:
        start, stop = stage["start"], stage["start"] + stage["batches"]
        actual = Counter(lane for entry in entries if start <= entry["batch_index"] < stop for lane in entry["lanes"])
        planned = Counter(stage["planned_slots"])
        slots = sum(actual.values())
        protected_share = actual.get("protected", 0) / max(1, slots)
        floor = stage["floors"].get("protected", 0)
        result[stage["name"]] = {
            "planned": dict(planned), "actual": dict(actual),
            "protected_floor": floor, "actual_protected_share": protected_share,
            "floor_met": protected_share >= floor, "match": actual == planned and protected_share >= floor,
        }
    return result


def write_evidence(base: Path, checks: dict[str, dict[str, Any]]) -> None:
    bundle = {
        "generated_by": "S6.tdes.run_demo", "schema_version": 1,
        "overall_pass": all(item["passed"] for item in checks.values()), "requirements": checks,
        "bundle_hash": digest(checks),
    }
    write_json(base / "evidence.json", bundle)
    labels = {
        "tokenizer_integrity": "Tokenizer integrity", "evaluation_firewall": "Evaluation firewall",
        "packing_correctness": "Packing correctness", "mixture_compliance": "Mixture compliance",
        "opus_audit_trail": "OPUS audit trail", "crash_recovery": "Crash recovery",
        "replay": "Replay", "learning_trace": "Learning trace", "throughput": "Throughput",
    }
    lines = ["# Execution Evidence", "", "| REQUIREMENT | RESULT | EVIDENCE |", "|---|---|---|"]
    for key, label in labels.items():
        item = checks[key]
        lines.append(f"| {label} | {'PASS' if item['passed'] else 'FAIL'} | {item['evidence']} |")
    lines += ["", f"Overall result: **{'PASS' if bundle['overall_pass'] else 'FAIL'}**", ""]
    (base / "evidence.md").write_text("\n".join(lines), encoding="utf-8")


def run_crash_phase(artifacts: Path) -> None:
    """Build and train through batch 5; caller terminates this worker abruptly."""
    if artifacts.exists():
        shutil.rmtree(artifacts)
    for directory in ("manifests", "ledgers", "checkpoints", "shards", "reports"):
        (artifacts / directory).mkdir(parents=True, exist_ok=True)
    log = EventLog(artifacts / "run.log")
    started = time.perf_counter()
    tokenizer = FrozenByteTokenizer(TOKENIZER_SPEC)
    manifests, records = build_shards(artifacts, tokenizer)
    log.event(f"shards created | count={len(manifests)}")
    validate_manifests(artifacts, tokenizer, manifests)
    log.event("manifests validated")
    log.passed("tokenizer_hash_verified", tokenizer.hash)

    blocked = []
    for split in ("validation", "eval"):
        candidate = next(x for x in manifests if x["split"] == split)
        try:
            firewall(candidate)
        except PermissionError as error:
            blocked.append({"split": split, "shard_id": candidate["shard_id"], "reason": str(error)})
            log.event(f"{split} data blocked | {candidate['shard_id']}")
    write_json(artifacts / "reports/firewall.json", {"blocked_attempts": blocked})
    log.passed("eval_shard_blocked", f"blocked={len(blocked)}")

    decisions = opus_decisions(artifacts / "ledgers/opus.jsonl")
    log.event("OPUS decisions recorded | accept=2 reject=1 defer=1 protected_floor_override=1")
    schedule = compile_schedule(decisions)
    write_json(artifacts / "manifests/mixture_schedule.json", schedule)
    log.event(f"mixture compiled | schedule_hash={schedule['schedule_hash']}")

    consumption = HashedLedger.open(artifacts / "ledgers/consumption.jsonl")
    learning = HashedLedger.open(artifacts / "ledgers/learning.jsonl")
    model, packer_state = TinyBigramModel(), initial_packer_state()
    for index in range(6):
        batch = build_batch(index, schedule, records, packer_state)
        loss = train_batch(batch, model, consumption, learning)
        log.event(f"batches packed | id={batch['batch_id']} hash={batch['batch_hash']} loss={loss:.6f}")
        if index in (2, 5):
            path = artifacts / f"checkpoints/main-{index:04d}.json"
            save_checkpoint(path, model, packer_state, index + 1, consumption, learning, schedule, tokenizer.hash)
            log.event(f"checkpoint saved | {path.name} ledger_offset={consumption.offset}")
            log.passed("checkpoint_saved", path.name)
    expected_next = build_batch(6, schedule, records, dict(packer_state))
    write_json(artifacts / "reports/crash_expectation.json", {
        "expected_batch_id": expected_next["batch_id"], "expected_batch_hash": expected_next["batch_hash"],
        "crashing_worker_pid": os.getpid(), "planned_exit_code": 86,
        "last_durable_consumption_offset": consumption.offset,
    })
    write_json(artifacts / "reports/runtime_state.json", {"started_perf_counter": started})
    log.event("crash simulated | terminating worker process before batch-0006 consumption")


def run_resume_phase(artifacts: Path) -> dict[str, Any]:
    """Resume in a fresh worker using only durable artifacts, then audit."""
    log = EventLog(artifacts / "run.log", append=True)
    tokenizer = FrozenByteTokenizer(TOKENIZER_SPEC)
    manifests, records = build_shards(artifacts, tokenizer)
    validate_manifests(artifacts, tokenizer, manifests)
    schedule = read_json(artifacts / "manifests/mixture_schedule.json")
    decisions = read_jsonl(artifacts / "ledgers/opus.jsonl")
    consumption = HashedLedger.open(artifacts / "ledgers/consumption.jsonl")
    learning = HashedLedger.open(artifacts / "ledgers/learning.jsonl")
    checkpoint_5 = artifacts / "checkpoints/main-0005.json"
    checkpoint_2 = artifacts / "checkpoints/main-0002.json"
    model, packer_state, next_index = load_checkpoint(checkpoint_5, consumption, learning, schedule, tokenizer.hash)
    log.event(f"run resumed | checkpoint={checkpoint_5.name} next_batch={next_index} process={os.getpid()}")
    expected_next = read_json(artifacts / "reports/crash_expectation.json")
    resumed = build_batch(next_index, schedule, records, packer_state)
    resume_match = (resumed["batch_id"] == expected_next["expected_batch_id"]
                    and resumed["batch_hash"] == expected_next["expected_batch_hash"])
    if not resume_match:
        raise AssertionError("resumed batch differs from expected next batch")
    log.passed("resume_next_batch_matched", f"{resumed['batch_id']} {resumed['batch_hash']}")
    for index in range(next_index, 10):
        batch = resumed if index == next_index else build_batch(index, schedule, records, packer_state)
        loss = train_batch(batch, model, consumption, learning)
        log.event(f"batches packed | id={batch['batch_id']} hash={batch['batch_hash']} loss={loss:.6f}")

    original = {x["batch_index"]: x for x in read_jsonl(artifacts / "ledgers/consumption.jsonl")}
    replayed = reconstruct_to(6, schedule, records)[2:6]
    replay_records = []
    for batch in replayed:
        entry = original[batch["batch_index"]]
        matches = {
            "batch_id": batch["batch_id"] == entry["batch_id"],
            "batch_hash": batch["batch_hash"] == entry["batch_hash"],
            "token_spans": [x["source_spans"] for x in batch["samples"]] == entry["token_spans"],
        }
        replay_records.append({"batch_index": batch["batch_index"], "matches": matches})
    replay_match = all(all(x["matches"].values()) for x in replay_records)
    write_json(artifacts / "reports/replay.json", {"interval": [2, 6], "records": replay_records, "passed": replay_match})
    log.event("historical stream replayed | interval=[2,6)")
    if replay_match:
        log.passed("replay_hash_matched", "batch ids, token spans and hashes identical")

    # Fork from an earlier durable checkpoint into isolated branch ledgers.
    parent_cp = read_json(checkpoint_2)
    fork_consumption_path = artifacts / "ledgers/fork_consumption.jsonl"
    fork_learning_path = artifacts / "ledgers/fork_learning.jsonl"
    # Seed branch ledgers with immutable prefixes referenced by the parent checkpoint.
    for entry in read_jsonl(artifacts / "ledgers/consumption.jsonl")[:3]:
        append_jsonl(fork_consumption_path, entry)
    for entry in read_jsonl(artifacts / "ledgers/learning.jsonl")[:3]:
        append_jsonl(fork_learning_path, entry)
    fork_consumption = HashedLedger.open(fork_consumption_path)
    fork_learning = HashedLedger.open(fork_learning_path)
    fork_model, fork_state, fork_next = load_checkpoint(checkpoint_2, fork_consumption, fork_learning, schedule, tokenizer.hash)
    fork_batch = build_batch(fork_next, schedule, records, fork_state)
    train_batch(fork_batch, fork_model, fork_consumption, fork_learning)
    fork_cp = save_checkpoint(artifacts / "checkpoints/fork-from-0002.json", fork_model, fork_state, fork_next + 1,
                              fork_consumption, fork_learning, schedule, tokenizer.hash, parent_cp["checkpoint_hash"])
    log.event(f"branch forked | parent={parent_cp['checkpoint_hash']} child={fork_cp['checkpoint_hash']}")

    entries = read_jsonl(artifacts / "ledgers/consumption.jsonl")
    mixture = verify_mixture(schedule, entries)
    write_json(artifacts / "reports/mixture_compliance.json", mixture)

    # Derive every packing and throughput numerator from the durable ledger.
    rebuilt = reconstruct_to(len(entries), schedule, records)
    rebuilt_hashes_match = all(batch["batch_hash"] == entry["batch_hash"] for batch, entry in zip(rebuilt, entries))
    record_lengths = {record["document_id"]: len(record["tokens"]) for lane in records.values() for record in lane}
    spans_complete = all(
        span["source_token_start"] == 0 and span["source_token_end"] == record_lengths[span["document_id"]]
        for entry in entries for sample in entry["token_spans"] for span in sample
    )
    packing_report = {
        "validated_batches": len(entries), "sequence_length": SEQ_LEN,
        "policies": sorted({span["packing_policy"] for e in entries for sample in e["token_spans"] for span in sample}),
        "invariants": {
            "all_batches_rebuilt_from_sources": rebuilt_hashes_match,
            "all_source_spans_complete": spans_complete,
            "attention_causal_and_document_isolated": True,
            "positions_reset_per_document": True,
            "prompt_and_padding_loss_masked": True,
        },
        "validation_method": "rebuild_batch calls validate_sample for every reconstructed sample",
    }
    write_json(artifacts / "reports/packing.json", packing_report)

    runtime = read_json(artifacts / "reports/runtime_state.json")
    elapsed = max(time.perf_counter() - runtime["started_perf_counter"], 1e-9)
    capacity = len(entries) * BATCH_SIZE * SEQ_LEN
    total_nonpad = sum(x["non_padding_tokens"] for x in entries)
    total_useful = sum(x["loss_bearing_tokens"] for x in entries)
    elapsed_reported = round(elapsed, 6)
    performance = {
        "elapsed_seconds": elapsed_reported, "batches": len(entries), "capacity_tokens": capacity,
        "non_padding_tokens": total_nonpad, "loss_bearing_tokens": total_useful,
        "packing_utilization": round(total_nonpad / capacity, 6),
        "useful_loss_tokens_per_second": round(total_useful / elapsed_reported, 3),
        "measurement": "wall_clock_across_crash_and_resume_workers",
        "derivation": {
            "capacity_tokens": "consumption entries * batch_size * sequence_length",
            "non_padding_tokens": "sum(consumption.non_padding_tokens)",
            "loss_bearing_tokens": "sum(consumption.loss_bearing_tokens)",
            "packing_utilization": "non_padding_tokens / capacity_tokens",
            "useful_loss_tokens_per_second": "loss_bearing_tokens / elapsed_seconds",
        },
    }
    write_json(artifacts / "performance.json", performance)

    train_shards = {x["shard_id"] for x in manifests if x["split"] == "train"}
    consumed_shards = {span["shard_id"] for e in entries for sample in e["token_spans"] for span in sample}
    firewall_report = read_json(artifacts / "reports/firewall.json")
    firewall_report.update({
        "allowed_train_shards": sorted(train_shards), "consumed_shards": sorted(consumed_shards),
        "non_training_shards_consumed": sorted(consumed_shards - train_shards),
        "all_consumed_shards_are_train": consumed_shards <= train_shards,
        "enforcement": "pack_sample rejects every record whose split is not train",
    })
    write_json(artifacts / "reports/firewall.json", firewall_report)

    opus_ok = ({x["decision"] for x in decisions} == {"accept", "reject", "defer"}
               and any(x["override"] and x["override"]["reason"] == "protected_floor" for x in decisions)
               and set(schedule["opus_decisions_applied"]) == {x["id"] for x in decisions if x["decision"] == "accept"})
    learning_entries = read_jsonl(artifacts / "ledgers/learning.jsonl")
    learning_ok = all(
        x["sample_traces"] and x["consumption_entry_hash"] == entries[x["batch_index"]]["entry_hash"]
        and x["batch_hash"] == entries[x["batch_index"]]["batch_hash"]
        and x["model_before_hash"] != x["model_after_hash"]
        and x["updated_token_count"] == sum(len(s["token_losses"]) for s in x["sample_traces"])
        for x in learning_entries
    )
    crash_process_ok = (expected_next.get("observed_exit_code") == expected_next["planned_exit_code"] == 86
                        and expected_next["crashing_worker_pid"] != os.getpid())
    firewall_ok = ({x["split"] for x in firewall_report["blocked_attempts"]} == {"validation", "eval"}
                   and firewall_report["all_consumed_shards_are_train"])
    performance_ok = (
        performance["capacity_tokens"] == len(entries) * BATCH_SIZE * SEQ_LEN
        and performance["non_padding_tokens"] == sum(x["non_padding_tokens"] for x in entries)
        and performance["loss_bearing_tokens"] == sum(x["loss_bearing_tokens"] for x in entries)
        and abs(performance["packing_utilization"] - total_nonpad / capacity) < 1e-6
        and performance["useful_loss_tokens_per_second"] > 0
    )
    checks = {
        "tokenizer_integrity": {"passed": all(x["tokenizer_hash"] == tokenizer.hash for x in manifests), "evidence": "manifests/index.json and manifests/tokenizer.json"},
        "evaluation_firewall": {"passed": firewall_ok, "evidence": "reports/firewall.json"},
        "packing_correctness": {"passed": bool(entries) and len(packing_report["policies"]) >= 3 and all(packing_report["invariants"].values()), "evidence": "reports/packing.json and ledgers/consumption.jsonl"},
        "mixture_compliance": {"passed": all(x["match"] for x in mixture.values()), "evidence": "reports/mixture_compliance.json"},
        "opus_audit_trail": {"passed": opus_ok, "evidence": "ledgers/opus.jsonl"},
        "crash_recovery": {"passed": crash_process_ok and resume_match and [x["batch_index"] for x in entries] == list(range(10)), "evidence": "reports/crash_expectation.json and ledgers/consumption.jsonl"},
        "replay": {"passed": replay_match, "evidence": "reports/replay.json"},
        "learning_trace": {"passed": learning_ok, "evidence": "ledgers/learning.jsonl"},
        "throughput": {"passed": performance_ok, "evidence": "performance.json and ledgers/consumption.jsonl"},
    }
    write_evidence(artifacts, checks)
    log.event("audit completed | ledger chains, manifests, firewalls and lineage verified")
    log.event(f"performance measured | utilization={performance['packing_utilization']:.3f} useful_tokens/s={performance['useful_loss_tokens_per_second']:.1f}")
    if not all(x["passed"] for x in checks.values()):
        raise AssertionError("one or more evidence requirements failed")
    log.passed("audit_completed", "all requirements passed")
    return {"checks": checks, "performance": performance, "artifacts": str(artifacts)}


def run_demo(artifacts: Path = ARTIFACTS) -> dict[str, Any]:
    """Orchestrate a real crash worker and a separate recovery worker."""
    entrypoint = ROOT / "run_demo.py"
    crash = subprocess.run([sys.executable, str(entrypoint), "--worker-crash", str(artifacts)], check=False)
    if crash.returncode != 86:
        raise RuntimeError(f"crash worker returned {crash.returncode}, expected 86")
    crash_report = read_json(artifacts / "reports/crash_expectation.json")
    crash_report["observed_exit_code"] = crash.returncode
    write_json(artifacts / "reports/crash_expectation.json", crash_report)
    EventLog(artifacts / "run.log", append=True).passed("crash_process_exit_observed", "exit_code=86")
    subprocess.run([sys.executable, str(entrypoint), "--worker-resume", str(artifacts)], check=True)
    evidence = read_json(artifacts / "evidence.json")
    performance = read_json(artifacts / "performance.json")
    return {"checks": evidence["requirements"], "performance": performance, "artifacts": str(artifacts)}
