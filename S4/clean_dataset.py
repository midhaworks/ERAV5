#!/usr/bin/env python3
"""Deterministically audit and clean a reasoning-summary JSONL dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_SUMMARY_FIELDS = ("title", "sub_title", "summary", "cur_task")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)"
    r"\d{3}[-.\s]?\d{4}(?!\w)"
)
PHONE_CONTEXT_RE = re.compile(
    r"\b(?:phone|mobile|telephone|tel|cell|sms|recipient|sender|call)\b", re.I
)
SECRET_PATTERNS = (
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?"
               r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.S),
)
TOXIC_RE = re.compile(
    r"\b(?:nigg(?:er|a)s?|fagg?ots?|retards?|kikes?|chinks?)\b", re.I
)
BENCHMARK_RE = re.compile(
    r"(?<!['’\w])(?:AIME(?:\s+20(?:24|25))?|MATH-500|GSM8K|"
    r"GPQA(?:\s+Diamond)?|MMLU(?:-Pro)?|HumanEval|MBPP|"
    r"ARC-Challenge|HellaSwag|WinoGrande)\b"
)
INJECTION_RE = re.compile(
    r"(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+"
    r"(?:instructions?|prompts?)", re.I,
)
INCOMPLETE_RE = re.compile(
    r"(?:message|request|problem|prompt).{0,80}"
    r"(?:incomplete|cut off|missing|no actual|not provided)|"
    r"(?:only|just)\s+(?:provided|shared|sent).{0,100}"
    r"(?:heading|header|constraints?|fragment)", re.I | re.S,
)
CODE_LINE_RE = re.compile(
    r"^\s*(?:def |class |from \S+ import |import \S+|function |const |let |var |"
    r"if\s*\(|for\s*\(|while\s*\(|return\b|[{}]|</?\w+|#include|SELECT\b|"
    r"CREATE TABLE\b|revision\s*=|down_revision\s*=)", re.I
)
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def script_family(char: str) -> str | None:
    if not char.isalpha():
        return None
    point = ord(char)
    if (
        0x0900 <= point <= 0x0DFF
        or 0x0F00 <= point <= 0x109F
        or 0xA8E0 <= point <= 0xA8FF
    ):
        return "indic"
    if 0x0600 <= point <= 0x08FF or 0xFB50 <= point <= 0xFEFF:
        return "arabic"
    if 0x0400 <= point <= 0x052F:
        return "cyrillic"
    if (
        0x3040 <= point <= 0x30FF
        or 0x3400 <= point <= 0x9FFF
        or 0xAC00 <= point <= 0xD7AF
    ):
        return "cjk"
    if "LATIN" in unicodedata.name(char, ""):
        return "latin"
    return "other"


def script_profile(text: str) -> str:
    counts = Counter(
        family for char in text if (family := script_family(char)) is not None
    )
    total = sum(counts.values())
    if not total:
        return "no_letters"
    dominant, count = counts.most_common(1)[0]
    if count / total >= 0.9:
        return f"{dominant}_dominant"
    return "mixed_script"


def normalize_text(text: str) -> tuple[str, int]:
    original = text
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = "".join(
        char for char in text
        if char in "\n\t" or unicodedata.category(char) not in {"Cc", "Cf"}
    )
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    return text, int(text != original)


def redact_sensitive(text: str) -> tuple[str, Counter]:
    counts: Counter[str] = Counter()
    text, n = EMAIL_RE.subn("[REDACTED_EMAIL]", text)
    counts["email_redactions"] += n

    # Long binary strings, timestamps and IDs can look like bare phone numbers.
    # Redact only formatted/+prefixed candidates or digits near phone-like context.
    phone_redactions = 0

    def redact_phone(match: re.Match[str]) -> str:
        nonlocal phone_redactions
        candidate = match.group(0)
        context = text[max(0, match.start() - 48):min(len(text), match.end() + 48)]
        formatted = candidate.startswith("+") or bool(re.search(r"[().\s-]", candidate))
        if formatted or PHONE_CONTEXT_RE.search(context):
            phone_redactions += 1
            return "[REDACTED_PHONE]"
        return candidate

    text = PHONE_RE.sub(redact_phone, text)
    counts["phone_redactions"] += phone_redactions
    for pattern in SECRET_PATTERNS:
        text, n = pattern.subn("[REDACTED_SECRET]", text)
        counts["secret_redactions"] += n
    return text, counts


def canonical(text: str) -> str:
    return " ".join(WORD_RE.findall(text.casefold()))


def code_dominant(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        return False
    code_lines = sum(bool(CODE_LINE_RE.search(line)) for line in lines)
    symbol_count = sum(text.count(char) for char in "{}();[]=<>")
    return code_lines / len(lines) >= 0.45 or (
        code_lines / len(lines) >= 0.3 and symbol_count >= 80
    )


def simhash(text: str) -> int:
    tokens = canonical(text).split()
    if len(tokens) < 3:
        shingles = tokens
    else:
        shingles = (" ".join(tokens[i:i + 3]) for i in range(len(tokens) - 2))
    vector = [0] * 64
    for shingle in set(shingles):
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    return sum(1 << bit for bit, score in enumerate(vector) if score >= 0)


def near_duplicate(
    signature: int,
    signatures: list[int],
    buckets: dict[tuple[int, int], list[int]],
) -> bool:
    candidates: set[int] = set()
    for band in range(4):
        candidates.update(buckets[(band, (signature >> (band * 16)) & 0xFFFF)])
    for index in candidates:
        if (signature ^ signatures[index]).bit_count() <= 3:
            return True
    return False


def add_signature(
    signature: int,
    index: int,
    buckets: dict[tuple[int, int], list[int]],
) -> None:
    for band in range(4):
        buckets[(band, (signature >> (band * 16)) & 0xFFFF)].append(index)


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "data/raw/reasoning-summaries-61k.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data/clean/reasoning-summaries-61k.clean.jsonl",
    )
    parser.add_argument(
        "--quarantine",
        type=Path,
        default=Path(__file__).parent / "data/quarantine/rejected.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).parent / "data/cleanup-report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.quarantine.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    issue_occurrences: Counter[str] = Counter()
    script_profiles_all: Counter[str] = Counter()
    script_profiles_kept: Counter[str] = Counter()
    field_lengths: dict[str, list[int]] = defaultdict(list)
    seen_exact: set[str] = set()
    signatures: list[int] = []
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)

    with (
        args.input.open(encoding="utf-8") as source,
        args.output.open("w", encoding="utf-8") as clean,
        args.quarantine.open("w", encoding="utf-8") as quarantine,
    ):
        for line_number, line in enumerate(source, 1):
            stats["input_rows"] += 1
            stats["input_bytes"] += len(line.encode("utf-8"))
            reasons: list[str] = []
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = {"_raw_line": line.rstrip("\n")}
                reasons.append("malformed_jsonl")

            if set(row) != {"user", "assistant"}:
                reasons.append("invalid_record_schema")
            if not reasons and not all(isinstance(row[key], str) for key in ("user", "assistant")):
                reasons.append("non_string_record_field")
            if not reasons:
                stats["input_characters"] += len(row["user"]) + len(row["assistant"])

            summary: dict = {}
            if not reasons:
                try:
                    summary = json.loads(row["assistant"])
                except json.JSONDecodeError:
                    reasons.append("malformed_summary_json")
                if not isinstance(summary, dict):
                    reasons.append("summary_not_object")
                elif set(summary) != set(REQUIRED_SUMMARY_FIELDS):
                    reasons.append("invalid_summary_schema")
                elif not all(
                    isinstance(summary[field], str) for field in REQUIRED_SUMMARY_FIELDS
                ):
                    reasons.append("non_string_summary_field")

            if not reasons:
                user, changed = normalize_text(row["user"])
                stats["normalized_user_rows"] += changed
                profile = script_profile(user)
                script_profiles_all[profile] += 1
                normalized_summary = {}
                for field in REQUIRED_SUMMARY_FIELDS:
                    value, changed = normalize_text(summary[field])
                    normalized_summary[field] = value
                    stats["normalized_summary_fields"] += changed
                summary = normalized_summary

                user, redactions = redact_sensitive(user)
                stats.update(redactions)
                for field in REQUIRED_SUMMARY_FIELDS:
                    summary[field], redactions = redact_sensitive(summary[field])
                    stats.update(redactions)

                field_lengths["user"].append(len(user))
                for field in REQUIRED_SUMMARY_FIELDS:
                    field_lengths[field].append(len(summary[field]))

                if not 80 <= len(user) <= 50_000:
                    reasons.append("user_length_outlier")
                if not 3 <= len(summary["title"]) <= 120:
                    reasons.append("title_length_outlier")
                if not 10 <= len(summary["sub_title"]) <= 240:
                    reasons.append("subtitle_length_outlier")
                if not 30 <= len(summary["summary"]) <= 2_000:
                    reasons.append("summary_length_outlier")
                if not 20 <= len(summary["cur_task"]) <= 500:
                    reasons.append("current_task_length_outlier")
                if len(user) >= 300 and len(summary["summary"]) > len(user) * 1.2:
                    reasons.append("non_compressive_summary")

                all_text = user + "\n" + "\n".join(summary.values())
                if TOXIC_RE.search(all_text):
                    reasons.append("high_confidence_toxicity")
                if BENCHMARK_RE.search(all_text):
                    reasons.append("benchmark_contamination")
                if INJECTION_RE.search(all_text):
                    reasons.append("prompt_injection")
                if len(user) < 700 and INCOMPLETE_RE.search(user):
                    reasons.append("incomplete_or_placeholder_task")
                if code_dominant(user):
                    reasons.append("code_provenance_unknown")

                stable_summary = json.dumps(
                    summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                )
                fingerprint_text = canonical(user + "\n" + stable_summary)
                exact_hash = hashlib.sha256(
                    fingerprint_text.encode("utf-8")
                ).hexdigest()
                if exact_hash in seen_exact:
                    reasons.append("exact_duplicate")

                if not reasons:
                    signature = simhash(user + "\n" + summary["summary"])
                    if near_duplicate(signature, signatures, buckets):
                        reasons.append("near_duplicate")
                    else:
                        kept_index = len(signatures)
                        signatures.append(signature)
                        add_signature(signature, kept_index, buckets)
                        seen_exact.add(exact_hash)

                row = {"user": user, "assistant": stable_summary}

            for reason in set(reasons):
                issue_occurrences[reason] += 1
            if reasons:
                primary = reasons[0]
                stats[f"rejected:{primary}"] += 1
                row["_cleanup_reasons"] = reasons
                row["_source_line"] = line_number
                quarantine.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                continue

            rendered = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            clean.write(rendered + "\n")
            stats["output_rows"] += 1
            script_profiles_kept[profile] += 1
            stats["output_bytes"] += len((rendered + "\n").encode("utf-8"))
            stats["output_characters"] += len(row["user"]) + len(row["assistant"])

    rejected = stats["input_rows"] - stats["output_rows"]
    input_mb = stats["input_bytes"] / 1_000_000
    output_mb = stats["output_bytes"] / 1_000_000
    input_tokens_estimate = round(stats["input_characters"] / 4)
    output_tokens_estimate = round(stats["output_characters"] / 4)
    report = {
        "dataset": {
            "id": "SupraLabs/reasoning-summaries-61k",
            "source_url": "https://huggingface.co/datasets/SupraLabs/reasoning-summaries-61k",
            "files_url": "https://huggingface.co/datasets/SupraLabs/reasoning-summaries-61k/tree/main",
            "declared_license": "apache-2.0",
            "input_sha256": file_sha256(args.input),
            "output_sha256": file_sha256(args.output),
            "quarantine_sha256": file_sha256(args.quarantine),
            "pipeline_sha256": file_sha256(Path(__file__)),
            "record_schema": ["user", "assistant"],
            "size_evidence": {
                "interpretation": "10–100 MB decimal file-size band",
                "assignment_min_bytes": 10_000_000,
                "assignment_max_bytes": 100_000_000,
                "hugging_face_listed_size_mb": 79.4,
                "local_input_bytes": stats["input_bytes"],
                "local_input_mb_decimal": input_mb,
                "within_assignment_band": (
                    10_000_000 <= stats["input_bytes"] <= 100_000_000
                ),
                "measurement": (
                    "Exact UTF-8 bytes summed from the downloaded JSONL; "
                    "cross-checked against the local file size."
                ),
            },
        },
        "stages": [
            {
                "number": 1,
                "name": "Extract",
                "applied": "Parse JSONL and nested assistant JSON; enforce the two-field record schema.",
            },
            {
                "number": 2,
                "name": "Normalize",
                "applied": "NFC Unicode, newline and control-character normalization while preserving meaningful structure.",
            },
            {
                "number": 3,
                "name": "Language ID",
                "applied": "Conservative Unicode script profiling; report mixed scripts without deleting multilingual reasoning.",
            },
            {
                "number": 4,
                "name": "Quality filter",
                "applied": "Schema, field-length, completeness, compression, injection and high-confidence toxicity gates.",
            },
            {
                "number": 5,
                "name": "Deduplicate",
                "applied": "Canonical SHA-256 exact matching plus 64-bit 3-word-shingle SimHash near matching.",
            },
            {
                "number": 6,
                "name": "PII scrub",
                "applied": "Redact high-confidence email, contextual phone and secret patterns.",
            },
            {
                "number": 7,
                "name": "Decontaminate",
                "applied": "Quarantine named benchmark traces and code records with unresolved source provenance.",
            },
            {
                "number": 8,
                "name": "Manifest",
                "applied": "Record source, policy, counts, distributions and SHA-256 hashes for reproducibility.",
            },
        ],
        "policy": {
            "deterministic": True,
            "unicode_normalization": "NFC",
            "language_id_method": (
                "Unicode script-family profile, used for reporting rather than deletion"
            ),
            "near_duplicate_method": "64-bit 3-word-shingle SimHash; Hamming distance <= 3",
            "benchmark_markers_quarantined": True,
            "high_confidence_pii_and_secret_redaction": True,
            "code_dominant_records_quarantined": (
                "record-level upstream provenance is unavailable"
            ),
            "unresolved_provenance_concern": (
                "The collection card names an upstream code dataset with no "
                "discoverable license and provides no per-record source field."
            ),
        },
        "counts": {
            "input_rows": stats["input_rows"],
            "output_rows": stats["output_rows"],
            "rejected_rows": rejected,
            "rejection_rate": rejected / stats["input_rows"],
            "retention_rate": stats["output_rows"] / stats["input_rows"],
            "input_bytes": stats["input_bytes"],
            "output_bytes": stats["output_bytes"],
            "bytes_removed": stats["input_bytes"] - stats["output_bytes"],
            "input_megabytes_decimal": input_mb,
            "output_megabytes_decimal": output_mb,
            "byte_reduction": 1 - stats["output_bytes"] / stats["input_bytes"],
            "input_characters": stats["input_characters"],
            "output_characters": stats["output_characters"],
            "characters_removed": (
                stats["input_characters"] - stats["output_characters"]
            ),
            "character_reduction": (
                1 - stats["output_characters"] / stats["input_characters"]
            ),
            "estimated_input_tokens_at_4_chars": input_tokens_estimate,
            "estimated_output_tokens_at_4_chars": output_tokens_estimate,
            "estimated_tokens_removed_at_4_chars": (
                input_tokens_estimate - output_tokens_estimate
            ),
            "estimated_token_reduction": (
                1 - output_tokens_estimate / input_tokens_estimate
            ),
            "average_input_bytes_per_row": (
                stats["input_bytes"] / stats["input_rows"]
            ),
            "average_output_bytes_per_row": (
                stats["output_bytes"] / stats["output_rows"]
            ),
        },
        "mutations": {
            key: value for key, value in sorted(stats.items())
            if key.startswith(("normalized_", "email_", "phone_", "secret_"))
        },
        "primary_rejections": {
            key.removeprefix("rejected:"): value
            for key, value in sorted(stats.items())
            if key.startswith("rejected:")
        },
        "issue_occurrences": dict(sorted(issue_occurrences.items())),
        "script_profiles": {
            "all_valid_rows": dict(sorted(script_profiles_all.items())),
            "retained_rows": dict(sorted(script_profiles_kept.items())),
        },
        "lengths": {
            field: {
                "min": min(values, default=0),
                "p50": percentile(values, 0.5),
                "p95": percentile(values, 0.95),
                "max": max(values, default=0),
            }
            for field, values in field_lengths.items()
        },
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["counts"], indent=2))
    print(json.dumps(report["primary_rejections"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
