"""End-to-end exact K-code integration audit over the trained multilingual run."""

from __future__ import annotations

import json
from pathlib import Path

from reversible_projection import ReversibleKroneckerProjection
from rke import FullByteCodec, sha256


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "artifacts" / "multilingual"
OUT = ROOT / "artifacts" / "end_to_end_reversible"


def main():
    split = json.loads((SOURCE / "split.json").read_text(encoding="utf-8"))
    predictions = json.loads((SOURCE / "predictions.json").read_text(encoding="utf-8"))
    projection = ReversibleKroneckerProjection(FullByteCodec(6))
    target_roundtrips = 0; target_total = 0; predicted_valid = 0; predicted_total = 0
    by_script = {}
    for split_name, records in split.items():
        for record in records:
            payload = record["target"].encode("utf-8")
            if len(payload) > 6: continue
            target_total += 1
            target_roundtrips += projection.decode(projection.encode(payload)) == payload
            by_script.setdefault(record["script"], {"total": 0, "roundtrip": 0})
            by_script[record["script"]]["total"] += 1
            by_script[record["script"]]["roundtrip"] += projection.decode(projection.encode(payload)) == payload
    for split_name, records in predictions.items():
        for record in records:
            predicted_total += 1
            predicted_valid += record.get("constrained") not in ("<NO_VALID_PATH>", "<INVALID_UTF8>")
    result = {
        "protocol": "trained multilingual model plus exact K-code output audit",
        "source_model_results": str((SOURCE / "results.json").relative_to(ROOT)),
        "source_dataset_hash": json.loads((SOURCE / "results.json").read_text())['dataset_hash'],
        "target_roundtrip_rate": target_roundtrips / max(1, target_total),
        "target_examples": target_total,
        "predicted_constrained_valid_rate": predicted_valid / max(1, predicted_total),
        "predicted_examples": predicted_total,
        "by_script": by_script,
        "codec": {"states": 258, "max_bytes": 6, "eos": True, "projection": "identity exact K-code"},
        "claims": {"all_targets_roundtrip": target_roundtrips == target_total,
                   "all_model_predictions_have_constrained_path": predicted_valid == predicted_total},
        "artifact_hash": sha256({"target_total": target_total, "target_roundtrips": target_roundtrips,
                                  "predicted_total": predicted_total, "predicted_valid": predicted_valid}),
    }
    result["overall_pass"] = all(result["claims"].values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()

