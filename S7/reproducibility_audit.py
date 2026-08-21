"""Determinism, serialization and tamper-detection audit for exact K-code."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from reversible_projection import ReversibleKroneckerProjection
from rke import FullByteCodec, sha256


OUT = Path(__file__).resolve().parent / "artifacts" / "reproducibility"


def main():
    config = {"codec": "FullByteCodec", "max_bytes": 256, "states": 258,
              "eos_id": 1, "pad_id": 0, "format_version": 1}
    payloads = [b"", b"hello", "తెలుగు".encode(), "𐀀🦄".encode(), bytes(range(256)), b"x" * 64]
    projection = ReversibleKroneckerProjection(FullByteCodec(256))
    codes = [projection.encode(payload) for payload in payloads]
    hashes_a = [hashlib.sha256(code.tobytes()).hexdigest() for code in codes]
    hashes_b = [hashlib.sha256(projection.encode(payload).tobytes()).hexdigest() for payload in payloads]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "codes.npz"
        np.savez(path, **{f"code_{i}": code for i, code in enumerate(codes)})
        loaded = np.load(path)
        reload_roundtrip = all(projection.decode(loaded[f"code_{i}"]) == payload
                                for i, payload in enumerate(payloads))
    manifest_hash = sha256(config)
    tampered = dict(config); tampered["max_bytes"] = 63
    result = {
        "protocol": "codec determinism/serialization audit",
        "config": config, "manifest_hash": manifest_hash,
        "repeat_hashes_match": hashes_a == hashes_b,
        "serialized_codes_reload_exactly": reload_roundtrip,
        "tamper_changes_manifest_hash": manifest_hash != sha256(tampered),
        "payload_count": len(payloads),
    }
    result["overall_pass"] = all((result["repeat_hashes_match"],
                                   result["serialized_codes_reload_exactly"],
                                   result["tamper_changes_manifest_hash"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
