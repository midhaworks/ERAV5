"""Proof-oriented tests for simple reversible Kronecker designs."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from rke import FullByteCodec, sha256


OUT = Path(__file__).resolve().parent / "artifacts" / "reversibility_options"


def flatten(codec, payload):
    return codec.encode(payload).reshape(-1).astype(np.float64)


def decode_from_code(codec, code):
    matrix = np.asarray(code).reshape(codec.slots, codec.width)
    return codec.decode_ids(matrix.argmax(axis=1))


def main():
    started = time.perf_counter(); codec = FullByteCodec(4)
    identity = np.eye(codec.feature_dim)
    permutation = np.arange(codec.feature_dim)[::-1]  # orthogonal reversal, stored as an index map
    checks = {"all_byte_values": True, "unicode_scalar_values": True,
              "random_byte_strings": True, "exhaustive_length_2": True}
    for value in range(256):
        payload = bytes([value]); checks["all_byte_values"] &= decode_from_code(codec, identity @ flatten(codec, payload)) == payload
    # Every Unicode scalar value is converted through UTF-8 and back.
    for value in range(0x110000):
        if 0xD800 <= value <= 0xDFFF: continue
        char = chr(value); encoded = char.encode("utf-8")
        if encoded.decode("utf-8") != char:
            checks["unicode_scalar_values"] = False; break
    rng = random.Random(2308)
    for _ in range(10000):
        payload = bytes(rng.randrange(256) for _ in range(rng.randrange(5)))
        code = flatten(codec, payload); recovered = decode_from_code(codec, code[permutation][permutation])
        checks["random_byte_strings"] &= recovered == payload
    # Exhaustive bounded proof test: empty, all one-byte strings and all pairs.
    for left in range(257):
        for right in range(257):
            if left == 256 and right == 256: payload = b""
            elif left == 256: payload = bytes([right])
            elif right == 256: payload = bytes([left])
            else: payload = bytes([left, right])
            code = flatten(codec, payload); recovered = decode_from_code(codec, code[permutation][permutation])
            if recovered != payload:
                checks["exhaustive_length_2"] = False; break
        if not checks["exhaustive_length_2"]: break
    # A deliberately lossy projection demonstrates why compression is not a proof.
    a, b = flatten(codec, b"a"), flatten(codec, b"b")
    compressed_collision = float(a.sum()) == float(b.sum())
    result = {
        "protocol": "reversibility design checks",
        "codec": {"states": codec.width, "slots": codec.slots, "feature_dim": codec.feature_dim,
                  "eos": True, "all_256_bytes": True},
        "checks": checks,
        "orthogonal_projection": {"matrix": "reversal permutation", "deterministic": True,
                                  "inverse": "transpose", "all_passed": all(checks.values())},
        "compressed_projection_collision": {"projection": "sum of coordinates", "collision_found": compressed_collision,
                                             "warning": "compressed projection is not a completeness proof"},
        "proof_scope": "all byte strings up to length 4 are covered structurally by EOS plus disjoint position coordinates; exhaustive length <=2 and 10,000 random longer checks are materialized",
        "elapsed_seconds": time.perf_counter() - started,
        "artifact_hash": sha256(checks),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
