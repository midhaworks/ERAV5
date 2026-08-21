"""CPU performance/memory stress report for exact byte and CONT codecs."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from rke import ContinuationByteCodec, FullByteCodec


OUT = Path(__file__).resolve().parent / "artifacts" / "performance_stress"


def main():
    rng = np.random.default_rng(2308); codec = ContinuationByteCodec(24)
    rows = []
    for length in (0, 1, 32, 256, 1024, 10_000):
        payload = bytes(rng.integers(0, 256, size=length, dtype=np.uint8))
        start = time.perf_counter(); encoded = codec.encode(payload); encode_seconds = time.perf_counter() - start
        states = codec.ids(payload)
        start = time.perf_counter(); recovered = codec.decode_ids(states); decode_seconds = time.perf_counter() - start
        rows.append({"payload_bytes": length, "blocks": len(encoded),
                     "encode_bytes_per_second": length / max(encode_seconds, 1e-12),
                     "decode_bytes_per_second": length / max(decode_seconds, 1e-12),
                     "exact": recovered == payload,
                     "encoded_bytes": sum(block.nbytes for block in encoded)})
    full = FullByteCodec(256)
    result = {"protocol": "exact K-code CPU performance stress", "block_bytes": 24,
              "rows": rows, "full_byte_code_feature_dim_at_256": full.feature_dim,
              "maximum_tested_payload": 10_000,
              "all_exact": all(row["exact"] for row in rows),
              "interpretation": "codec-only CPU stress; excludes transformer compute and accelerator kernels"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
