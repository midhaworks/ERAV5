import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rke import ContinuationByteCodec, FullByteCodec  # noqa: E402
from reversible_projection import ReversibleKroneckerProjection  # noqa: E402


class ReversibleProjectionTests(unittest.TestCase):
    def test_all_256_bytes_and_unknown_words(self):
        projection = ReversibleKroneckerProjection(FullByteCodec(4))
        for value in range(256):
            payload = bytes([value])
            self.assertEqual(projection.decode(projection.encode(payload)), payload)
        for text in ("𐀀", "é", "अ", "తెలుగు", "未知"):
            payload = text.encode("utf-8")
            if len(payload) <= 4:
                self.assertEqual(projection.decode(projection.encode(payload)), payload)

    def test_exhaustive_strings_through_length_two(self):
        projection = ReversibleKroneckerProjection(FullByteCodec(2))
        self.assertEqual(projection.decode(projection.encode(b"")), b"")
        for value in range(256):
            self.assertEqual(projection.decode(projection.encode(bytes([value]))), bytes([value]))
        for left in range(256):
            for right in range(256):
                payload = bytes([left, right])
                self.assertEqual(projection.decode(projection.encode(payload)), payload)

    def test_random_binary_strings_and_long_continuation_payload(self):
        projection = ReversibleKroneckerProjection(FullByteCodec(32))
        rng = np.random.default_rng(2308)
        for _ in range(1000):
            payload = bytes(rng.integers(0, 256, size=int(rng.integers(0, 33)), dtype=np.uint8))
            self.assertEqual(projection.decode(projection.encode(payload)), payload)
        codec = ContinuationByteCodec(24)
        payload = bytes(rng.integers(0, 256, size=10000, dtype=np.uint8))
        self.assertEqual(codec.decode_ids(codec.ids(payload)), payload)

    def test_dimension_and_invalid_shape_are_explicit(self):
        projection = ReversibleKroneckerProjection(FullByteCodec(4))
        self.assertEqual(projection.dimension, 5 * 258)
        with self.assertRaises(ValueError):
            projection.decode(np.zeros(projection.dimension - 1))


if __name__ == "__main__":
    unittest.main()
