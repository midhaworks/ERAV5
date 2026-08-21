"""Exact reversible projection wrapper for the discrete Kronecker codec."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rke import FullByteCodec


@dataclass(frozen=True)
class ReversibleKroneckerProjection:
    """An exact code path using an invertible coordinate permutation.

    The projection is deliberately not dimension-reducing. Compression belongs
    to the separate transformer input path; this object is the exact decoder.
    """

    codec: FullByteCodec

    @property
    def dimension(self) -> int:
        return self.codec.feature_dim

    def encode(self, payload: bytes) -> np.ndarray:
        return self.codec.encode(payload).reshape(-1).astype(np.float64)

    def decode(self, embedding: np.ndarray) -> bytes:
        vector = np.asarray(embedding, dtype=np.float64)
        if vector.shape != (self.dimension,):
            raise ValueError("embedding has the wrong dimension")
        return self.codec.decode_ids(vector.reshape(self.codec.slots, self.codec.width).argmax(axis=1))

