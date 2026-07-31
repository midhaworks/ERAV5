# Copyright 2026 Avnish Midha. All rights reserved.
# Author: Avnish Midha
# GitHub: avnishbm
# Purpose: Implement lossless PieceVocab encoding, decoding, and serialization.

"""Lossless whole-piece vocabulary with character fallback.

The encoder keeps leading whitespace attached to the following non-whitespace
piece.  Selected frequent pieces use one token; every other piece falls back
to literal Unicode-character tokens.  Decoding is therefore just literal
concatenation and does not need spacing heuristics or normalization.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


PIECE_RE = re.compile(r"\s*\S+")
ESCAPE_TOKEN = "\x00UNICODE_ESCAPE"
HEX_TOKENS = tuple(f"\x00HEX_{digit:X}" for digit in range(16))
SPECIAL_FALLBACK_TOKENS = (ESCAPE_TOKEN,) + HEX_TOKENS


def split_pieces(text: str) -> list[str]:
    """Partition *text* without dropping or changing any character."""
    matches = list(PIECE_RE.finditer(text))
    pieces = [match.group(0) for match in matches]
    consumed = matches[-1].end() if matches else 0
    if consumed < len(text):
        pieces.append(text[consumed:])  # trailing or all-whitespace input
    return pieces


class PieceVocab:
    """A reversible piece tokenizer whose public encoding uses integer IDs."""

    def __init__(self, vocab: Iterable[str], whole_pieces: Iterable[str]):
        self.vocab = list(vocab)
        if len(self.vocab) != len(set(self.vocab)):
            raise ValueError("vocabulary contains duplicate token strings")
        self.token_to_id = {token: token_id for token_id, token in enumerate(self.vocab)}
        self.whole_pieces = frozenset(whole_pieces)
        missing = self.whole_pieces.difference(self.token_to_id)
        if missing:
            raise ValueError(f"whole-piece tokens missing from vocabulary: {len(missing)}")
        self.has_unicode_fallback = all(
            token in self.token_to_id for token in SPECIAL_FALLBACK_TOKENS
        )

    def encode(self, text: str) -> list[int]:
        """Encode text losslessly, raising rather than silently emitting UNK."""
        ids: list[int] = []
        for piece in split_pieces(text):
            piece_id = self.token_to_id.get(piece)
            if piece in self.whole_pieces and piece_id is not None:
                ids.append(piece_id)
                continue
            for char in piece:
                char_id = self.token_to_id.get(char)
                if char_id is not None:
                    ids.append(char_id)
                elif self.has_unicode_fallback:
                    ids.append(self.token_to_id[ESCAPE_TOKEN])
                    ids.extend(
                        self.token_to_id[HEX_TOKENS[int(digit, 16)]]
                        for digit in f"{ord(char):06X}"
                    )
                else:
                    raise ValueError(
                        f"character U+{ord(char):04X} is outside this corpus vocabulary"
                    )
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """Decode IDs exactly; no normalization or whitespace insertion occurs."""
        token_ids = list(ids)
        output: list[str] = []
        index = 0
        while index < len(token_ids):
            token_id = token_ids[index]
            if not isinstance(token_id, int):
                raise TypeError(f"token ID must be int, got {type(token_id).__name__}")
            if token_id < 0 or token_id >= len(self.vocab):
                raise ValueError(f"token ID {token_id} is outside the vocabulary")
            token = self.vocab[token_id]
            if token == ESCAPE_TOKEN:
                payload = token_ids[index + 1:index + 7]
                if len(payload) != 6:
                    raise ValueError("truncated Unicode escape in token sequence")
                try:
                    digits_list = []
                    for payload_id in payload:
                        if not isinstance(payload_id, int):
                            raise TypeError("Unicode escape payload ID must be int")
                        if payload_id < 0 or payload_id >= len(self.vocab):
                            raise ValueError("Unicode escape payload ID is outside vocabulary")
                        digits_list.append(
                            f"{HEX_TOKENS.index(self.vocab[payload_id]):X}"
                        )
                    digits = "".join(digits_list)
                    output.append(chr(int(digits, 16)))
                except (IndexError, TypeError, ValueError) as exc:
                    raise ValueError("invalid Unicode escape in token sequence") from exc
                index += 7
                continue
            if token in HEX_TOKENS:
                raise ValueError("hex fallback token appeared outside a Unicode escape")
            output.append(token)
            index += 1
        return "".join(output)

    def encode_tokens(self, text: str) -> list[str]:
        return [self.vocab[token_id] for token_id in self.encode(text)]

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def to_dict(self, meta: dict | None = None) -> dict:
        bundle = {
            "format": "piecevocab_v1",
            "runtime": {
                "required": True,
                "file": "piecevocab.py",
                "loader": "piecevocab.load",
                "warning": "This custom JSON is data only. Load it with piecevocab.py; do not evaluate the JSON as a standalone tokenizer.",
            },
            "vocab": self.vocab,
            "whole_pieces": sorted(self.whole_pieces),
        }
        if meta is not None:
            bundle["meta"] = meta
        return bundle

    @classmethod
    def from_dict(cls, bundle: dict) -> "PieceVocab":
        if bundle.get("format") not in {"piecevocab_v1", "faithful_wordvocab_v1"}:
            raise ValueError("unsupported PieceVocab format")
        return cls(bundle["vocab"], bundle["whole_pieces"])


def load(path: str | Path) -> PieceVocab:
    with Path(path).open(encoding="utf-8") as handle:
        return PieceVocab.from_dict(json.load(handle))
