"""Unicode/UTF-8 adversarial audit for the reversible K-code path."""

from __future__ import annotations

import json
from pathlib import Path

from rke import FullByteCodec, constrained_utf8_decode
from reversible_projection import ReversibleKroneckerProjection


OUT = Path(__file__).resolve().parent / "artifacts" / "unicode_security"


def main():
    malformed = {
        "overlong_slash": bytes.fromhex("c0af"),
        "overlong_nul": bytes.fromhex("c080"),
        "truncated_2": bytes.fromhex("c2"),
        "truncated_3": bytes.fromhex("e0a0"),
        "truncated_4": bytes.fromhex("f09080"),
        "surrogate": bytes.fromhex("eda080"),
        "bad_continuation": bytes.fromhex("e228a1"),
        "invalid_ff": bytes.fromhex("ff"),
    }
    valid = ["", "\x00", "é", "अ", "తెలుగు", "𐀀", "🦄", "한글"]
    decode_rejected = {name: False for name in malformed}
    for name, payload in malformed.items():
        try: payload.decode("utf-8")
        except UnicodeDecodeError: decode_rejected[name] = True
    projection = ReversibleKroneckerProjection(FullByteCodec(32))
    valid_roundtrips = {text: projection.decode(projection.encode(text.encode("utf-8"))) == text.encode("utf-8")
                       for text in valid}
    arbitrary_bytes_roundtrip = all(
        projection.decode(projection.encode(payload)) == payload
        for payload in malformed.values()
    )
    # Force invalid logits to be highest; constrained decoding must still return
    # a valid UTF-8 path rather than emitting malformed bytes.
    logits = [[-100.0] * 258 for _ in range(3)]
    logits[0][2 + 0xFF] = 100.0; logits[1][1] = 100.0; logits[2][1] = 100.0
    constrained = constrained_utf8_decode(__import__("numpy").array(logits), 2)
    result = {
        "protocol": "UTF-8 adversarial/security audit",
        "malformed_cases": list(malformed),
        "all_malformed_rejected_by_strict_decoder": all(decode_rejected.values()),
        "valid_unicode_roundtrips": valid_roundtrips,
        "all_valid_roundtrips": all(valid_roundtrips.values()),
        "arbitrary_invalid_bytes_remain_lossless_at_byte_layer": arbitrary_bytes_roundtrip,
        "constrained_decoder_output": list(constrained),
        "constrained_decoder_valid_utf8": constrained.decode("utf-8") is not None,
    }
    result["overall_pass"] = all((result["all_malformed_rejected_by_strict_decoder"],
                                   result["all_valid_roundtrips"],
                                   result["arbitrary_invalid_bytes_remain_lossless_at_byte_layer"],
                                   result["constrained_decoder_valid_utf8"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__": main()
