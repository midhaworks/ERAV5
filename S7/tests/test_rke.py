import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rke import (EOS, FullByteCodec, ReversibleCodec, TinyTransformer,
                 constrained_utf8_decode, make_split, original_truncated_code)  # noqa: E402
from lm_compare import make_dataset  # noqa: E402


class ReversibleKroneckerTests(unittest.TestCase):
    def setUp(self):
        self.codec = ReversibleCodec("abc012", 6)

    def test_round_trip_and_unique_codes(self):
        strings = ["", "a", "abc", "012", "abc012", "a0b1c2"]
        codes = []
        for text in strings:
            code = self.codec.encode(text)
            self.assertEqual(text, self.codec.decode_logits(code))
            codes.append(code.tobytes())
        self.assertEqual(len(strings), len(set(codes)))

    def test_eos_makes_prefixes_distinct(self):
        short = self.codec.ids("a")
        long = self.codec.ids("aa")
        self.assertNotEqual(short.tolist(), long.tolist())
        self.assertEqual(self.codec.symbol_to_id[EOS], short[1])

    def test_original_truncation_collision_counterexample(self):
        self.assertEqual(original_truncated_code("abcd00", 4), original_truncated_code("abcd11", 4))
        self.assertNotEqual(self.codec.encode("abc012").tobytes(), self.codec.encode("abc011").tobytes())

    def test_split_has_no_vocab_leakage(self):
        train, test = make_split(4, "abc012", 6, 100, 40)
        self.assertFalse(set(train) & set(test))

    def test_invalid_code_without_eos_is_rejected(self):
        ids = np.full(self.codec.slots, self.codec.symbol_to_id["a"])
        with self.assertRaises(ValueError):
            self.codec.decode_ids(ids)

    def test_full_byte_codec_covers_every_byte(self):
        codec = FullByteCodec(4)
        for value in range(256):
            payload = bytes([value])
            self.assertEqual(payload, codec.decode_ids(codec.ids(payload)))

    def test_manual_transformer_gradient_matches_finite_difference(self):
        codec = ReversibleCodec("abc", 2)
        model = TinyTransformer(codec, d_slot=2, seed=3)
        features = model.features(["ab"], ["c"])
        targets = np.stack([codec.ids("ab")])
        _, grads = model.loss_and_grads(features, targets)
        epsilon = 1e-5
        for key, index in (("Ein", (0, 0)), ("Wo", (0, 0)), ("Wq", (0, 0))):
            original = model.params[key][index]
            model.params[key][index] = original + epsilon
            plus = model.loss_and_grads(features, targets)[0]
            model.params[key][index] = original - epsilon
            minus = model.loss_and_grads(features, targets)[0]
            model.params[key][index] = original
            numerical = (plus - minus) / (2 * epsilon)
            self.assertAlmostEqual(float(grads[key][index]), float(numerical), places=5)

    def test_utf8_decoder_masks_invalid_bytes_and_incomplete_eos(self):
        # Prefer illegal 0xFF everywhere, but leave the valid UTF-8 path for “é”.
        logits = np.full((3, 258), -10.0)
        logits[:, 257] = 20.0  # byte 0xFF: always illegal
        logits[0, 0xC3 + 2] = 19.0
        logits[0, 1] = 18.0  # EOS is legal here but lower than valid C3
        logits[1, 1] = 19.5  # illegal while C3 is incomplete
        logits[1, 0xA9 + 2] = 19.0
        logits[2, 1] = 19.0
        self.assertEqual("é", constrained_utf8_decode(logits, 2).decode("utf-8"))

    def test_eos_loss_masks_every_post_eos_pad_slot(self):
        codec = ReversibleCodec("abc", 4)
        model = TinyTransformer(codec, d_slot=2, seed=8)
        features = model.features(["a"], ["b"])
        targets = np.stack([codec.ids("a")])
        mask = targets != codec.symbol_to_id["<PAD>"]
        changed = targets.copy()
        changed[~mask] = codec.symbol_to_id["c"]
        loss_a, grads_a = model.loss_and_grads(features, targets, mask)
        loss_b, grads_b = model.loss_and_grads(features, changed, mask)
        self.assertAlmostEqual(loss_a, loss_b, places=12)
        for key in grads_a:
            np.testing.assert_allclose(grads_a[key], grads_b[key], atol=1e-12)

    def test_lm_split_holds_out_whole_tokens_not_components(self):
        data = make_dataset()
        train_targets = {x["target"] for x in data["train"]}
        oov_targets = {x["target"] for x in data["held_out_compositions"]}
        self.assertFalse(train_targets & oov_targets)
        self.assertLessEqual({x["stem"] for x in data["held_out_compositions"]}, {x["stem"] for x in data["train"]})
        self.assertLessEqual({x["suffix"] for x in data["held_out_compositions"]}, {x["suffix"] for x in data["train"]})


if __name__ == "__main__":
    unittest.main()
