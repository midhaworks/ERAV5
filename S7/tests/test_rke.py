import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rke import (EOS, ContinuationByteCodec, FullByteCodec, MaskedSlotRKE, RefinedSlotRKE,
                 ReversibleCodec, TinyTransformer,
                 constrained_utf8_decode, make_split, original_truncated_code)  # noqa: E402
from lm_compare import make_dataset  # noqa: E402
from multilingual import dataset as multilingual_dataset  # noqa: E402
from natural_corpus import _choose_rke_state, build_dataset, tokenize  # noqa: E402
from torch_port import run_parity  # noqa: E402
from continuation_neural import make_payloads, run_continuation_neural  # noqa: E402
from cross_block_lm import build_long_dataset, choose_state  # noqa: E402


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

    def test_continuation_codec_round_trips_unbounded_lengths(self):
        codec = ContinuationByteCodec(block_bytes=4)
        rng = np.random.default_rng(2026)
        for length in (0, 1, 3, 4, 5, 8, 9, 97, 10_000):
            payload = rng.integers(0, 256, size=length, dtype=np.uint8).tobytes()
            blocks = codec.ids(payload)
            self.assertEqual(payload, codec.decode_ids(blocks))
            self.assertEqual(max(1, (length + 3) // 4), len(blocks))
            self.assertTrue(all(sum(int(state in (1, 2)) for state in block) == 1 for block in blocks))

    def test_continuation_codec_rejects_missing_or_wrong_terminators(self):
        codec = ContinuationByteCodec(block_bytes=4)
        blocks = codec.ids(b"abcdefgh")
        broken = [block.copy() for block in blocks]
        broken[0][4] = 1  # intermediate block must use CONT
        with self.assertRaises(ValueError):
            codec.decode_ids(broken)
        with self.assertRaises(ValueError):
            codec.decode_ids([])
        predicted = codec.ids(b"abc")[0]
        predicted[4:] = 258  # arbitrary predictions in loss-masked suffix
        self.assertEqual(b"abc", codec.decode_ids([predicted]))

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

    def test_masked_slot_gradient_and_causality(self):
        codec = ReversibleCodec("abc", 3)
        model = MaskedSlotRKE(codec, d_slot=2, kernel_size=2, seed=13)
        features = model.features(["ab"], ["c"])
        targets = np.stack([codec.ids("ab")])
        _, grads = model.loss_and_grads(features, targets, targets != 0)
        epsilon, key, index = 1e-5, "Wslot", (0, 0, 0)
        original = model.params[key][index]
        model.params[key][index] = original + epsilon
        plus = model.loss_and_grads(features, targets, targets != 0)[0]
        model.params[key][index] = original - epsilon
        minus = model.loss_and_grads(features, targets, targets != 0)[0]
        model.params[key][index] = original
        self.assertAlmostEqual(float(grads[key][index]), float((plus - minus) / (2 * epsilon)), places=5)

        # A source slot can affect later slots, never earlier ones.
        base = np.zeros((1, codec.slots, model.d_slot))
        base[0, 1] = 1.0
        pre = base.copy()
        for lag in range(1, model.kernel_size + 1):
            pre[:, lag:] += base[:, :-lag] @ model.params["Wslot"][lag - 1]
        self.assertTrue(np.allclose(pre[0, 0], base[0, 0]))
        self.assertFalse(np.allclose(pre[0, 2], base[0, 2]))

    def test_two_pass_refiner_gradients_match_finite_difference(self):
        codec = ReversibleCodec("abc", 3)
        model = RefinedSlotRKE(codec, d_slot=2, kernel_size=2, seed=17)
        features = model.features(["ab"], ["c"])
        targets = np.stack([codec.ids("ab")])
        mask = targets != 0
        _, grads = model.loss_and_grads(features, targets, mask)
        epsilon = 1e-5
        for key, index in (("Wref", (0, 0, 0)), ("Ein", (1, 0)), ("Wq", (0, 0))):
            original = model.params[key][index]
            model.params[key][index] = original + epsilon
            plus = model.loss_and_grads(features, targets, mask)[0]
            model.params[key][index] = original - epsilon
            minus = model.loss_and_grads(features, targets, mask)[0]
            model.params[key][index] = original
            self.assertAlmostEqual(float(grads[key][index]), float((plus - minus) / (2 * epsilon)), places=5)

    def test_two_pass_refiner_has_fixed_parallel_shapes(self):
        codec = FullByteCodec(4)
        model = RefinedSlotRKE(codec, d_slot=2, kernel_size=2, seed=19)
        sample = np.stack([np.stack([codec.encode(b"c"), codec.encode(b"a"), codec.encode(b"b")])])
        logits, cache = model.forward(sample)
        self.assertEqual((1, 5, 258), logits.shape)
        self.assertEqual((1, 5, 258), cache["proposal_probs"].shape)
        np.testing.assert_allclose(cache["proposal_probs"].sum(axis=-1), 1.0, atol=1e-12)
        # Zero-initialized refinement is an exact residual identity.
        np.testing.assert_allclose(logits, cache["proposal_logits"], atol=1e-12)

    def test_numpy_torch_parity(self):
        result = run_parity()
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertLessEqual(result["errors"]["gradient_max_abs"], result["tolerance"])

    def test_continuation_payload_generator_terminates_and_is_disjoint(self):
        train, test = make_payloads(train_size=30, test_size=10)
        self.assertEqual((30, 10), (len(train), len(test)))
        self.assertFalse(set(train) & set(test))
        self.assertIn(0, {len(item) for item in train + test})
        self.assertGreater(max(map(len, train + test)), 24)

    def test_neural_continuation_integration_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_continuation_neural(Path(directory))
        self.assertTrue(result["passed"])
        self.assertEqual(1.0, result["test"]["exact_match"])
        self.assertGreater(result["test"]["loss_bearing_cont_states"], 0)

    def test_cross_block_dataset_is_natural_and_paragraph_disjoint(self):
        data, audit = build_long_dataset()
        self.assertFalse(audit["paragraph_leakage"])
        self.assertFalse(audit["document_leakage"])
        self.assertEqual("document", audit["split_unit"])
        self.assertEqual(40, audit["corpus_documents"])
        self.assertEqual({"train": 4000, "validation": 800, "test": 500},
                         {name: len(rows) for name, rows in data.items()})
        self.assertTrue(all(len(row["target"].encode("utf-8")) > 24 for rows in data.values() for row in rows))

    def test_cross_block_decoder_forbids_first_block_eos(self):
        scores = np.full(259, -10.0); scores[1] = 20.0; scores[ord("a") + 3] = 19.0
        self.assertEqual(ord("a") + 3, choose_state(scores, b"", b"", 0, 0, True))
        self.assertEqual(1, choose_state(scores, b"x" * 24, b"", 0, 1, True))

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

    def test_multilingual_split_holds_out_words_not_components(self):
        data = multilingual_dataset()
        self.assertFalse({x["target"] for x in data["train"]} & {x["target"] for x in data["held_out"]})
        for script in {x["script"] for x in data["held_out"]}:
            train = [x for x in data["train"] if x["script"] == script]
            held = [x for x in data["held_out"] if x["script"] == script]
            self.assertLessEqual({x["stem"] for x in held}, {x["stem"] for x in train})
            self.assertLessEqual({x["suffix"] for x in held}, {x["suffix"] for x in train})

    def test_natural_tokenizer_preserves_unicode_letters_and_marks(self):
        self.assertEqual(["भारत", "తెలుగు", "سنڌي", "english"], tokenize("भारत, తెలుగు; سنڌي — English!"))

    def test_natural_paragraph_splits_do_not_leak(self):
        data, audit = build_dataset({"train": 20, "validation": 10, "test": 10})
        self.assertFalse(audit["split_policy"]["paragraph_leakage"])
        paragraph_sets = {split: {(x["language"], x["paragraph"]) for x in rows} for split, rows in data.items()}
        self.assertFalse(paragraph_sets["train"] & paragraph_sets["validation"])
        self.assertFalse(paragraph_sets["train"] & paragraph_sets["test"])

    def test_causal_rke_state_decoder_enforces_utf8(self):
        row = np.full(258, -10.0)
        row[257] = 20.0  # illegal byte FF
        row[0xC3 + 2] = 19.0
        self.assertEqual(0xC3, _choose_rke_state(row, b""))
        row = np.full(258, -10.0)
        row[1] = 20.0  # EOS is illegal in an incomplete sequence
        row[0xA9 + 2] = 19.0
        self.assertEqual(0xA9, _choose_rke_state(row, bytes([0xC3])))
        row = np.full(258, -10.0); row[1] = 20.0
        self.assertIsNone(_choose_rke_state(row, "é".encode("utf-8")))


if __name__ == "__main__":
    unittest.main()
