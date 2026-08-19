import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import acquire_multidoc as corpus  # noqa: E402


class MultiDocumentCorpusTests(unittest.TestCase):
    @staticmethod
    def fake_links(language, root_title, limit=500):
        return [f"{language}-linked-{index}" for index in range(9)]

    @staticmethod
    def fake_pages(language, titles):
        result = []
        for index, title in enumerate(titles):
            text = (f"{language} {title} reproducible research document. " * 70)[:2_200]
            result.append({
                "pageid": (sum(map(ord, language)) * 10_000) + index,
                "title": title,
                "revision_id": 1_000_000 + index,
                "parent_revision_id": 999_999 + index,
                "revision_timestamp": "2026-01-01T00:00:00Z",
                "canonical_url": f"https://{language}.wikipedia.org/wiki/{index}",
                "source_characters": len(text),
                "stored_characters": len(text),
                "stored_truncated": False,
                "text": text,
            })
        return result

    def test_acquisition_is_manifested_validated_and_reused_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with patch.object(corpus, "linked_titles", side_effect=self.fake_links), \
                    patch.object(corpus, "fetch_pages", side_effect=self.fake_pages), \
                    patch.object(corpus.time, "sleep"):
                manifest = corpus.acquire(destination)
            audit = corpus.validate_manifest(destination, manifest)
            self.assertTrue(audit["passed"], audit["errors"])
            self.assertEqual({language: 10 for language in corpus.ROOT_TITLES}, audit["documents"])
            self.assertFalse((destination / "manifest.partial.json").exists())

            # A valid final manifest is a strict, zero-network cache boundary.
            with patch.object(corpus, "linked_titles", side_effect=AssertionError("network used")), \
                    patch.object(corpus, "fetch_pages", side_effect=AssertionError("network used")):
                self.assertEqual(manifest, corpus.acquire(destination))

    def test_validator_detects_payload_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with patch.object(corpus, "linked_titles", side_effect=self.fake_links), \
                    patch.object(corpus, "fetch_pages", side_effect=self.fake_pages), \
                    patch.object(corpus.time, "sleep"):
                manifest = corpus.acquire(destination)
            document = manifest["languages"]["en"]["documents"][0]
            (destination / document["relative_path"]).write_text("tampered\n", encoding="utf-8")
            audit = corpus.validate_manifest(destination, manifest)
            self.assertFalse(audit["passed"])
            self.assertTrue(any("mismatch" in error or "bounds" in error for error in audit["errors"]))

    def test_partial_manifest_resumes_only_missing_languages(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with patch.object(corpus, "linked_titles", side_effect=self.fake_links), \
                    patch.object(corpus, "fetch_pages", side_effect=self.fake_pages), \
                    patch.object(corpus.time, "sleep"):
                complete = corpus.acquire(destination)
            partial = {key: value for key, value in complete.items() if key != "manifest_content_hash"}
            partial["languages"] = {"en": complete["languages"]["en"]}
            (destination / "manifest.partial.json").write_text(
                json.dumps(partial, ensure_ascii=False), encoding="utf-8")
            (destination / "manifest.json").unlink()

            link_calls = []

            def links(language, root_title, limit=500):
                link_calls.append(language)
                return self.fake_links(language, root_title, limit)

            with patch.object(corpus, "linked_titles", side_effect=links), \
                    patch.object(corpus, "fetch_pages", side_effect=self.fake_pages), \
                    patch.object(corpus.time, "sleep"):
                resumed = corpus.acquire(destination)
            self.assertNotIn("en", link_calls)
            self.assertEqual({"hi", "te", "sd"}, set(link_calls))
            self.assertTrue(corpus.validate_manifest(destination, resumed)["passed"])

    def test_validator_rejects_path_escape_even_with_rehashed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with patch.object(corpus, "linked_titles", side_effect=self.fake_links), \
                    patch.object(corpus, "fetch_pages", side_effect=self.fake_pages), \
                    patch.object(corpus.time, "sleep"):
                original = corpus.acquire(destination)
            forged = copy.deepcopy(original)
            forged["languages"]["en"]["documents"][0]["relative_path"] = "../outside.txt"
            forged["manifest_content_hash"] = corpus._canonical_hash(forged)
            audit = corpus.validate_manifest(destination, forged)
            self.assertFalse(audit["passed"])
            self.assertTrue(any("escapes corpus" in error for error in audit["errors"]))


if __name__ == "__main__":
    unittest.main()
