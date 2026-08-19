import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import acquire_multidoc as corpus  # noqa: E402


class MultiDocumentCorpusTests(unittest.TestCase):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            return b'{"query":{"pages":[]}}'

    @staticmethod
    def fake_links(language, root_title, limit=500):
        return [f"{language}-{root_title}-linked-{index}" for index in range(9)][:limit]

    @staticmethod
    def fake_pages(language, titles):
        result = []
        for title in titles:
            identity = int(hashlib.sha256(f"{language}:{title}".encode()).hexdigest()[:12], 16)
            text = (f"{language} {title} reproducible research document. " * 70)[:2_200]
            result.append({
                "pageid": identity,
                "title": title,
                "revision_id": identity + 1_000_000,
                "parent_revision_id": identity + 999_999,
                "revision_timestamp": "2026-01-01T00:00:00Z",
                "canonical_url": f"https://{language}.wikipedia.org/wiki/{identity}",
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
            self.assertEqual({language: 100 for language in corpus.ROOT_TITLES}, audit["documents"])
            self.assertTrue(all(
                count == corpus.DOCUMENTS_PER_TOPIC
                for counts in audit["topics"].values() for count in counts.values()))
            self.assertFalse((destination / "manifest.partial.json").exists())

            # A valid final manifest is a strict, zero-network cache boundary.
            with patch.object(corpus, "linked_titles", side_effect=AssertionError("network used")), \
                    patch.object(corpus, "fetch_pages", side_effect=AssertionError("network used")):
                self.assertEqual(manifest, corpus.acquire(destination))

    def test_api_uses_post_only_when_encoded_query_is_too_long(self):
        with patch.object(corpus.urllib.request, "urlopen", return_value=self.FakeResponse()) as opened:
            corpus.api("te", {"titles": "short"})
            short_request = opened.call_args.args[0]
            self.assertIsNone(short_request.data)
            self.assertIn("?", short_request.full_url)

            corpus.api("te", {"titles": "తెలుగు" * 2_000})
            long_request = opened.call_args.args[0]
            self.assertIsNotNone(long_request.data)
            self.assertNotIn("?", long_request.full_url)
            self.assertIn(b"titles=", long_request.data)

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

    def test_partial_topic_batch_is_durable_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            calls = 0

            def interrupted_pages(language, titles):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return self.fake_pages(language, titles[:4])
                raise RuntimeError("simulated acquisition interruption")

            with patch.object(corpus, "linked_titles", side_effect=self.fake_links), \
                    patch.object(corpus, "fetch_pages", side_effect=interrupted_pages), \
                    patch.object(corpus.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "simulated acquisition interruption"):
                    corpus.acquire(destination)

            partial = json.loads((destination / "manifest.partial.json").read_text(encoding="utf-8"))
            self.assertEqual(4, len(partial["languages"]["en"]["documents"]))
            self.assertEqual(
                {"country"},
                {document["source_topic"] for document in partial["languages"]["en"]["documents"]},
            )
            self.assertTrue(corpus.validate_manifest(
                destination, partial, require_hash=False, allow_incomplete=True)["passed"])

            with patch.object(corpus, "linked_titles", side_effect=self.fake_links), \
                    patch.object(corpus, "fetch_pages", side_effect=self.fake_pages), \
                    patch.object(corpus.time, "sleep"):
                resumed = corpus.acquire(destination)
            self.assertTrue(corpus.validate_manifest(destination, resumed)["passed"])

    def test_direct_only_policy_upgrade_prunes_linked_topic_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with patch.object(corpus, "linked_titles", side_effect=self.fake_links), \
                    patch.object(corpus, "fetch_pages", side_effect=self.fake_pages), \
                    patch.object(corpus.time, "sleep"):
                complete = corpus.acquire(destination)

            partial = copy.deepcopy(complete)
            partial.pop("manifest_content_hash")
            # Simulate a manifest written before Sindhi sports became
            # direct-only. Its first sports record came from a broad landing
            # page and is valid under that older policy but not the new one.
            partial["direct_link_limits"] = {"music": {"sd": 0}}
            sports = [document for document in partial["languages"]["sd"]["documents"]
                      if document["source_topic"] == "sports"]
            sports[0]["title"] = "off-topic linked geography page"
            sports[0]["source_root_title"] = corpus.TOPIC_ROOTS["sports"]["sd"]
            (destination / "manifest.partial.json").write_text(
                json.dumps(partial, ensure_ascii=False), encoding="utf-8")
            (destination / "manifest.json").unlink()

            with patch.object(corpus, "linked_titles", side_effect=self.fake_links), \
                    patch.object(corpus, "fetch_pages", side_effect=self.fake_pages), \
                    patch.object(corpus.time, "sleep"):
                upgraded = corpus.acquire(destination)

            upgraded_sports = [
                document for document in upgraded["languages"]["sd"]["documents"]
                if document["source_topic"] == "sports"
            ]
            self.assertEqual(corpus.DOCUMENTS_PER_TOPIC, len(upgraded_sports))
            self.assertNotIn("off-topic linked geography page",
                             {document["title"] for document in upgraded_sports})
            self.assertTrue(all(document["title"] == document["source_root_title"]
                                for document in upgraded_sports))
            self.assertTrue(corpus.validate_manifest(destination, upgraded)["passed"])

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

    def test_validator_rejects_duplicate_content_and_malformed_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with patch.object(corpus, "linked_titles", side_effect=self.fake_links), \
                    patch.object(corpus, "fetch_pages", side_effect=self.fake_pages), \
                    patch.object(corpus.time, "sleep"):
                original = corpus.acquire(destination)
            forged = copy.deepcopy(original)
            duplicate = copy.deepcopy(forged["languages"]["en"]["documents"][0])
            duplicate["pageid"] = 999_999_999
            duplicate["revision_id"] = 999_999_999
            duplicate["revision_timestamp"] = None
            duplicate["canonical_url"] = "not-a-url"
            forged["languages"]["en"]["documents"].append(duplicate)
            forged["manifest_content_hash"] = corpus._canonical_hash(forged)
            audit = corpus.validate_manifest(destination, forged)
            self.assertFalse(audit["passed"])
            self.assertTrue(any("document count mismatch" in error for error in audit["errors"]))
            self.assertTrue(any("duplicate payload path" in error for error in audit["errors"]))
            self.assertTrue(any("duplicate payload hash" in error for error in audit["errors"]))
            self.assertTrue(any("revision timestamp" in error for error in audit["errors"]))
            self.assertTrue(any("canonical URL" in error for error in audit["errors"]))


if __name__ == "__main__":
    unittest.main()
