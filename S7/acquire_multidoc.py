#!/usr/bin/env python3
"""Acquire a revision-pinned, topic-stratified multilingual Wikipedia corpus."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any


LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "sd": "Sindhi"}
TOPIC_ROOTS = {
    "country": {"en": "India", "hi": "भारत", "te": "భారతదేశం", "sd": "ڀارت"},
    "science": {"en": "Science", "hi": "विज्ञान", "te": "విజ్ఞానం", "sd": "سائنس"},
    "history": {"en": "History", "hi": "इतिहास", "te": "చరిత్ర", "sd": "تاريخ"},
    "geography": {"en": "Geography", "hi": "भूगोल", "te": "భూగోళం", "sd": "جاگرافي"},
    "literature": {"en": "Literature", "hi": "साहित्य", "te": "సాహిత్యం", "sd": "ادب"},
    "mathematics": {"en": "Mathematics", "hi": "गणित", "te": "గణితం", "sd": "علم رياضيات"},
    "technology": {"en": "Technology", "hi": "प्रौद्योगिकी", "te": "సాంకేతిక విజ్ఞానం", "sd": "ٽيڪنالاجي"},
    "biology": {"en": "Biology", "hi": "जीव विज्ञान", "te": "జీవ శాస్త్రం", "sd": "حياتيات"},
    "music": {"en": "Music", "hi": "संगीत", "te": "సంగీతం", "sd": "موسيقي"},
    "sports": {"en": "Sport", "hi": "खेल", "te": "క్రీడలు", "sd": "رانديون"},
}
TOPIC_FALLBACK_ROOTS = {
    "science": {"te": ["విజ్ఞానశాస్త్రం"]},
    "history": {"te": ["ప్రపంచ చరిత్ర"]},
    "geography": {"te": ["భౌగోళికం"], "sd": ["پاڪستان جي جاگرافي"]},
    "mathematics": {"sd": ["حساب", "جاميٽري", "انگ", "شماريات", "رياضيات جي تاريخ",
                              "لڪيري آلجبرا", "اطلاقي رياضيات", "امڪان",
                              "انگن جو نظريو", "رياضياتي ثبوت"]},
    # The Sindhi music landing page has substantial prose but no qualifying
    # direct-link neighbourhood.  Use explicit music-domain pages instead of
    # broad search results or performer biographies.
    "music": {"sd": ["ڪلاسيڪي موسيقي", "اليڪٽرانڪ موسيقي", "ھپ ھاپ موسيقي",
                       "راڪ موسيقي", "رگي", "بلوز", "جاز", "گٽار", "ٽرمپيٽ",
                       "وائلن", "سمفني", "چمٽا", "يڪتارو", "تنبورو", "جل ترنگ",
                       "پيانو", "ميوزڪ اسڪول", "فلامينڪو", "ستار", "گانو"]},
    "sports": {"sd": ["ڪرڪيٽ", "بيس بال", "باسڪٽ بال", "سنڌ ڪرڪيٽ ٽيم",
                        "ڪراچي ڪنگز", "نياز اسٽيڊيم", "ايسوسيئيشن فٽبال",
                        "رگبي فٽبال", "نارين جي فٽبال سَڀا",
                        "ايسوسيئيشن فٽبال جي بين الاقوامي فيڊريشن", "هاڪي", "ٽينس",
                        "اولمپڪ رانديون", "گالف"]},
}
# Compatibility name used by callers that need only the language codes.
ROOT_TITLES = TOPIC_ROOTS["country"]
DOCUMENTS_PER_TOPIC = 10
DOCUMENTS_PER_LANGUAGE = DOCUMENTS_PER_TOPIC * len(TOPIC_ROOTS)
DOCUMENT_SPLIT_PER_TOPIC = {"train": 8, "validation": 1, "test": 1}
DEPTH2_ROOT_LIMITS = {"geography": {"sd": 50}, "mathematics": {"sd": 50}}
DIRECT_LINK_LIMITS = {"music": {"sd": 0}, "sports": {"sd": 0}}
SELECTION_ORDER = ("primary root, sorted direct links and sorted depth-2 link neighborhoods; "
                   "then declared fallback roots by the same policy if needed")
MAX_STORED_CHARACTERS = 60_000
MIN_DOCUMENT_CHARACTERS = 1_500
FETCH_BATCH_SIZE = 50
REQUEST_DELAY_SECONDS = 0.25
LICENSE = {"name": "CC BY-SA 4.0 / GFDL", "url": "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use"}


def _canonical_hash(manifest: dict[str, Any]) -> str:
    value = {key: item for key, item in manifest.items() if key != "manifest_content_hash"}
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def validate_manifest(destination: Path, manifest: dict[str, Any], *, require_hash: bool = True,
                      allow_incomplete: bool = False) -> dict[str, Any]:
    """Validate provenance, paths, UTF-8 payloads and every recorded digest."""
    errors: list[str] = []
    expected_header = {
        "schema": 2,
        "provider": "Wikimedia Foundation",
        "license": LICENSE,
        "documents_per_language": DOCUMENTS_PER_LANGUAGE,
        "documents_per_topic": DOCUMENTS_PER_TOPIC,
        "topic_roots": TOPIC_ROOTS,
        "topic_fallback_roots": TOPIC_FALLBACK_ROOTS,
        "document_split_per_topic": DOCUMENT_SPLIT_PER_TOPIC,
        "depth2_root_limits": DEPTH2_ROOT_LIMITS,
        "direct_link_limits": DIRECT_LINK_LIMITS,
        "selection_order": SELECTION_ORDER,
        "min_document_characters": MIN_DOCUMENT_CHARACTERS,
        "max_stored_characters": MAX_STORED_CHARACTERS,
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            errors.append(f"invalid {key}")
    if require_hash and manifest.get("manifest_content_hash") != _canonical_hash(manifest):
        errors.append("manifest content hash mismatch")

    destination = destination.resolve()
    seen_pages: set[tuple[str, int]] = set()
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    languages = manifest.get("languages", {})
    if not isinstance(languages, dict):
        return {"passed": False, "errors": [*errors, "invalid languages"], "documents": {}}
    unexpected_languages = set(languages) - set(ROOT_TITLES)
    if unexpected_languages:
        errors.append("unexpected languages: " + ", ".join(sorted(unexpected_languages)))
    completed_roots = manifest.get("acquisition_completed_roots", [])
    if not isinstance(completed_roots, list):
        errors.append("invalid acquisition completed roots")
        completed_roots = []
    valid_root_records: set[tuple[str, str, str]] = set()
    for record in completed_roots:
        if not isinstance(record, list) or len(record) != 3 or not all(
                isinstance(value, str) for value in record):
            errors.append("invalid acquisition completed root record")
            continue
        language, topic, root_title = record
        valid_roots = ([TOPIC_ROOTS.get(topic, {}).get(language)]
                       + TOPIC_FALLBACK_ROOTS.get(topic, {}).get(language, []))
        if language not in ROOT_TITLES or topic not in TOPIC_ROOTS or root_title not in valid_roots:
            errors.append("unknown acquisition completed root")
        key = (language, topic, root_title)
        if key in valid_root_records:
            errors.append("duplicate acquisition completed root")
        valid_root_records.add(key)
    topic_counts: dict[str, dict[str, int]] = {}
    for language in ROOT_TITLES:
        entry = languages.get(language)
        if not isinstance(entry, dict):
            if not allow_incomplete:
                errors.append(f"missing language {language}")
            continue
        if entry.get("name") != LANGUAGE_NAMES[language]:
            errors.append(f"invalid language metadata {language}")
        documents = entry.get("documents", [])
        if not isinstance(documents, list):
            errors.append(f"invalid documents {language}")
            continue
        if ((not allow_incomplete and len(documents) != DOCUMENTS_PER_LANGUAGE)
                or (allow_incomplete and len(documents) > DOCUMENTS_PER_LANGUAGE)):
            errors.append(f"document count mismatch {language}")
        counts = {topic: 0 for topic in TOPIC_ROOTS}
        for index, document in enumerate(documents):
            label = f"{language}[{index}]"
            if not isinstance(document, dict):
                errors.append(f"invalid document metadata {label}")
                continue
            required = ("pageid", "title", "revision_id", "revision_timestamp", "canonical_url",
                        "source_topic", "source_root_title",
                        "source_characters", "stored_characters", "stored_truncated",
                        "relative_path", "sha256", "bytes")
            if any(key not in document for key in required):
                errors.append(f"missing document metadata {label}")
                continue
            integers = ("pageid", "revision_id", "source_characters", "stored_characters", "bytes")
            if any(not isinstance(document[key], int) or isinstance(document[key], bool)
                   for key in integers):
                errors.append(f"invalid integer metadata {label}")
                continue
            if document["pageid"] <= 0 or document["revision_id"] <= 0 or document["bytes"] <= 0:
                errors.append(f"non-positive identity or byte count {label}")
            if not isinstance(document["title"], str) or not document["title"].strip():
                errors.append(f"invalid title {label}")
            topic = document["source_topic"]
            if topic not in TOPIC_ROOTS:
                errors.append(f"invalid source topic {label}")
            elif document["source_root_title"] not in (
                    [TOPIC_ROOTS[topic][language]]
                    + TOPIC_FALLBACK_ROOTS.get(topic, {}).get(language, [])):
                errors.append(f"invalid source root {label}")
            else:
                counts[topic] += 1
            if not isinstance(document["stored_truncated"], bool):
                errors.append(f"invalid truncation flag {label}")
            timestamp = document["revision_timestamp"]
            try:
                if not isinstance(timestamp, str):
                    raise ValueError
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                errors.append(f"invalid revision timestamp {label}")
            url = document["canonical_url"]
            parsed_url = urllib.parse.urlparse(url) if isinstance(url, str) else None
            expected_host = f"{language}.wikipedia.org"
            if parsed_url is None or parsed_url.scheme != "https" or parsed_url.hostname != expected_host:
                errors.append(f"invalid canonical URL {label}")
            digest = document["sha256"]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)):
                errors.append(f"invalid payload digest {label}")
                continue
            identity = (language, document["pageid"])
            if identity in seen_pages:
                errors.append(f"duplicate page {label}")
            seen_pages.add(identity)
            relative_path = document["relative_path"]
            if not isinstance(relative_path, str) or not relative_path:
                errors.append(f"invalid relative path {label}")
                continue
            path = (destination / relative_path).resolve()
            try:
                path.relative_to(destination)
            except ValueError:
                errors.append(f"path escapes corpus {label}")
                continue
            canonical_path = str(path)
            if canonical_path in seen_paths:
                errors.append(f"duplicate payload path {label}")
            seen_paths.add(canonical_path)
            if digest in seen_hashes:
                errors.append(f"duplicate payload hash {label}")
            seen_hashes.add(digest)
            if not path.is_file():
                errors.append(f"missing payload {label}")
                continue
            payload = path.read_bytes()
            if len(payload) != document["bytes"]:
                errors.append(f"byte count mismatch {label}")
            if hashlib.sha256(payload).hexdigest() != digest:
                errors.append(f"payload hash mismatch {label}")
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError:
                errors.append(f"invalid UTF-8 {label}")
                continue
            stored = text[:-1] if text.endswith("\n") else text
            if len(stored) != document["stored_characters"]:
                errors.append(f"character count mismatch {label}")
            if not (MIN_DOCUMENT_CHARACTERS <= len(stored) <= MAX_STORED_CHARACTERS):
                errors.append(f"document length out of bounds {label}")
            if document["source_characters"] < document["stored_characters"]:
                errors.append(f"invalid source length {label}")
            expected_truncated = document["source_characters"] > document["stored_characters"]
            if document["stored_truncated"] != expected_truncated:
                errors.append(f"truncation flag mismatch {label}")
        topic_counts[language] = counts
        for topic, count in counts.items():
            if ((not allow_incomplete and count != DOCUMENTS_PER_TOPIC)
                    or (allow_incomplete and count > DOCUMENTS_PER_TOPIC)):
                errors.append(f"topic document count mismatch {language}:{topic}")
    return {"passed": not errors, "errors": errors,
            "documents": {language: len(languages.get(language, {}).get("documents", []))
                          for language in ROOT_TITLES},
            "topics": topic_counts}


def api(language: str, parameters: dict[str, Any]) -> dict[str, Any]:
    query = {"action": "query", "format": "json", "formatversion": 2, **parameters}
    endpoint = f"https://{language}.wikipedia.org/w/api.php"
    encoded_query = urllib.parse.urlencode(query).encode("ascii")
    headers = {"User-Agent": "RKE-V2.1-research/1.0 (reproducible corpus pilot)"}
    if len(endpoint) + 1 + len(encoded_query) <= 6_000:
        request = urllib.request.Request(
            f"{endpoint}?{encoded_query.decode('ascii')}", headers=headers)
    else:
        # Large Unicode title batches can exceed proxy/server request-line
        # limits. MediaWiki's read API accepts the identical form as POST.
        request = urllib.request.Request(
            endpoint, data=encoded_query,
            headers={**headers, "Content-Type": "application/x-www-form-urlencoded"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 5:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = min(float(retry_after) if retry_after else 2 ** (attempt + 1), 45.0)
            time.sleep(delay)
    raise RuntimeError("unreachable retry loop")


def linked_titles(language: str, root_title: str, limit: int = 500) -> list[str]:
    titles, continuation = [], {}
    while len(titles) < limit:
        result = api(language, {"prop": "links", "titles": root_title, "plnamespace": 0,
                                "pllimit": "max", **continuation})
        pages = result.get("query", {}).get("pages", [])
        if pages:
            titles.extend(link["title"] for link in pages[0].get("links", []))
        if "continue" not in result:
            break
        continuation = result["continue"]
    return sorted(set(titles))[:limit]


def fetch_pages(language: str, titles: list[str]) -> list[dict[str, Any]]:
    result = api(language, {"prop": "extracts|revisions|info", "titles": "|".join(titles), "explaintext": 1,
                            "rvprop": "ids|timestamp", "inprop": "url"})
    pages = result.get("query", {}).get("pages", [])
    output = []
    for page in pages:
        if page.get("missing"):
            continue
        text = page.get("extract", "").strip(); revisions = page.get("revisions", [])
        if len(text) < MIN_DOCUMENT_CHARACTERS or not revisions:
            continue
        revision = revisions[0]; stored = text[:MAX_STORED_CHARACTERS]
        output.append({"pageid": page["pageid"], "title": page["title"], "revision_id": revision["revid"],
                       "parent_revision_id": revision.get("parentid"), "revision_timestamp": revision["timestamp"],
                       "canonical_url": page.get("canonicalurl") or page.get("fullurl"),
                       "source_characters": len(text), "stored_characters": len(stored),
                       "stored_truncated": len(stored) < len(text), "text": stored})
    return sorted(output, key=lambda x: x["title"])


def _manifest_skeleton() -> dict[str, Any]:
    return {"schema": 2, "provider": "Wikimedia Foundation", "license": LICENSE,
            "documents_per_language": DOCUMENTS_PER_LANGUAGE,
            "documents_per_topic": DOCUMENTS_PER_TOPIC,
            "topic_roots": TOPIC_ROOTS,
            "topic_fallback_roots": TOPIC_FALLBACK_ROOTS,
            "document_split_per_topic": DOCUMENT_SPLIT_PER_TOPIC,
            "depth2_root_limits": DEPTH2_ROOT_LIMITS,
            "direct_link_limits": DIRECT_LINK_LIMITS,
            "selection_order": SELECTION_ORDER,
            "min_document_characters": MIN_DOCUMENT_CHARACTERS,
            "max_stored_characters": MAX_STORED_CHARACTERS,
            "acquisition_completed_roots": [], "languages": {}}


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def acquire(destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    final_path = destination / "manifest.json"
    if final_path.exists():
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        audit = validate_manifest(destination, existing)
        if audit["passed"]:
            return existing
    partial_path = destination / "manifest.partial.json"
    manifest = _manifest_skeleton()
    if partial_path.exists():
        cached = json.loads(partial_path.read_text(encoding="utf-8"))
        if cached.get("selection_order") in {
                "root then lexicographically sorted namespace-0 links",
                "root, sorted direct links, then sorted depth-2 link neighborhoods if needed"}:
            # Direct-link selections are a valid prefix of the depth-2 policy;
            # upgrade an interrupted schema-v2 acquisition without refetching.
            cached["selection_order"] = SELECTION_ORDER
            cached["topic_fallback_roots"] = TOPIC_FALLBACK_ROOTS
        cached_fallbacks = cached.get("topic_fallback_roots", {})
        fallback_addition_only = (
            isinstance(cached_fallbacks, dict)
            and all(isinstance(by_language, dict) for by_language in cached_fallbacks.values())
            and all(isinstance(roots, list)
                    and set(roots) <= set(TOPIC_FALLBACK_ROOTS.get(topic, {}).get(language, []))
                    for topic, by_language in cached_fallbacks.items()
                    for language, roots in by_language.items()))
        if fallback_addition_only:
            # Declaring a new fallback does not invalidate pages already tied
            # to an earlier primary/fallback root.
            cached["topic_fallback_roots"] = TOPIC_FALLBACK_ROOTS
        if "depth2_root_limits" not in cached:
            sindhi_geography_documents = [
                document
                for document in cached.get("languages", {}).get("sd", {}).get("documents", [])
                if document.get("source_topic") == "geography"]
            if not sindhi_geography_documents:
                cached["depth2_root_limits"] = DEPTH2_ROOT_LIMITS
        else:
            cached_limits = cached.get("depth2_root_limits", {})
            limit_addition_only = (
                isinstance(cached_limits, dict)
                and all(isinstance(by_language, dict) for by_language in cached_limits.values())
                and all(DEPTH2_ROOT_LIMITS.get(topic, {}).get(language) == value
                        for topic, by_language in cached_limits.items()
                        for language, value in by_language.items()))
            if limit_addition_only:
                cached["depth2_root_limits"] = DEPTH2_ROOT_LIMITS
        cached_direct_limits = cached.get("direct_link_limits", {})
        direct_limit_addition_only = (
            isinstance(cached_direct_limits, dict)
            and all(isinstance(by_language, dict)
                    for by_language in cached_direct_limits.values())
            and all(DIRECT_LINK_LIMITS.get(topic, {}).get(language) == value
                    for topic, by_language in cached_direct_limits.items()
                    for language, value in by_language.items()))
        if direct_limit_addition_only:
            # A newly direct-only stratum invalidates linked pages accepted by
            # an interrupted older policy. Keep exact roots and reacquire the
            # rest; payload files can remain as an unreferenced local cache.
            newly_direct_only = {
                (language, topic)
                for topic, by_language in DIRECT_LINK_LIMITS.items()
                for language, limit in by_language.items()
                if limit == 0 and cached_direct_limits.get(topic, {}).get(language) != 0
            }
            for language, topic in newly_direct_only:
                entry = cached.get("languages", {}).get(language)
                if isinstance(entry, dict) and isinstance(entry.get("documents"), list):
                    entry["documents"] = [
                        document for document in entry["documents"]
                        if (document.get("source_topic") != topic
                            or document.get("title") == document.get("source_root_title"))
                    ]
            cached["direct_link_limits"] = DIRECT_LINK_LIMITS
        cached.setdefault("acquisition_completed_roots", [])
        compatible = all(cached.get(key) == manifest.get(key) for key in manifest
                         if key not in {"languages", "acquisition_completed_roots"})
        if compatible and validate_manifest(destination, cached, require_hash=False,
                                            allow_incomplete=True)["passed"]:
            manifest["languages"] = cached.get("languages", {})
            cached_languages = set(manifest["languages"])
            manifest["acquisition_completed_roots"] = [
                record for record in cached.get("acquisition_completed_roots", [])
                if record[0] in cached_languages
            ]

    seen_pages = {(language, document["pageid"])
                  for language, entry in manifest["languages"].items()
                  for document in entry["documents"]}
    seen_hashes = {document["sha256"] for entry in manifest["languages"].values()
                   for document in entry["documents"]}
    completed_search_roots = {
        tuple(record) for record in manifest["acquisition_completed_roots"]}
    topic_order = {topic: index for index, topic in enumerate(TOPIC_ROOTS)}
    for language in ROOT_TITLES:
        language_dir = destination / language; language_dir.mkdir(parents=True, exist_ok=True)
        entry = manifest["languages"].setdefault(
            language, {"name": LANGUAGE_NAMES[language], "documents": []})
        documents = entry["documents"]
        for topic, titles_by_language in TOPIC_ROOTS.items():
            existing_topic = [document for document in documents if document["source_topic"] == topic]
            if len(existing_topic) == DOCUMENTS_PER_TOPIC:
                continue
            root_title = titles_by_language[language]
            attempted_titles: set[str] = set()

            def consume(candidate_titles: list[str], source_root_title: str) -> None:
                fresh = [title for title in candidate_titles if title not in attempted_titles]
                attempted_titles.update(fresh)
                for start in range(0, len(fresh), FETCH_BATCH_SIZE):
                    before = len(existing_topic)
                    for page in fetch_pages(language, fresh[start:start + FETCH_BATCH_SIZE]):
                        identity = (language, page["pageid"])
                        if identity in seen_pages:
                            continue
                        text = page.pop("text")
                        payload = (text + "\n").encode("utf-8")
                        digest = hashlib.sha256(payload).hexdigest()
                        if digest in seen_hashes:
                            continue
                        path = language_dir / f"{page['pageid']}.txt"
                        path.write_bytes(payload)
                        page.update({"source_topic": topic, "source_root_title": source_root_title,
                                     "relative_path": str(path.relative_to(destination)),
                                     "sha256": digest, "bytes": len(payload)})
                        documents.append(page); existing_topic.append(page)
                        seen_pages.add(identity); seen_hashes.add(digest)
                        if len(existing_topic) >= DOCUMENTS_PER_TOPIC:
                            _write_json_atomic(partial_path, manifest)
                            return
                    if len(existing_topic) != before:
                        # An interrupted run resumes from every durable batch,
                        # including a partially completed topic.
                        _write_json_atomic(partial_path, manifest)
                    if len(existing_topic) < DOCUMENTS_PER_TOPIC:
                        time.sleep(REQUEST_DELAY_SECONDS)

            def exhaust_root(search_root: str) -> None:
                search_key = (language, topic, search_root)
                if search_key in completed_search_roots:
                    return
                direct_limit = DIRECT_LINK_LIMITS.get(topic, {}).get(language, 500)
                candidates = [search_root] + [title for title in linked_titles(
                    language, search_root, limit=direct_limit)
                                               if title != search_root]
                consume(candidates, search_root)
                if len(existing_topic) < DOCUMENTS_PER_TOPIC:
                    # Smaller editions can have sparse topic landing pages.
                    # Expand every direct link in deterministic order while
                    # retaining the actual root used as page provenance.
                    root_limit = DEPTH2_ROOT_LIMITS.get(topic, {}).get(language)
                    expansion_roots = candidates[1:] if root_limit is None else candidates[1:1 + root_limit]
                    for expansion_root in expansion_roots:
                        consume(linked_titles(language, expansion_root, limit=200), search_root)
                        if len(existing_topic) >= DOCUMENTS_PER_TOPIC:
                            break
                if len(existing_topic) < DOCUMENTS_PER_TOPIC:
                    completed_search_roots.add(search_key)
                    manifest["acquisition_completed_roots"] = [
                        list(key) for key in sorted(completed_search_roots)]
                    _write_json_atomic(partial_path, manifest)

            exhaust_root(root_title)
            for fallback_root in TOPIC_FALLBACK_ROOTS.get(topic, {}).get(language, []):
                if len(existing_topic) >= DOCUMENTS_PER_TOPIC:
                    break
                exhaust_root(fallback_root)
            if len(existing_topic) < DOCUMENTS_PER_TOPIC:
                raise RuntimeError(
                    f"only {len(existing_topic)} eligible {language}:{topic} documents")
            documents.sort(key=lambda document: (
                topic_order[document["source_topic"]], document["title"], document["pageid"]))
            _write_json_atomic(partial_path, manifest)
    manifest["manifest_content_hash"] = _canonical_hash(manifest)
    audit = validate_manifest(destination, manifest)
    if not audit["passed"]:
        raise RuntimeError("generated corpus failed validation: " + "; ".join(audit["errors"]))
    _write_json_atomic(final_path, manifest)
    partial_path.unlink(missing_ok=True)
    return manifest


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "data" / "multidoc"
    value = acquire(output)
    print(json.dumps({"output": str(output), "manifest_hash": value["manifest_content_hash"],
                      "documents": {key: len(item["documents"]) for key, item in value["languages"].items()}},
                     indent=2))
