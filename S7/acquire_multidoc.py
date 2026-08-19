#!/usr/bin/env python3
"""Acquire a revision-pinned, hashed multilingual Wikipedia pilot corpus."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any


ROOT_TITLES = {"en": "India", "hi": "भारत", "te": "భారతదేశం", "sd": "ڀارت"}
LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "sd": "Sindhi"}
DOCUMENTS_PER_LANGUAGE = 10
MAX_STORED_CHARACTERS = 60_000
MIN_DOCUMENT_CHARACTERS = 1_500
LICENSE = {"name": "CC BY-SA 4.0 / GFDL", "url": "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use"}


def _canonical_hash(manifest: dict[str, Any]) -> str:
    value = {key: item for key, item in manifest.items() if key != "manifest_content_hash"}
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def validate_manifest(destination: Path, manifest: dict[str, Any], *, require_hash: bool = True) -> dict[str, Any]:
    """Validate provenance, paths, UTF-8 payloads and every recorded digest."""
    errors: list[str] = []
    expected_header = {
        "schema": 1,
        "provider": "Wikimedia Foundation",
        "license": LICENSE,
        "documents_per_language": DOCUMENTS_PER_LANGUAGE,
        "min_document_characters": MIN_DOCUMENT_CHARACTERS,
        "max_stored_characters": MAX_STORED_CHARACTERS,
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            errors.append(f"invalid {key}")
    if require_hash and manifest.get("manifest_content_hash") != _canonical_hash(manifest):
        errors.append("manifest content hash mismatch")

    destination = destination.resolve()
    seen: set[tuple[str, int]] = set()
    languages = manifest.get("languages", {})
    for language, root_title in ROOT_TITLES.items():
        entry = languages.get(language)
        if not isinstance(entry, dict):
            errors.append(f"missing language {language}")
            continue
        if entry.get("name") != LANGUAGE_NAMES[language] or entry.get("root_title") != root_title:
            errors.append(f"invalid language metadata {language}")
        documents = entry.get("documents", [])
        if len(documents) < DOCUMENTS_PER_LANGUAGE:
            errors.append(f"insufficient documents {language}")
        for index, document in enumerate(documents):
            label = f"{language}[{index}]"
            required = ("pageid", "title", "revision_id", "revision_timestamp", "canonical_url",
                        "source_characters", "stored_characters", "stored_truncated",
                        "relative_path", "sha256", "bytes")
            if any(key not in document for key in required):
                errors.append(f"missing document metadata {label}")
                continue
            identity = (language, document["pageid"])
            if identity in seen:
                errors.append(f"duplicate page {label}")
            seen.add(identity)
            path = (destination / document["relative_path"]).resolve()
            try:
                path.relative_to(destination)
            except ValueError:
                errors.append(f"path escapes corpus {label}")
                continue
            if not path.is_file():
                errors.append(f"missing payload {label}")
                continue
            payload = path.read_bytes()
            if len(payload) != document["bytes"]:
                errors.append(f"byte count mismatch {label}")
            if hashlib.sha256(payload).hexdigest() != document["sha256"]:
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
    return {"passed": not errors, "errors": errors,
            "documents": {language: len(languages.get(language, {}).get("documents", []))
                          for language in ROOT_TITLES}}


def api(language: str, parameters: dict[str, Any]) -> dict[str, Any]:
    query = {"action": "query", "format": "json", "formatversion": 2, **parameters}
    url = f"https://{language}.wikipedia.org/w/api.php?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(url, headers={"User-Agent": "RKE-V2.1-research/1.0 (reproducible corpus pilot)"})
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


def acquire(destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    final_path = destination / "manifest.json"
    if final_path.exists():
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        audit = validate_manifest(destination, existing)
        if audit["passed"]:
            return existing
    partial_path = destination / "manifest.partial.json"
    manifest: dict[str, Any] = {"schema": 1, "provider": "Wikimedia Foundation",
                                "license": LICENSE, "documents_per_language": DOCUMENTS_PER_LANGUAGE,
                                "min_document_characters": MIN_DOCUMENT_CHARACTERS,
                                "max_stored_characters": MAX_STORED_CHARACTERS, "languages": {}}
    if partial_path.exists():
        cached = json.loads(partial_path.read_text(encoding="utf-8"))
        compatible = all(cached.get(key) == manifest.get(key) for key in
                         ("schema", "provider", "license", "documents_per_language",
                          "min_document_characters", "max_stored_characters"))
        if compatible:
            for language, entry in cached.get("languages", {}).items():
                if language not in ROOT_TITLES:
                    continue
                candidate = {**manifest, "languages": {language: entry}}
                # The full validator expects all languages, so accept this cache
                # only when its own language has no language-scoped errors.
                audit = validate_manifest(destination, candidate, require_hash=False)
                scoped = [error for error in audit["errors"]
                          if language in error or "payload" in error or "page" in error
                          or "document" in error or "count" in error or "UTF-8" in error
                          or "source length" in error]
                if not scoped and audit["documents"].get(language, 0) >= DOCUMENTS_PER_LANGUAGE:
                    manifest["languages"][language] = entry
    for language, root_title in ROOT_TITLES.items():
        if language in manifest["languages"]:
            continue
        language_dir = destination / language; language_dir.mkdir(parents=True, exist_ok=True)
        candidates = [root_title] + [x for x in linked_titles(language, root_title) if x != root_title]
        documents = []
        for start in range(0, len(candidates), 20):
            for page in fetch_pages(language, candidates[start:start + 20]):
                text = page.pop("text")
                path = language_dir / f"{page['pageid']}.txt"
                path.write_text(text + "\n", encoding="utf-8")
                payload = path.read_bytes()
                page.update({"relative_path": str(path.relative_to(destination)),
                             "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)})
                documents.append(page)
                if len(documents) >= DOCUMENTS_PER_LANGUAGE:
                    break
            if len(documents) >= DOCUMENTS_PER_LANGUAGE:
                break
            time.sleep(2.0)
        if len(documents) < DOCUMENTS_PER_LANGUAGE:
            raise RuntimeError(f"only {len(documents)} eligible {language} documents")
        manifest["languages"][language] = {"name": LANGUAGE_NAMES[language], "root_title": root_title,
                                            "documents": documents}
        partial_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                                encoding="utf-8")
    manifest["manifest_content_hash"] = _canonical_hash(manifest)
    audit = validate_manifest(destination, manifest)
    if not audit["passed"]:
        raise RuntimeError("generated corpus failed validation: " + "; ".join(audit["errors"]))
    temporary = destination / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(final_path)
    partial_path.unlink(missing_ok=True)
    return manifest


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "data" / "multidoc"
    value = acquire(output)
    print(json.dumps({"output": str(output), "manifest_hash": value["manifest_content_hash"],
                      "documents": {key: len(item["documents"]) for key, item in value["languages"].items()}},
                     indent=2))
