#!/usr/bin/env python3
# Copyright 2026 Avnish Midha. All rights reserved.
# Author: Avnish Midha
# GitHub: avnishbm
# Purpose: Download faithful Wikipedia India-page corpora for language evaluation.
"""Discover and download faithful Markdown for every language linked from India.

Dependencies: requests, beautifulsoup4, lxml, markdownify
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as markdownify


USER_AGENT = "ERA-V5-S2-language-search/1.0 (educational tokenizer experiment)"
DISCOVERY_URL = "https://en.wikipedia.org/w/api.php"


def faithful_units(text: str) -> int:
    count = 0
    in_run = False
    for char in text:
        word_char = unicodedata.category(char)[0] in {"L", "M", "N"}
        if word_char:
            if not in_run:
                count += 1
            in_run = True
        else:
            in_run = False
            if not char.isspace():
                count += 1
    return count


def request(session: requests.Session, url: str, **kwargs) -> requests.Response:
    for attempt in range(8):
        try:
            response = session.get(url, timeout=(10, 60), **kwargs)
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 2**attempt))
                time.sleep(min(60.0, max(1.0, retry_after)))
                raise requests.HTTPError("temporary HTTP 429")
            if response.status_code >= 500:
                raise requests.HTTPError(f"temporary HTTP {response.status_code}")
            response.raise_for_status()
            return response
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == 7:
                raise
            time.sleep(min(30, 2**attempt))
    raise AssertionError("unreachable")


def discover(session: requests.Session) -> list[dict]:
    """Return English plus every interlanguage link on the English India page."""
    pages = [{
        "code": "en",
        "name": "English",
        "autonym": "English",
        "title": "India",
        "domain": "en.wikipedia.org",
        "article_url": "https://en.wikipedia.org/wiki/India",
    }]
    continuation: dict[str, str] = {}
    while True:
        params = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "redirects": 1,
            "prop": "langlinks",
            "titles": "India",
            "lllimit": "max",
            "llprop": "url|langname|autonym",
            "llinlanguagecode": "en",
            **continuation,
        }
        data = request(session, DISCOVERY_URL, params=params).json()
        query_pages = data.get("query", {}).get("pages", [])
        if not query_pages:
            raise RuntimeError("MediaWiki langlinks response contained no page")
        for link in query_pages[0].get("langlinks", []):
            article_url = link["url"]
            pages.append({
                "code": link["lang"],
                "name": link.get("langname", link["lang"]),
                "autonym": link.get("autonym", link.get("langname", link["lang"])),
                "title": link["title"],
                "domain": urlparse(article_url).netloc,
                "article_url": article_url,
            })
        if "continue" not in data:
            break
        continuation = data["continue"]
    return sorted(pages, key=lambda page: page["code"])


def absolutize_links(soup: BeautifulSoup, domain: str) -> None:
    origin = f"https://{domain}"
    for tag in soup.find_all(["a", "img", "source"]):
        attr = "href" if tag.name == "a" else "src"
        value = tag.get(attr)
        if not value:
            continue
        if value.startswith("//"):
            tag[attr] = "https:" + value
        elif value.startswith("/"):
            tag[attr] = urljoin(origin, value)
        elif value.startswith("./"):
            tag[attr] = urljoin(origin + "/wiki/", value[2:])


def convert_html(html: str, domain: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    body = soup.find("body") or soup
    for tag in body(["script", "style", "meta"]):
        tag.decompose()
    for tag in body.find_all("link"):
        rel = " ".join(tag.get("rel") or [])
        href = tag.get("href") or ""
        if "mw:PageProp/Category" in rel and href:
            tag.replace_with(soup.new_string(f"\nCategory: {href}\n"))
        else:
            tag.decompose()
    absolutize_links(body, domain)
    text = markdownify(str(body), heading_style="ATX", bullets="-", strip=["span"])
    text = text.replace("\xa0", " ")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip() + "\n"


def download_one(
    session: requests.Session,
    page: dict,
    output: Path,
    refresh: bool,
    keep_html: bool,
) -> dict:
    code = page["code"]
    text_path = output / f"{code}.faithful.txt"
    meta_path = output / f"{code}.meta.json"
    if text_path.exists() and meta_path.exists() and not refresh:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    # Current MediaWiki REST route documented at /w/rest.php/v1/page/{title}/html.
    rest_url = (
        f"https://{page['domain']}/w/rest.php/v1/page/"
        f"{quote(page['title'], safe='')}/html"
    )
    response = request(session, rest_url)
    markdown = convert_html(response.text, page["domain"])
    text_path.write_text(markdown, encoding="utf-8")
    (output / f"{code}.faithful.md").write_text(markdown, encoding="utf-8")
    if keep_html:
        (output / f"{code}.raw.html").write_text(response.text, encoding="utf-8")
    meta = {
        **page,
        "source_url": rest_url,
        "variant": "wiki_faithful_markdown",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "characters": len(markdown),
        "faithful_units": faithful_units(markdown),
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "candidate-corpus",
    )
    parser.add_argument("--include", nargs="*", help="Only these language codes")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--limit", type=int, help="Download at most N after filtering")
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--keep-html", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    pages = discover(session)
    (args.output / "languages.json").write_text(
        json.dumps(pages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    include = set(args.include) if args.include else None
    exclude = set(args.exclude)
    pages = [
        page for page in pages
        if (include is None or page["code"] in include) and page["code"] not in exclude
    ]
    if args.limit is not None:
        pages = pages[:args.limit]

    failures = []
    for index, page in enumerate(pages, 1):
        try:
            meta = download_one(session, page, args.output, args.refresh, args.keep_html)
            print(
                f"[{index}/{len(pages)}] {page['code']} {page['name']}: "
                f"{meta['faithful_units']} units",
                flush=True,
            )
        except Exception as exc:  # continue the batch and record actionable failures
            failures.append({"page": page, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(pages)}] {page['code']} FAILED: {exc}", flush=True)
        if index != len(pages) and args.delay:
            time.sleep(args.delay)
    (args.output / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Downloaded/cached {len(pages) - len(failures)}; failures: {len(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
