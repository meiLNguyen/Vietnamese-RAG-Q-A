"""Polite, resumable fetcher for the Vietnamese RAG corpus.

Source : vi.wikipedia.org (MediaWiki API) — license CC BY-SA 4.0
Design constraints (see SCOPE.md):
  * never hammer the API  -> fixed delay between requests + exponential backoff on 429/5xx
  * resumable            -> already-fetched articles are skipped, so an interrupted run can continue
  * provenance           -> every document keeps its pageid, revision id and source URL,
                            so the corpus can be audited and re-fetched

Usage (from the repo root, with the project venv):
    python scripts/fetch_corpus.py --delay 1.5 --depth 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

API = "https://vi.wikipedia.org/w/api.php"
# Wikimedia requires a descriptive User-Agent; anonymous generic agents get throttled harder.
USER_AGENT = "rag-qa-vietnamese/0.1 (educational RAG project; https://github.com/meiLNguyen)"

SEED_CATEGORIES = [
    "Thể loại:Trí tuệ nhân tạo",
    "Thể loại:Học máy",
    "Thể loại:Học sâu",
    "Thể loại:Xử lý ngôn ngữ tự nhiên",
    "Thể loại:Thị giác máy tính",
    "Thể loại:Trí tuệ nhân tạo tạo sinh",
]

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CATALOG = RAW / "_catalog.json"


class RateLimited(Exception):
    """Raised when the API keeps returning 429/5xx after all retries."""


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s


def api_get(session: requests.Session, params: dict, retries: int = 5, backoff: float = 5.0) -> dict:
    """GET the MediaWiki API with exponential backoff. Respects Retry-After on 429."""
    params = {**params, "format": "json", "formatversion": 2}
    delay = backoff
    for attempt in range(1, retries + 1):
        try:
            r = session.get(API, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("Retry-After", delay))
                print(f"    [{r.status_code}] backoff {wait:.0f}s (attempt {attempt}/{retries})", flush=True)
                time.sleep(wait)
                delay = min(delay * 2, 120)
                continue
            r.raise_for_status()
        except requests.RequestException as exc:
            print(f"    [net] {type(exc).__name__} — retry in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 120)
    raise RateLimited(f"API stayed unavailable after {retries} attempts")


def list_category(session: requests.Session, category: str, kind: str = "page") -> list[str]:
    """Return every member title of a category (paginated via cmcontinue)."""
    titles, cont = [], None
    while True:
        params = {
            "action": "query", "list": "categorymembers", "cmtitle": category,
            "cmlimit": 500, "cmtype": kind,
        }
        if cont:
            params["cmcontinue"] = cont
        data = api_get(session, params)
        titles += [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            return titles


def discover(session: requests.Session, categories: list[str], depth: int, delay: float) -> list[str]:
    """Breadth-first walk of the category tree; returns a de-duplicated, sorted title list."""
    seen_cats, titles, frontier = set(), set(), list(categories)
    for level in range(depth + 1):
        next_frontier = []
        for cat in frontier:
            if cat in seen_cats:
                continue
            seen_cats.add(cat)
            pages = list_category(session, cat, "page")
            titles.update(pages)
            print(f"  [depth {level}] {cat} -> {len(pages)} bài (tổng {len(titles)})", flush=True)
            time.sleep(delay)
            if level < depth:
                subs = list_category(session, cat, "subcat")
                next_frontier += subs
                time.sleep(delay)
        frontier = next_frontier
    return sorted(titles)


def fetch_article(session: requests.Session, title: str) -> dict | None:
    """Fetch the plain-text extract + provenance metadata for one article."""
    data = api_get(session, {
        "action": "query", "prop": "extracts|info", "explaintext": 1, "exlimit": 1,
        "inprop": "url", "redirects": 1, "titles": title,
    })
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None
    page = pages[0]
    text = (page.get("extract") or "").strip()
    if page.get("missing") or not text:
        return None
    return {
        "title": page["title"],
        "pageid": page["pageid"],
        "url": page.get("fullurl", ""),
        "revid": page.get("lastrevid"),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "license": "CC BY-SA 4.0",
        "source": "vi.wikipedia.org",
        "chars": len(text),
        "text": text,
    }


def load_catalog() -> dict:
    if CATALOG.exists():
        return json.loads(CATALOG.read_text(encoding="utf-8"))
    return {"generated_at": None, "source": "vi.wikipedia.org", "license": "CC BY-SA 4.0", "docs": {}}


def save_catalog(catalog: dict) -> None:
    CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch the Vietnamese Wikipedia AI/ML corpus.")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between API calls (default 1.5)")
    ap.add_argument("--depth", type=int, default=1, help="subcategory recursion depth (default 1)")
    ap.add_argument("--max", type=int, default=0, help="stop after N articles (0 = all)")
    ap.add_argument("--min-chars", type=int, default=0, help="skip articles shorter than N characters")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    session = make_session()
    catalog = load_catalog()

    # 1. Discover the article list (cached in the catalog so re-runs are cheap)
    if catalog["docs"]:
        titles = sorted(catalog["docs"].keys())
        print(f"Catalog found: {len(titles)} bài đã biết — bỏ qua bước discover.", flush=True)
    else:
        print("Đang discover bài từ category tree...", flush=True)
        titles = discover(session, SEED_CATEGORIES, args.depth, args.delay)
        catalog["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        catalog["categories"] = SEED_CATEGORIES
        for t in titles:
            catalog["docs"].setdefault(t, {})
        save_catalog(catalog)
        print(f"→ {len(titles)} bài. Bắt đầu fetch.\n", flush=True)

    # 2. Fetch each article, skipping anything already on disk (resume support)
    done = skipped = failed = short = 0
    started = time.time()
    for i, title in enumerate(titles, 1):
        entry = catalog["docs"].get(title, {})
        if entry.get("status") == "ok" and entry.get("pageid"):
            skipped += 1
            continue

        if args.max and done >= args.max:
            print(f"\nĐạt giới hạn --max {args.max}, dừng.", flush=True)
            break

        try:
            doc = fetch_article(session, title)
        except RateLimited as exc:
            print(f"\n⚠️  {exc} — đã lưu catalog, chạy lại sau để tiếp tục.", flush=True)
            save_catalog(catalog)
            return 2

        if doc is None or doc["chars"] < args.min_chars:
            catalog["docs"][title] = {"status": "skipped", "reason": "empty or too short"}
            short += 1
        else:
            path = RAW / f"{doc['pageid']}.json"
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
            catalog["docs"][title] = {
                "status": "ok", "pageid": doc["pageid"], "file": path.name,
                "chars": doc["chars"], "url": doc["url"], "fetched_at": doc["fetched_at"],
            }
            done += 1
            if done % 10 == 0:
                save_catalog(catalog)
                rate = done / max(time.time() - started, 1)
                print(f"  [{i}/{len(titles)}] {done} bài · {rate:.2f} bài/s · mới nhất: {doc['title'][:40]}", flush=True)

        time.sleep(args.delay)

    save_catalog(catalog)
    total_chars = sum(d.get("chars", 0) for d in catalog["docs"].values() if d.get("status") == "ok")
    print(f"\n=== XONG ===\nbài mới: {done} · bỏ qua (đã có): {skipped} · quá ngắn/rỗng: {short} · "
          f"lỗi: {failed}\ntổng text: {total_chars/1e6:.2f} MB · catalog: {CATALOG.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
