"""Curation: raw corpus -> curated corpus.

The raw layer (data/raw/) keeps everything the crawler found, with provenance.
This script produces the layer the pipeline actually indexes, applying two
filters that are each recorded with a count so the decision stays auditable:

  1. TOPICALITY — keep only documents that belong to one of the AI/ML seed
     categories. Depth-1 category expansion pulled in tangents (a video-game
     article and a sci-fi title both survived into the raw layer), and a corpus
     is only as clean as its worst document.
  2. MINIMUM LENGTH — drop stubs shorter than --min-chars (default 1000). A
     300-character stub cannot answer a question and only adds noise to recall@k.
  3. BLOCKLIST — a handful of titles that sit in a seed category (Wikipedia
     category membership is community-maintained and imperfect) but are not
     about AI/ML at all. Kept explicit and short so the decision is auditable;
     every removal is reported.

Usage (from the repo root, with the project venv):
    python scripts/curate_corpus.py
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from fetch_corpus import SEED_CATEGORIES, list_category, make_session

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CURATED = PROCESSED / "curated.jsonl"
REPORT = PROCESSED / "curation_report.json"

# Manual review of the curated set — off-topic documents that passed both automatic filters.
BLOCKLIST = {
    "A.I. Artificial Intelligence",   # 2001 film
    "DreadClub: Vampire's Verdict",   # video game
    "Chụp ảnh tự sướng 3D",           # 3D selfie topic, not AI/ML
}


def load_raw() -> list[dict]:
    docs = []
    for path in sorted(RAW.glob("*.json")):
        if path.name.startswith("_"):
            continue
        docs.append(json.loads(path.read_text(encoding="utf-8")))
    return docs


def seed_titles(session: requests.Session, delay: float) -> set[str]:
    """Titles that sit directly in a seed category (6 cheap API calls)."""
    titles: set[str] = set()
    for cat in SEED_CATEGORIES:
        members = list_category(session, cat, "page")
        titles.update(members)
        print(f"  {cat:40s} {len(members):4d} bài")
        time.sleep(delay)
    return titles


def main() -> int:
    ap = argparse.ArgumentParser(description="Curate the raw corpus into an indexable set.")
    ap.add_argument("--min-chars", type=int, default=1000, help="drop documents shorter than this (default 1000)")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between API calls")
    args = ap.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    docs = load_raw()
    if not docs:
        print("data/raw/ rỗng — chạy `python scripts/fetch_corpus.py` trước.")
        return 1

    print(f"raw: {len(docs)} bài. Đang lấy danh sách bài thuộc 6 thể loại hạt giống...")
    session = make_session()
    seeds = seed_titles(session, args.delay)

    kept, drop_topic, drop_short, drop_blocked = [], 0, 0, 0
    seen_pageids = set()
    for d in docs:
        if d["pageid"] in seen_pageids:      # safety net against duplicate fetches
            continue
        seen_pageids.add(d["pageid"])
        if d["title"] not in seeds:
            drop_topic += 1
            continue
        if d["chars"] < args.min_chars:
            drop_short += 1
            continue
        if d["title"] in BLOCKLIST:
            drop_blocked += 1
            continue
        kept.append(d)

    kept.sort(key=lambda d: d["title"])
    with CURATED.open("w", encoding="utf-8") as fh:
        for d in kept:
            fh.write(json.dumps({
                "doc_id": str(d["pageid"]),
                "title": d["title"],
                "url": d["url"],
                "revid": d["revid"],
                "chars": d["chars"],
                "text": d["text"],
            }, ensure_ascii=False) + "\n")

    total_chars = sum(d["chars"] for d in kept)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "vi.wikipedia.org (CC BY-SA 4.0)",
        "raw_documents": len(docs),
        "seed_category_documents": len(seeds),
        "dropped_off_topic": drop_topic,
        "dropped_too_short": drop_short,
        "dropped_blocklist": drop_blocked,
        "curated_documents": len(kept),
        "curated_chars": total_chars,
        "min_chars": args.min_chars,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"""
=== CURATION REPORT ===
raw                : {len(docs):4d} bài
thuộc chủ đề AI/ML : {len(seeds):4d} bài  (đúng 6 thể loại hạt giống)
- loại lạc đề      : {drop_topic:4d} bài
- loại quá ngắn    : {drop_short:4d} bài  (< {args.min_chars} ký tự)
- loại theo blocklist: {drop_blocked:2d} bài  (soi thủ công)
= curated          : {len(kept):4d} bài · {total_chars/1e6:.2f} MB text
                      (trung bình {total_chars/max(len(kept),1):,.0f} ký tự/bài)

→ ghi vào {CURATED.relative_to(ROOT)} + {REPORT.relative_to(ROOT)}""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
