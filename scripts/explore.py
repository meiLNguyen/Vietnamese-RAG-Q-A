"""Corpus explorer — inspect the corpus before writing qrels.

Why this exists: qrels must be grounded in the real documents, not in memory.
This tool finds which documents actually discuss a topic, lets you read them,
and gives you the exact doc ids to judge — which is the whole job of building
a ground-truth set.

Accent-insensitive search is included on purpose: Vietnamese users type
"hoc may" far more often than "học máy", and a search tool that fails on that
is not a search tool.

Reads the CURATED corpus when it exists (data/processed/curated.jsonl) — qrels
must reference ids from the corpus the pipeline actually indexes — and falls
back to the raw layer otherwise.

Usage (from the repo root, with the project venv):
    python scripts/explore.py --stats
    python scripts/explore.py --titles
    python scripts/explore.py "học có giám sát"
    python scripts/explore.py "hoc may" --limit 10
    python scripts/explore.py --doc 12345 --full
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CURATED = ROOT / "data" / "processed" / "curated.jsonl"


def strip_accents(text: str) -> str:
    """'học máy' -> 'hoc may' (đ/Đ handled separately — they do not decompose)."""
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def load_docs() -> tuple[list[dict], str]:
    """Return (docs, source_label). Prefers the curated corpus."""
    if CURATED.exists():
        docs = [json.loads(line) for line in CURATED.read_text(encoding="utf-8").splitlines() if line.strip()]
        return docs, f"curated ({CURATED.relative_to(ROOT)})"

    docs = []
    for path in sorted(RAW.glob("*.json")):
        if path.name.startswith("_"):
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        d["doc_id"] = str(d["pageid"])
        docs.append(d)
    return docs, f"raw ({RAW.relative_to(ROOT)}) — chạy curate_corpus.py để có bản đã lọc"


def snippet(text: str, term: str, width: int = 160) -> str:
    """Return a window of text around the first accent-insensitive match."""
    idx = strip_accents(text.lower()).find(strip_accents(term.lower()))
    if idx < 0:
        return text[:width].replace("\n", " ") + " ..."
    start = max(0, idx - width // 3)
    return ("..." if start else "") + text[start:start + width].replace("\n", " ") + "..."


def main() -> int:
    ap = argparse.ArgumentParser(description="Search and inspect the corpus.")
    ap.add_argument("term", nargs="?", help="search term (accent-insensitive)")
    ap.add_argument("--doc", help="print one document by doc_id")
    ap.add_argument("--stats", action="store_true", help="corpus statistics")
    ap.add_argument("--titles", action="store_true", help="list every document id + title")
    ap.add_argument("--limit", type=int, default=5, help="max results to show (default 5)")
    ap.add_argument("--full", action="store_true", help="with --doc, print the whole text")
    args = ap.parse_args()

    docs, source = load_docs()
    if not docs:
        print("Corpus rỗng — chạy `python scripts/fetch_corpus.py` trước.")
        return 1

    if args.stats:
        total = sum(d["chars"] for d in docs)
        lens = sorted(d["chars"] for d in docs)
        print(f"nguồn         : {source}")
        print(f"số bài        : {len(docs)}")
        print(f"tổng text     : {total/1e6:.2f} MB")
        print(f"dài trung bình: {total/len(docs):,.0f} ký tự")
        print(f"ngắn nhất     : {lens[0]:,} | trung vị: {lens[len(lens)//2]:,} | dài nhất: {lens[-1]:,}")
        print("\n10 bài dài nhất:")
        for d in sorted(docs, key=lambda x: -x["chars"])[:10]:
            print(f"  {d['doc_id']:>9}  {d['chars']:>7,}  {d['title']}")
        return 0

    if args.titles:
        print(f"nguồn: {source} — {len(docs)} bài\n")
        for d in sorted(docs, key=lambda x: x["title"]):
            print(f"  {d['doc_id']:>9}  {d['chars']:>7,}  {d['title']}")
        return 0

    if args.doc:
        for d in docs:
            if d["doc_id"] == str(args.doc):
                print(f"# {d['title']}\n{d['url']}\ndoc_id={d['doc_id']} revid={d.get('revid')} "
                      f"chars={d['chars']}\n")
                print(d["text"] if args.full else d["text"][:2000] + "\n[... dùng --full để in hết]")
                return 0
        print(f"Không tìm thấy doc_id {args.doc}")
        return 1

    if not args.term:
        ap.print_help()
        return 1

    # Rank: title hits first, then raw occurrence count, then shorter docs (denser topic signal)
    pattern = re.compile(re.escape(strip_accents(args.term.lower())))
    hits = []
    for d in docs:
        count = len(pattern.findall(strip_accents(d["text"].lower())))
        title_hit = bool(pattern.search(strip_accents(d["title"].lower())))
        if count or title_hit:
            hits.append((count + (10 if title_hit else 0), count, d))
    hits.sort(key=lambda x: (-x[0], x[2]["chars"]))

    print(f"'{args.term}' -> {len(hits)} bài khớp trong {len(docs)} (hiện {min(args.limit, len(hits))})")
    print(f"nguồn: {source}\n")
    for score, count, d in hits[:args.limit]:
        print(f"[{d['doc_id']}] {d['title']}  ({count} lần · {d['chars']:,} ký tự)")
        print(f"    {snippet(d['text'], args.term)}\n")
    print("→ `--doc <id> --full` để đọc toàn văn trước khi chấm relevance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
