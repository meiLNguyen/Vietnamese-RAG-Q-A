"""Corpus explorer — inspect the corpus before writing qrels.

Why this exists: qrels must be grounded in the real documents, not in memory.
This tool finds which documents actually discuss a topic, lets you read them,
and gives you the exact doc ids to judge — which is the whole job of building
a ground-truth set.

Design decisions worth knowing:

  * ACCENT-INSENSITIVE — Vietnamese users type "hoc may" far more often than
    "học máy"; a search tool that fails on that is not a search tool.

  * TOKEN OVERLAP, NOT SUBSTRING — an earlier version matched the query as one
    literal string, so "Claude là gì?" found nothing while "Claude" worked.
    The query is now tokenised and Vietnamese stopwords are dropped.

  * IDF + LENGTH NORMALISATION — raw term counts have two classic failure
    modes: common words ("học", "dữ liệu") dominate, and long documents win
    just by being long. Scoring each term by its rarity (IDF) and dividing by
    sqrt(document length) fixes both.

This is deliberately a CRUDE cousin of BM25, not BM25 itself. A validation tool
should be easy to reason about; the real ranker in the pipeline adds term
saturation (k1) and tunable length normalisation (b) on top of these same ideas.
If this search returns weak candidates, that is a signal about retrieval on this
corpus — not necessarily a bug.

Reads the CURATED corpus when it exists (data/processed/curated.jsonl) — qrels
must reference ids from the corpus the pipeline actually indexes — and falls
back to the raw layer otherwise.

Usage (from the repo root, with the project venv):
    python scripts/explore.py --stats
    python scripts/explore.py --titles
    python scripts/explore.py "Học có giám sát là gì?"
    python scripts/explore.py "hoc may" --limit 10
    python scripts/explore.py --doc 12345 --full
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CURATED = ROOT / "data" / "processed" / "curated.jsonl"

# Vietnamese function words: they carry no retrieval signal and would otherwise
# dominate the overlap score of every natural-language question.
STOPWORDS = {
    "la", "gi", "cua", "va", "cac", "mot", "nhung", "co", "khong", "duoc",
    "trong", "tren", "voi", "de", "khi", "nhu", "the", "nao", "ra", "vao",
    "thi", "ma", "o", "nay", "do", "cho", "tu", "den", "boi", "ve", "theo",
    "hay", "hoac", "vi", "nen", "da", "dang", "se", "rat", "cung", "chi",
    "con", "phai", "lam", "sao", "ai", "dau", "bao", "nhieu", "cach", "nguoi",
    "bang", "tai", "giua", "sau", "truoc", "moi", "cai", "no", "minh",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")
TITLE_WEIGHT = 3.0        # a query term in the title is a strong topical signal
SNIPPET_WINDOW = 180


def strip_accents(text: str) -> str:
    """'học máy' -> 'hoc may' (đ/Đ handled separately — they do not decompose)."""
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> list[str]:
    """Accent-folded word tokens, stopwords removed."""
    return [t for t in TOKEN_RE.findall(strip_accents(text.lower())) if t not in STOPWORDS]


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


def build_index(docs: list[dict]) -> dict:
    """Term frequencies per document + document frequency per term."""
    df: dict[str, int] = {}
    per_doc: dict[str, dict] = {}
    total_len = 0
    for d in docs:
        tokens = tokenize(d["text"])
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        title_tokens = set(tokenize(d["title"]))
        per_doc[d["doc_id"]] = {"tf": tf, "title": title_tokens, "len": max(len(tokens), 1)}
        total_len += max(len(tokens), 1)
        for t in set(tf) | title_tokens:
            df[t] = df.get(t, 0) + 1
    return {"df": df, "per_doc": per_doc, "n_docs": len(docs),
            "avg_len": total_len / max(len(docs), 1)}


def search(query: str, docs: list[dict], k: int = 8, index: dict | None = None) -> list[dict]:
    """Rank documents for a query. Returns rows with id, title, match info, snippet."""
    index = index or build_index(docs)
    terms = tokenize(query)
    if not terms:
        return []

    n = index["n_docs"]
    results = []
    for d in docs:
        meta = index["per_doc"].get(d["doc_id"])
        if meta is None:
            continue
        score, matched = 0.0, 0
        for term in terms:
            dfi = index["df"].get(term, 0)
            if not dfi:
                continue
            idf = math.log((n + 1) / (dfi + 0.5)) + 1.0
            body = meta["tf"].get(term, 0)
            in_title = term in meta["title"]
            if not body and not in_title:
                continue
            matched += 1
            # length normalisation: divide by sqrt(doc length / average length)
            norm = math.sqrt(meta["len"] / index["avg_len"])
            score += idf * (body + (TITLE_WEIGHT if in_title else 0)) / norm
        if matched:
            results.append({"doc_id": d["doc_id"], "title": d["title"], "chars": d["chars"],
                            "matched": matched, "score": round(score, 2),
                            "snippet": snippet(d["text"], terms)})

    results.sort(key=lambda r: (-r["matched"], -r["score"]))
    return results[:k]


def snippet(text: str, terms: list[str], width: int = SNIPPET_WINDOW) -> str:
    """Window around the earliest query term actually present in the document."""
    flat = strip_accents(text.lower())
    pos = -1
    for term in terms:
        i = flat.find(term)
        if i >= 0 and (pos < 0 or i < pos):
            pos = i
    if pos < 0:
        return text[:width].replace("\n", " ") + " ..."
    start = max(0, pos - width // 3)
    return ("..." if start else "") + text[start:start + width].replace("\n", " ") + "..."


def main() -> int:
    ap = argparse.ArgumentParser(description="Search and inspect the corpus.")
    ap.add_argument("term", nargs="?", help="query — accented or not, full question or keywords")
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

    terms = tokenize(args.term)
    if not terms:
        print(f"'{args.term}' chỉ gồm từ dừng (stopwords) — thử thêm từ khoá chính.")
        return 1

    results = search(args.term, docs, k=args.limit)
    print(f"truy vấn: '{args.term}'")
    print(f"token dùng để tìm: {terms}")
    print(f"-> {len(results)} kết quả · nguồn: {source}\n")

    for r in results:
        print(f"[{r['doc_id']}] {r['title']}  ({r['matched']}/{len(terms)} token khớp · điểm {r['score']} · {r['chars']:,} ký tự)")
        print(f"    {r['snippet']}\n")
    print("→ `--doc <id> --full` để đọc toàn văn trước khi chấm relevance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
