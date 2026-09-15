"""Validate the qrels file before it is used to score anything.

qrels are the ground truth every metric stands on, so they deserve the same
treatment as code: automated checks, run every time the file changes.

Checks performed:
  1. every doc_id exists in the curated corpus   (a typo silently becomes a miss)
  2. every query has at least one doc graded 2   (without one, the query cannot
     measure anything — recall@k is 0 for every retriever)
  3. each query_id maps to exactly ONE query text (copy-pasting a judge() call
     and forgetting to edit the text is the classic qrels bug)
  4. no duplicate (query_id, doc_id) pairs
  5. relevance is in {0, 1, 2}; query_class is in {literal, paraphrase, multi-doc}
  6. class coverage against the target mix

Exit code is non-zero when any check fails, so it can gate a build or a commit hook.

Usage (from the repo root, with the project venv):
    python scripts/validate_qrels.py
    python scripts/validate_qrels.py --target 4 4 4     # want 4 of each class
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QRELS = ROOT / "eval" / "qrels.csv"
CURATED = ROOT / "data" / "processed" / "curated.jsonl"


def load_doc_ids() -> set[str]:
    return {json.loads(line)["doc_id"] for line in CURATED.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate eval/qrels.csv")
    ap.add_argument("--target", nargs=3, type=int, metavar=("LIT", "PARA", "MULTI"),
                    default=[4, 4, 4], help="target number of queries per class (default 4 4 4)")
    args = ap.parse_args()

    if not QRELS.exists():
        print(f"✗ Không có {QRELS.relative_to(ROOT)}")
        return 1

    doc_ids = load_doc_ids()
    rows = [r for r in csv.DictReader(QRELS.open(encoding="utf-8"))
            if (r.get("doc_id") or "").strip() and (r.get("relevance") or "").strip()]

    if not rows:
        print("✗ qrels trống (chưa chấm dòng nào).")
        return 1

    errors: list[str] = []
    warnings: list[str] = []

    texts: dict[str, set[str]] = defaultdict(set)
    seen_pairs: set[tuple[str, str]] = set()
    grades_by_query: dict[str, list[int]] = defaultdict(list)
    class_by_query: dict[str, str] = {}

    for i, r in enumerate(rows, start=2):        # line 2 = first data row
        qid = (r.get("query_id") or "").strip()
        qtext = (r.get("query") or "").strip()
        did = (r.get("doc_id") or "").strip()
        rel_raw = (r.get("relevance") or "").strip()
        qclass = (r.get("query_class") or "").strip()

        if not qid or not qtext:
            errors.append(f"dòng {i}: thiếu query_id hoặc query")
            continue
        if did not in doc_ids:
            errors.append(f"dòng {i}: doc_id {did} KHÔNG có trong corpus")
        if rel_raw not in {"0", "1", "2"}:
            errors.append(f"dòng {i}: relevance '{rel_raw}' không hợp lệ (chỉ 0/1/2)")
        if qclass not in {"literal", "paraphrase", "multi-doc"}:
            errors.append(f"dòng {i}: query_class '{qclass}' không hợp lệ")
        if (qid, did) in seen_pairs:
            errors.append(f"dòng {i}: cặp ({qid}, {did}) bị lặp")
        seen_pairs.add((qid, did))

        texts[qid].add(qtext)
        class_by_query.setdefault(qid, qclass)
        if rel_raw in {"0", "1", "2"}:
            grades_by_query[qid].append(int(rel_raw))

    # check 3: one query text per query_id
    for qid, ts in texts.items():
        if len(ts) > 1:
            errors.append(f"{qid}: có {len(ts)} query text khác nhau — copy-paste thiếu sửa? {sorted(ts)}")

    # check: one query text must not be reused by a different query_id
    by_text: dict[str, set[str]] = defaultdict(set)
    for qid, ts in texts.items():
        for t in ts:
            by_text[t].add(qid)
    for t, qids in by_text.items():
        if len(qids) > 1:
            errors.append(f"cùng một câu hỏi bị dùng cho nhiều query_id {sorted(qids)}: '{t[:55]}'")

    # check 2: every query needs at least one doc graded 2
    for qid, rels in grades_by_query.items():
        if 2 not in rels:
            errors.append(f"{qid}: không có bài nào điểm 2 → query này không đo được gì")
        if all(g == 0 for g in rels):
            warnings.append(f"{qid}: toàn điểm 0")

    # check 6: class coverage
    per_class = Counter(class_by_query.values())
    tgt = dict(zip(["literal", "paraphrase", "multi-doc"], args.target))
    for cls, want in tgt.items():
        got = per_class.get(cls, 0)
        if got < want:
            warnings.append(f"lớp {cls}: có {got}/{want} query mong muốn")

    print(f"qrels: {QRELS.relative_to(ROOT)}")
    print(f"  dòng: {len(rows)} · query: {len(texts)}")
    print(f"  theo lớp: {dict(per_class)}")
    print("  theo query:")
    for qid in sorted(grades_by_query):
        g = grades_by_query[qid]
        print(f"    {qid} [{class_by_query[qid]:>10}] {len(g)} bài · điểm "
              f"{sorted(Counter(g).items())} · text='{sorted(texts[qid])[0][:52]}'")

    if warnings:
        print("\n⚠️  CẢNH BÁO:")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print("\n✗ LỖI:")
        for e in errors:
            print(f"  - {e}")
        print(f"\n=> KHÔNG ĐẠT ({len(errors)} lỗi)")
        return 1
    print("\n✅ qrels hợp lệ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
