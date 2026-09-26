from __future__ import annotations
import json
import re
import statistics
from pathlib import Path
from collections import Counter

Root = Path(__file__).resolve().parents[1]
CURATED = Root / "data" / "processed" / "curated.jsonl"
CHUNKS = Root / "data" / "processed" / "chunks.jsonl"

CHUNK_SIZE = 600     # characters per chunk, chosen from the measured size distribution
OVERLAP = 80         # ký tự gối nhau giữa 2 chunk liền kề
MIN_CHARS = 100      # chunk ngắn hơn mức này thì bỏ (rác)
THRESHOLD = 0.4       # tỉ lệ token 1 ký tự > mức này thì coi là rác

def snap_to_sentence(text: str, target: int, window: int = 120) -> int:
    lo = max(0, target - window)
    seg = text[lo:target]
    k = max(seg.rfind("."), seg.rfind("!"), seg.rfind("?"), seg.rfind("\n"))
    if k != -1:
        return k + lo + 1
    return target

def chunk_document(doc: dict) -> list[dict]:
    text = doc["text"]
    doc_id = doc["doc_id"]
    pos = 0
    index = 0
    chunks = []
    dropped = 0
    text_len = len(text)
    while pos < text_len:
        target_pos = pos + CHUNK_SIZE                               #điểm cắt dự kiến
        end_pos = snap_to_sentence(text, target_pos)    #nắn lại điểm cắt dự kiến sao cho trùng với câu hoàn chỉnh
        piece = text[pos:end_pos].strip()                           #lấy đoạn text từ pos đến end_pos
        if len(piece) >= MIN_CHARS:
            if single_char_ratio(piece) > THRESHOLD:
                dropped += 1
            else:
                chunks.append({
                    "chunk_id": f"{doc_id}:{index:03d}",
                    "doc_id": doc_id,
                    "title": doc["title"],
                    "chunk_index": index,
                    "start_char": pos,
                    "chars": len(piece),
                    "text": piece
                })
                index += 1
        pos = max(pos+1, end_pos - OVERLAP)  # gối nhau
        # bỏ khoảng trắng đầu
        next_space = text.find(" ", pos)
        if next_space != -1 and next_space - pos < 20:
            pos = next_space + 1
        if end_pos >= text_len:
            break

    return chunks, dropped

def single_char_ratio(text: str) -> float:
    """Tỉ lệ token chỉ có 1 ký tự — văn bản thật ~0.05, rác toán ~0.5+"""
    toks = text.split()
    if not toks:
        return 1.0
    return sum(1 for t in toks if len(t) == 1) / len(toks)

def main() -> int:
    with open(CURATED, "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]

    all_chunks = []
    dropped_total = 0
    for doc in docs:
        text = re.sub(r"\{\\displaystyle[^}]*\}", " ", doc["text"])
        text = re.sub(r"\s+", " ", text)
        doc["text"] = text
        chunks, dropped = chunk_document(doc)
        all_chunks.extend(chunks)
        dropped_total += dropped

    with open(CHUNKS, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # ── Khối báo cáo ────────────────────────────────────────────
    sizes = [c["chars"] for c in all_chunks]
    if sizes:
        mean_size = statistics.mean(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
    else:
        mean_size = min_size = max_size = 0

    print(f"Tổng documents       : {len(docs)}")
    print(f"Tổng chunks          : {len(all_chunks)}")
    print(f"Chunk bị loại (rác)  : {dropped_total}")
    print(f"Size tb/min/max      : {mean_size:.0f} / {min_size} / {max_size}")

    # KIỂM TRA 1: bài nào mất HẾT chunk (biến mất khỏi corpus)
    by_doc = Counter(c["doc_id"] for c in all_chunks)
    no_chunk = [d["doc_id"] for d in docs if by_doc[d["doc_id"]] == 0]
    print(f"Bài mất hết chunk    : {len(no_chunk)} {no_chunk[:5]}")

    # KIỂM TRA 2: bài nào hụt đuôi  ← BẠN VIẾT
    tail_missing = []
    for doc in docs:
        cs = [c for c in all_chunks if c["doc_id"] == doc["doc_id"]]
        if not cs:
            continue
        stop = max(c["start_char"] + c["chars"] for c in cs)           
        if len(doc["text"]) - stop > 100:    # ← ngưỡng bao nhiêu thì coi là hụt?
            tail_missing.append((doc, stop))
    print(f"Bài hụt đuôi         : {len(tail_missing)} (kỳ vọng 0)")
    for doc, stop in tail_missing[:5]:
        r = single_char_ratio(doc["text"][stop:])
        verdict = "RÁC TOÁN (lọc được)" if r> THRESHOLD else "MẤT NỘI DUNG THẬT (phải xem)" 
        print(f"   [{doc['doc_id']}] {doc['title'][:32]} · hụt {len(doc['text'])-stop} kt · ratio {r:.2f} → {verdict}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
