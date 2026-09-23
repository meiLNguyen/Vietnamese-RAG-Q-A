import json, statistics

from chunk_corpus import single_char_ratio
chunks = [json.loads(l) for l in open("data/processed/chunks.jsonl", encoding="utf-8")]
ratios = sorted(single_char_ratio(c["text"]) for c in chunks)

print("trung vị :", round(statistics.median(ratios), 3))
print("p75/p90/p95/p99:", [round(ratios[int(len(ratios)*p)], 3) for p in (0.75, 0.90, 0.95, 0.99)])
print("max      :", round(ratios[-1], 3))

for th in (0.2, 0.3, 0.35, 0.4, 0.5):
    n = sum(1 for r in ratios if r > th)
    print(f"  ngưỡng {th}: loại {n:>5} chunk ({n/len(ratios)*100:.1f}%)")


scored = sorted(chunks, key=lambda c: single_char_ratio(c["text"]))
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                sắp xếp danh sách chunk theo điểm chất lượng (thấp → cao)

for c in scored:
    r = single_char_ratio(c["text"])
    if 0.4 < r:          # vùng biên — đọc thử 5 cái
        print(f"[{c['chunk_id']}] {r:.2f} {c['text'][:150]}")