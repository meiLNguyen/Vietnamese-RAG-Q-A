# SCOPE — Vietnamese RAG Q&A

Written before any pipeline code, on purpose. A RAG project without a locked scope turns
into an unbounded "add another retriever" loop.

## 1. Problem

Answer questions over a fixed Vietnamese document set, **with citations**, and measure
retrieval quality with a ground-truth query set.

Non-goals (explicitly out of scope for the MVP):
- fine-tuning or training an LLM
- multi-turn conversational memory
- serving at scale / production deployment
- agentic tool use

## 2. Corpus

**Decision: Vietnamese Wikipedia articles on AI/ML** (status: probing).

| Property | Value |
|---|---|
| Source | `vi.wikipedia.org` via MediaWiki API |
| License | CC BY-SA 4.0 — attribution required in README |
| Size (probe) | 204 unique articles across 9 categories |
| Format | plain text (`prop=extracts&explaintext`) |
| Fetch risk | HTTP 429 rate limiting observed → crawler must throttle + back off + resume |

Rejected alternatives:
- **Curriculum docs (English, 503 files, 1.7 MB)** — best ground-truth knowledge, zero fetch
  cost, but English: loses the Vietnamese-specific engineering that differentiates this project.
  Kept as a fallback if the crawl turns out infeasible.
- **Legal documents** — very realistic use case, but heavier document processing (PDF, layout,
  tables) than the MVP should take on.

## 3. Evaluation design (eval-first)

Written before the pipeline, so improvements are measurable.

- **Queries:** 40–50 Vietnamese questions, written from the corpus (not from memory).
  **Batch 1 done:** 12 queries / 31 judgments — 4 literal, 4 paraphrase, 4 multi-doc.
  Validated automatically by `scripts/validate_qrels.py` (missing grade-2, reused query
  text, unknown doc ids, duplicate pairs, class coverage).
- **Graded relevance:** 0 = irrelevant, 1 = related/partial, 2 = contains the answer.
- **qrels.csv columns:** `query_id, query, doc_id, relevance` — one row per judged pair.
- **Query mix:** literal (term overlap), paraphrased (no term overlap), and multi-document
  (answer requires 2+ docs) — sliced in the report, since averages hide per-class failures.
- **Metrics:** Recall@k, Precision@k, MRR (binary, position #1), nDCG@k (graded).
- **k values:** 1, 3, 5, 10.
- **Sanity rules:** a query whose gold set is larger than k caps out structurally — record the
  ceiling per query rather than reading a saturated metric as a failure.
- **Generation-side:** faithfulness + answer relevance on a 15-query subset, hand-judged
  (LLM-as-judge only as a cross-check, never as the sole judge).

## 4. Pipeline stages

1. **Fetch** — polite crawler, resume-safe, raw docs cached.
2. **Chunk** — start at ~512 characters with ~10% overlap; Vietnamese word boundaries handled
   with `underthesea`. Chunk size chosen by measurement, not by folklore.
3. **Index** — BM25 (`rank_bm25`) and dense (sentence-transformers, multilingual/VI model).
4. **Fuse** — hybrid score fusion (weighted) — pick the weight on the *validation* split of
   queries, never on the test queries.
5. **Rerank** — optional cross-encoder; only kept if it moves MRR.
6. **Generate** — LLM produces the answer with doc-id citations; prompt must force
   cite-or-refuse (no context → refuse, do not hallucinate).
7. **Evaluate** — the harness runs every variant against the same qrels.

## 5. Milestones

| # | Milestone | Definition of done |
|---|---|---|
| M3.1 | Corpus locked | crawler run completes, N articles on disk, README states source + license |
| M3.2 | qrels written | 40+ judged queries, relevance graded, query classes labelled (**batch 1: 12 queries ✅**) |
| M3.3 | BM25 baseline | ✅ done — Recall@10 0.875 · MRR 0.718 · nDCG@10 0.680 (random baseline MRR 0.010) |
| M3.4 | Dense + hybrid | ✅ done — BM25 nDCG 0.680 · dense 0.613 · **hybrid RRF 0.763** (+12% rel.) |
| M3.5 | Generator + citations | ✅ local qwen2.5:3b via Ollama — 10/12 answered · 0 invalid citations · 9/12 cited a gold doc (k=5, measured vs k=10: 6/12) |
| M3.6 | Demo app | ✅ `app.py` — Streamlit UI over the same `retrieval.py`/`rag_qa.py`; verified with `streamlit.testing` (answer path + refusal path, no exceptions) |
| M3.7 | README + push | results table complete, no `?` left, repo public |

## 6. Risks

| Risk | Mitigation |
|---|---|
| Wikipedia rate limiting (429 observed) | throttle + exponential backoff + resume from disk; run fetch as a background job |
| No LLM available locally, no API key | decide the generator path early: local model (Ollama, small Q4) vs free-tier hosted API vs extractive fallback |
| Vietnamese segmentation quality | compare `underthesea` vs whitespace tokenisation — measure, do not assume |
| qrels author bias (I wrote both corpus queries and answers) | grade against article text, and avoid queries whose answer is not literally present |
| Datasets drift under an uncommitted corpus | `data/raw/` gitignored; README links the source + documents the fetch command |
