# Vietnamese RAG Q&A — retrieval you can measure

A retrieval-augmented Q&A pipeline over **Vietnamese documents**, built to answer one
question honestly: *which retrieval strategy actually works, and by how much?*

Most RAG demos stop at "it runs". This one is **eval-first**: the ground-truth query set
(qrels) is written **before** the pipeline, so every design choice — chunk size, BM25 vs
dense, hybrid fusion — is justified by a measured delta in MRR / nDCG@k, not by vibes.

## Status

| Stage | State |
|---|---|
| Corpus acquisition | 🚧 in progress |
| Chunking | ⬜ |
| BM25 baseline | ⬜ |
| Dense retrieval | ⬜ |
| Hybrid (BM25 + dense) | ⬜ |
| Reranker | ⬜ (optional) |
| Generator + citations | ⬜ |
| Eval harness (qrels → MRR / nDCG@k) | ⬜ |
| Demo app | ⬜ |

## Results

*Filled in as stages land — real numbers only, never estimates.*

### Retrieval — 12 queries, k=10, 50 chunks retrieved per query
*(chunk hits aggregated to documents by max chunk score)*

| Retriever | Recall@10 | Precision@10 | MRR | nDCG@10 | Latency/query |
|---|---|---|---|---|---|
| Random (baseline) | 0.042 | 0.008 | 0.010 | 0.010 | — |
| **BM25** | **0.875** | **0.183** | **0.718** | **0.680** | 9 ms |
| Dense | ? | ? | ? | ? | ? |
| Hybrid | ? | ? | ? | ? | ? |

**BM25 by query class** — why the qrels set contains three kinds of question:

| Query class | Recall@10 | Precision@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| literal (shares terms with the document) | 1.000 | 0.150 | 0.875 | 0.882 |
| multi-document | 1.000 | 0.275 | 0.750 | 0.758 |
| **paraphrase (no term overlap)** | **0.625** | 0.125 | **0.528** | **0.398** |

BM25 scores 70× the random baseline on MRR and is near-perfect on literal queries — but loses
roughly half its ranking quality as soon as the question is phrased differently from the text,
and one paraphrased query (q006) fails outright: **0 of its gold documents retrieved**, even with
50 chunks retrieved per query. Closing that gap is the hypothesis the rest of this project tests.

### Generation (subset of queries, judged manually)

| Metric | Score |
|---|---|
| Faithfulness (claims grounded in retrieved context) | ? |
| Answer relevance | ? |
| Citation accuracy | ? |

## Architecture

```
documents ──► chunker ──┬──► BM25 index ──┐
                        │                 ├──► fusion ──► [reranker] ──► generator ──► answer + citations
                        └──► dense index ─┘                    │
                                                                ▼
                                                    eval: qrels → Recall@k / MRR / nDCG@k
```

## Why this repo is worth a look

- **Eval-first, not demo-first** — a hand-written qrels set with graded relevance (0/1/2),
  and every pipeline variant scored against the same queries.
- **Vietnamese-specific work** — word segmentation, diacritic handling, Vietnamese/multilingual
  embedding models, and the failure modes that come with them.
- **Honest negative results** — if dense retrieval does not beat BM25 on this corpus, the README
  says so, with the numbers.
- **Reproducible** — pinned requirements, a seeded query set, and a licensed, documented corpus.

## Scope & plan

See [`SCOPE.md`](SCOPE.md) for the corpus decision, eval design, milestones and risks.

## How to run

```bash
python -m venv .venv
source .venv/Scripts/activate     # Windows (Git Bash)
pip install -r requirements.txt

# fetch corpus
python scripts/fetch_corpus.py    # polite crawler (rate-limit aware, resumable)

# build indexes, run eval
./start_jupyter.sh                # notebooks/ in order, kernel "Python (rag-venv)"
```

## Repo layout

```
rag-qa-vietnamese/
├── SCOPE.md              # corpus + eval design decisions
├── data/raw/             # fetched documents (not tracked)
├── data/processed/       # chunks (not tracked)
├── scripts/
│   └── fetch_corpus.py   # rate-limit-aware, resumable fetcher
├── eval/
│   └── qrels.csv         # ground-truth query set (written BEFORE the pipeline)
├── notebooks/            # chunking, retrieval, eval
└── requirements.txt
```

## Data & license

*Source and license to be confirmed once the corpus decision is locked — the README must
state the source, license, and how to reproduce the fetch.*
