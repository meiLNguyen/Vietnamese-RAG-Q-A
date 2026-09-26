"""Retrieval layer: BM25, dense and hybrid RRF — shared by eval, CLI and demo.

Why a module: the demo app and the evaluation must run *the same* retrieval code.
Two copies drift, and then the numbers in the README stop describing the thing
users actually run.

Every number this module produces was cross-checked against the notebook
implementation before extraction (same qrels, same splits, identical metrics).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# The HF cache must be redirected before sentence_transformers is imported:
# C:\Users\<user>\.cache is access-denied on this machine.
os.environ.setdefault("HF_HOME", r"D:\Hermes-Workspace\hf-cache")
os.environ.setdefault("HF_HUB_CACHE", r"D:\Hermes-Workspace\hf-cache\hub")

import numpy as np

from explore import tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
EMB_PATH = ROOT / "data" / "processed" / "embeddings_minilm.npy"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RRF_K = 60          # standard constant from the RRF paper; see README for the K sweep


class Retriever:
    """Hybrid retriever over the chunk index.

    retrieve() returns chunks (with text) for generation;
    ranked_docs() returns just the document ranking for evaluation.
    """

    def __init__(self, chunks_path: Path = CHUNKS_PATH, emb_path: Path = EMB_PATH) -> None:
        self.chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.emb = np.load(emb_path)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.model = SentenceTransformer(MODEL_NAME)
        if len(self.chunks) != self.emb.shape[0]:
            raise ValueError(f"chunk/embedding mismatch: {len(self.chunks)} vs {self.emb.shape[0]}")

    # ── individual retrievers ───────────────────────────────────────────────
    def _aggregate(self, idx: np.ndarray, scores: np.ndarray, k: int) -> list[str]:
        """Chunk hits -> document ranking (max chunk score per document)."""
        best: dict[str, float] = {}
        for i in idx:
            doc = self.chunks[i]["doc_id"]
            best[doc] = max(best.get(doc, -1e9), float(scores[i]))
        return [d for d, _ in sorted(best.items(), key=lambda x: -x[1])[:k]]

    def bm25_docs(self, query: str, depth: int = 50) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        return self._aggregate(np.argsort(scores)[::-1][:depth], scores, depth)

    def dense_docs(self, query: str, depth: int = 50) -> list[str]:
        qvec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.emb @ qvec
        return self._aggregate(np.argsort(scores)[::-1][:depth], scores, depth)

    # ── fusion ──────────────────────────────────────────────────────────────
    @staticmethod
    def rrf(ranked_lists: list[list[str]], K: int = RRF_K) -> list[str]:
        """Reciprocal Rank Fusion: score(d) = sum 1/(K + rank), rank from 1.

        Uses ranks only, so BM25 scores and cosine similarities never need to be
        put on a common scale — which is exactly why RRF is the default choice.
        """
        scores: dict[str, float] = {}
        for lst in ranked_lists:
            for rank, doc in enumerate(lst, start=1):
                scores[doc] = scores.get(doc, 0.0) + 1.0 / (K + rank)
        return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]

    # ── public API ──────────────────────────────────────────────────────────
    def ranked_docs(self, query: str, k: int = 10, depth: int = 50) -> list[str]:
        """Hybrid document ranking — the metric-facing output."""
        return self.rrf([self.bm25_docs(query, depth), self.dense_docs(query, depth)])[:k]

    def _chunk_scores(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """BM25 and cosine score for EVERY chunk, each min-max normalised.

        Normalisation makes the two scales comparable, so "which passage matters
        most for this query" can be answered without pretending BM25 scores and
        cosine similarities are the same quantity.
        """
        bm = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)
        qvec = self.model.encode([query], normalize_embeddings=True)[0]
        de = self.emb @ qvec

        def scale(a: np.ndarray) -> np.ndarray:
            lo, hi = a.min(), a.max()
            return np.zeros_like(a) if hi - lo < 1e-12 else (a - lo) / (hi - lo)

        return scale(bm), scale(de)

    def retrieve(self, query: str, k: int = 5, depth: int = 50, per_doc: int = 1,
                 guarantee: int = 2) -> list[dict]:
        """Top passages for generation, most relevant first.

        Three things matter here, none of them obvious:

        1. Documents are ordered by the fused rank (that is what the eval measured),
           but WITHIN a document the passages are ordered by how well each one
           matches the query — not by position in the article. Handing the LLM the
           first chunks of a document instead of its most relevant chunks is how a
           retrieval score can look fine while the answerable passage never reaches
           the prompt.

        2. k is capped at 5 MEASURED, not guessed: with 10 passages the generator
           answers one more question out of 12 but cites the wrong passage in three
           more (9/12 vs 6/12 correct-gold citations). More context is not more
           faithfulness — a 3B model spreads its attention over distractors.

        3. The top `guarantee` documents of EACH retriever are always admitted.
           RRF rewards consensus, so a passage that only one retriever finds can be
           outranked by documents both retrievers half-agree on — measured on this
           corpus: for "Chưng cất tri thức là gì?" the correct article is BM25's #1
           hit yet falls outside the fused top 5, and the generator then (correctly)
           refuses to answer a question the corpus does answer. Context selection is
           a recall problem, so it gets a recall-safe rule.
        """
        bm_n, de_n = self._chunk_scores(query)
        chunk_score = np.maximum(bm_n, de_n)

        bm = self.bm25_docs(query, depth)
        de = self.dense_docs(query, depth)
        fused = self.rrf([bm, de])

        must = set(bm[:guarantee]) | set(de[:guarantee])
        ordered = [d for d in fused if d in must] + [d for d in fused if d not in must]

        by_doc: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            if c["doc_id"] in must or c["doc_id"] in set(fused):
                by_doc.setdefault(c["doc_id"], []).append(i)

        picked = []
        for doc in ordered:
            idxs = sorted(by_doc.get(doc, []), key=lambda i: -chunk_score[i])[:per_doc]
            for i in idxs:
                picked.append(self.chunks[i])
                if len(picked) >= k:
                    return picked
        return picked


if __name__ == "__main__":   # smoke test: python scripts/retrieval.py "câu hỏi"
    import sys

    r = Retriever()
    q = sys.argv[1] if len(sys.argv) > 1 else "Chưng cất tri thức là gì?"
    print(f"query: {q}\n")
    for i, c in enumerate(r.retrieve(q, k=3), 1):
        print(f"[{i}] {c['title']}  (chunk {c['chunk_id']})")
        print(f"    {c['text'][:150]}...\n")
