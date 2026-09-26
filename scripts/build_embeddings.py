"""Build the dense embedding index for every chunk.

The HuggingFace cache is redirected with `setdefault`, so an externally set HF_HOME
still wins. The default location (%USERPROFILE%\.cache\huggingface) is access-denied
on this machine — the only reason these two lines exist. Delete them if your default
cache location is writable.
"""

import os
from pathlib import Path

os.environ.setdefault("HF_HOME", r"D:\Hermes-Workspace\hf-cache")
os.environ.setdefault("HF_HUB_CACHE", r"D:\Hermes-Workspace\hf-cache\hub")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
OUT = ROOT / "data" / "processed" / "embeddings_minilm.npy"

chunks = pd.read_json(CHUNKS, lines=True)

print(f"loading {MODEL}\nHF cache: {os.environ['HF_HOME']}", flush=True)
model = SentenceTransformer(MODEL)
print("model ready — dim =", model.get_embedding_dimension(), flush=True)

# normalize_embeddings: the dot product is then the cosine similarity, which is what
# retrieval.py assumes.
emb = model.encode(chunks["text"].tolist(), batch_size=64,
                   show_progress_bar=False, normalize_embeddings=True)
np.save(OUT, emb)
print(f"saved: {OUT}\nshape={emb.shape}  ({emb.nbytes/1e6:.1f} MB)", flush=True)
