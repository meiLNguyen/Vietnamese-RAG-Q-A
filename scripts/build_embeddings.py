"""Build dense embeddings for all chunks.

Cache note: C:\\Users\\Administrator\\.cache is access-denied on this machine,
so the HuggingFace cache is redirected to D: — the env vars MUST be set before
importing sentence_transformers / huggingface_hub.
"""

import os

os.environ["HF_HOME"] = r"D:\Hermes-Workspace\hf-cache"
os.environ["HF_HUB_CACHE"] = r"D:\Hermes-Workspace\hf-cache\hub"

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNKS = r"D:\projects\rag-qa-vietnamese\data\processed\chunks.jsonl"
OUT = r"D:\projects\rag-qa-vietnamese\data\processed\embeddings_minilm.npy"

chunks = pd.read_json(CHUNKS, lines=True)
print(f"nạp model {MODEL}\ncache HF: {os.environ['HF_HOME']}", flush=True)
model = SentenceTransformer(MODEL)
print("model OK — dim =", model.get_sentence_embedding_dimension(), flush=True)

emb = model.encode(chunks["text"].tolist(), batch_size=64,
                   show_progress_bar=False, normalize_embeddings=True)
np.save(OUT, emb)
print(f"ĐÃ LƯU: {OUT}\nshape={emb.shape}  ({emb.nbytes/1e6:.1f} MB)", flush=True)
