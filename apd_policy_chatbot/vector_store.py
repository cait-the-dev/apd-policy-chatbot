from __future__ import annotations

from typing import List, Dict, Sequence
import os
import tempfile

import chromadb
from tqdm import tqdm

try:
    import tiktoken
except ImportError:
    tiktoken = None 

from .config import (
    EMBED_MODEL,
    EMBED_BATCH,
    LOCAL_MODEL_NAME,
    TOP_K,
    OPENAI,
    USE_LOCAL_EMBEDDER,
)

if USE_LOCAL_EMBEDDER:
    import torch
    from sentence_transformers import SentenceTransformer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _st_model = SentenceTransformer(LOCAL_MODEL_NAME, device=DEVICE)
    _st_model.max_seq_length = 512

__all__ = ["embed", "build_vector_store", "retrieve"]

MAX_TOKENS_PER_REQ = 290_000
TMP_DIR = os.path.join(tempfile.gettempdir(), "chroma_store")

def _count_tokens(text: str, model: str = EMBED_MODEL) -> int:
    if tiktoken:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    return max(1, len(text) // 4)

def embed(texts: Sequence[str], model: str = EMBED_MODEL) -> List[List[float]]:
    if USE_LOCAL_EMBEDDER:
        vecs = _st_model.encode(
            list(texts),
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vecs.tolist()

    resp = OPENAI.embeddings.create(model=model, input=list(texts))
    return [d.embedding for d in resp.data]

def build_vector_store(
    chunks: List[Dict[str, str | int]], *, name: str = "doc_chunks"
):
    client = chromadb.PersistentClient(path=".chroma")
    col = client.get_or_create_collection(name)

    ids = [f"c{i}" for i in range(len(chunks))]
    docs = [c["text"] for c in chunks]
    metas = [{"page": c["page"]} for c in chunks]

    idx = 0
    while idx < len(docs):
        batch_docs, batch_ids, batch_metas = [], [], []
        token_accum = 0
        while idx < len(docs) and len(batch_docs) < EMBED_BATCH:
            nt = _count_tokens(docs[idx])
            if token_accum + nt > MAX_TOKENS_PER_REQ and batch_docs:
                break
            batch_docs.append(docs[idx])
            batch_ids.append(ids[idx])
            batch_metas.append(metas[idx])
            token_accum += nt
            idx += 1

        col.add(
            embeddings=embed(batch_docs),
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids,
        )
        tqdm.write(f"Embedded {len(batch_docs):4d} chunks (≈{token_accum} toks)")
    return col

def retrieve(collection, query: str, *, k: int = TOP_K):
    qvec = embed([query])[0]
    return collection.query(query_embeddings=[qvec], n_results=k)
