from __future__ import annotations
import os
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

import logging
import os
import tempfile
from typing import List, Dict, Sequence

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
    """Return 2-D list of floats (local SBERT or OpenAI)."""
    if USE_LOCAL_EMBEDDER:
        return (
            _st_model.encode(
                list(texts),
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            .tolist()
        )

    resp = OPENAI.embeddings.create(model=model, input=list(texts))
    return [d.embedding for d in resp.data]


class _Embedder:
    def __init__(self, dim: int):
        self._dim = dim

    def __call__(self, input): 
        return embed(input)

    @property
    def dimension(self) -> int:  
        return self._dim


def build_vector_store(
    chunks: List[Dict[str, str | int]], *, name: str = "doc_chunks"
):
    client = chromadb.PersistentClient(path=".chroma")

    first_docs = [c["text"] for c in chunks[:EMBED_BATCH]]
    first_vecs = embed(first_docs)
    dim = len(first_vecs[0])

    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)

    col = client.get_or_create_collection(name, embedding_function=_Embedder(dim))

    col.add(
        embeddings=first_vecs,
        documents=first_docs,
        metadatas=[{"page": c["page"]} for c in chunks[:EMBED_BATCH]],
        ids=[f"c{i}" for i in range(len(first_docs))],
    )
    tqdm.write(f"Embedded {len(first_docs):4d} chunks (dim={dim})")

    for start in range(EMBED_BATCH, len(chunks), EMBED_BATCH):
        batch = chunks[start : start + EMBED_BATCH]
        col.add(
            embeddings=embed([c["text"] for c in batch]),
            documents=[c["text"] for c in batch],
            metadatas=[{"page": c["page"]} for c in batch],
            ids=[f"c{i}" for i in range(start, start + len(batch))],
        )
        tqdm.write(f"Embedded {len(batch):4d} chunks")
    return col


def retrieve(collection, query: str, *, k: int = TOP_K):
    qvec = embed([query])[0]
    res = collection.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    logging.getLogger("vector_store").debug(
        "Retrieve hits=%d pages=%s",
        len(res["documents"][0]),
        [m.get("page") for m in res["metadatas"][0]],
    )
    return res
