from __future__ import annotations

import types
from typing import List

import pytest

import apd_policy_chatbot.vector_store as vs

class _DummyCollection:
    def __init__(self):
        self.add_calls = 0
        self.docs = []

    def add(self, embeddings, documents, metadatas, ids):
        self.add_calls += 1
        for _id, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
            self.docs.append((_id, doc, meta, emb))

    def query(self, query_embeddings, n_results): 
        docs = [d for *_, d in self.docs][:n_results]
        metas = [m for *_, m, _ in self.docs][:n_results]
        return {"documents": [docs], "metadatas": [metas]}

class _DummyClient:
    def __init__(self, *_, **__):
        self.collections = {}

    def get_or_create_collection(self, name):
        self.collections.setdefault(name, _DummyCollection())
        return self.collections[name]

@pytest.fixture(autouse=True)
def stub_chroma(monkeypatch):
    monkeypatch.setattr(vs, "chromadb", types.SimpleNamespace(PersistentClient=_DummyClient))

@pytest.fixture(autouse=True)
def stub_embed(monkeypatch):
    def _fake_embed(texts: List[str], *_, **__):
        return [[float(i)] * 3 for i in range(len(texts))]
    monkeypatch.setattr(vs, "embed", _fake_embed)

def test_build_vector_store_batches_and_persists():
    chunks = [{"text": f"sentence {i}", "page": 1} for i in range(vs.EMBED_BATCH * 2 + 7)]
    col = vs.build_vector_store(chunks, name="batch_test")
    # Expect >1 add() calls because > EMBED_BATCH * token safety window
    assert col.add_calls >= 2
    assert len(col.docs) == len(chunks)


def test_retrieve_returns_top_k():
    chunks = [{"text": f"hello {i}", "page": i} for i in range(10)]
    col = vs.build_vector_store(chunks, name="retrieval_test")
    res = vs.retrieve(col, query="whatever", k=4)
    assert len(res["documents"][0]) == 4
    assert len(res["metadatas"][0]) == 4
