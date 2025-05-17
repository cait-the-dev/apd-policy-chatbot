from __future__ import annotations

from pathlib import Path

import types
import pytest
from fastapi.testclient import TestClient

import apd_policy_chatbot.api as api_module
import apd_policy_chatbot.vector_store as vs
import apd_policy_chatbot.pdf_utils as pu

class _DummyCollection:
    """Acts like a Chroma collection – stores docs in memory."""
    def __init__(self):
        self.docs = {}

    def add(self, embeddings, documents, metadatas, ids):
        for _id, doc, meta in zip(ids, documents, metadatas):
            self.docs[_id] = {"doc": doc, "meta": meta}

    def query(self, query_embeddings, n_results): 
        ids = list(self.docs)[:n_results]
        return {
            "documents": [[self.docs[_id]["doc"] for _id in ids]],
            "metadatas": [[self.docs[_id]["meta"] for _id in ids]],
        }

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
    monkeypatch.setattr(vs, "embed", lambda texts, *_, **__: [[1.0, 1.0, 1.0] for _ in texts])

@pytest.fixture(autouse=True)
def stub_extract_text(monkeypatch):
    monkeypatch.setattr(pu, "extract_text", lambda path: [("stub text for pdf", 1)])

@pytest.fixture(scope="module")
def client():
    with TestClient(api_module.app) as c:
        yield c

def test_healthz(client: TestClient):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_ingest_and_ask_roundtrip(tmp_path: Path, client: TestClient):
    # fake PDF file path
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_text("irrelevant")

    # ingest
    resp_ingest = client.post("/ingest", params={"pdf_path": str(pdf_path)})
    assert resp_ingest.status_code == 200
    assert resp_ingest.json()["status"] == "ok"

    # ask
    q_payload = {"question": "What is in this doc?"}
    resp_ask = client.post("/ask", json=q_payload)
    assert resp_ask.status_code == 200
    body = resp_ask.json()
    assert body["pages"] == [1]
    assert "answer" in body and body["answer"]
