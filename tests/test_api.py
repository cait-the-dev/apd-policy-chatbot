from __future__ import annotations

from pathlib import Path
import types

import pytest
from fastapi.testclient import TestClient

import apd_policy_chatbot.api as api_module
import apd_policy_chatbot.vector_store as vs
import apd_policy_chatbot.pdf_utils as pu
import apd_policy_chatbot.llm as llm

class _DummyCollection:
    def __init__(self):
        self.docs = {}
    def add(self, embeddings, documents, metadatas, ids):
        for _id, doc, meta in zip(ids, documents, metadatas):
            self.docs[_id] = {"doc": doc, "meta": meta}
    def query(self, query_embeddings, n_results):
        ids = list(self.docs)[:n_results]
        return {
            "documents": [[self.docs[i]["doc"] for i in ids]],
            "metadatas": [[self.docs[i]["meta"] for i in ids]],
        }

class _DummyClient:
    def __init__(self, *_, **__): self.collections = {}
    def get_or_create_collection(self, name):
        self.collections.setdefault(name, _DummyCollection())
        return self.collections[name]

@pytest.fixture(autouse=True)
def stub_chroma(monkeypatch):
    monkeypatch.setattr(vs, "chromadb", types.SimpleNamespace(PersistentClient=_DummyClient))

@pytest.fixture(autouse=True)
def stub_embed(monkeypatch):
    monkeypatch.setattr(vs, "embed", lambda txts, *_, **__: [[0.1, 0.2, 0.3] for _ in txts])

@pytest.fixture(autouse=True)
def stub_extract(monkeypatch):
    monkeypatch.setattr(pu, "extract_text", lambda _: [("stub text", 1)])

@pytest.fixture(autouse=True)
def stub_answer(monkeypatch):
    monkeypatch.setattr(
        llm,
        "answer_question",
        lambda *_args, **_kw: {"answer": "stub answer", "pages": [1]},
    )

@pytest.fixture(scope="module")
def client():
    with TestClient(api_module.app) as c:
        yield c

def test_healthz_initial_false(client: TestClient):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"ok": False}

def test_ingest_then_healthz_true_and_ask(tmp_path: Path, client: TestClient):
    pdf = tmp_path / "dummy.pdf"
    pdf.write_text("irrelevant")

    ing = client.post("/ingest", params={"pdf_path": str(pdf)})
    assert ing.status_code == 200 and ing.json()["status"] == "ok"

    assert client.get("/healthz").json() == {"ok": True}

    ask = client.post("/ask", json={"question": "whatever"})
    assert ask.status_code == 200
    body = ask.json()
    assert body["pages"] == [1] and "answer" in body
