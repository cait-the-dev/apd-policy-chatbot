from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import openai
from chromadb import PersistentClient
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import pdf_utils, chunking, vector_store, llm

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    print("[WARN] OPENAI_API_KEY not set — /ask will return 503 until provided.")

STARTUP_DOC = os.getenv("CHATBOT_PDF_PATH") 
CHROMA_PATH = ".chroma"
COLL_NAME   = "ingested"

def _load_persisted_collection() -> Optional[vector_store.chromadb.Collection]:
    client = PersistentClient(path=CHROMA_PATH)
    try:
        col = client.get_collection(COLL_NAME)
    except ValueError:
        return None
    return col if col.count() > 0 else None

def _build_collection(path: Path):
    pages = pdf_utils.extract_text(path)
    chunks = [c for txt, p in pages for c in chunking.chunk_page(txt, p)]
    return vector_store.build_vector_store(chunks, name="doc")

collection = None  # global cache for demo

# Pre-loads the document on server startup.
# @asynccontextmanager
# async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
#     global collection
#     if STARTUP_DOC and Path(STARTUP_DOC).expanduser().exists():
#         try:
#             print(f"[INFO] Pre-loading {STARTUP_DOC} …")
#             collection = _build_collection(Path(STARTUP_DOC).expanduser())
#             print("[INFO] Pre-load finished.")
#         except Exception as exc:
#             print(f"[WARN] Startup preload failed: {exc}")
#     else:
#         print("[INFO] No startup document — waiting for /ingest")
#     yield  

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("[INFO] Startup: not preloading any document. Use /ingest.")
    yield

app = FastAPI(title="APD Policy Chatbot", version="0.3.0", lifespan=lifespan)

class ChatRequest(BaseModel):
    question: str
    k: int = 5
    pdf_path: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    pages: List[int]

@app.post("/ask", response_model=ChatResponse)
async def ask(req: ChatRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(503, "OPENAI_API_KEY not configured")

    global collection

    if req.pdf_path:
        try:
            pages  = pdf_utils.extract_text(Path(req.pdf_path).expanduser())
            chunks = [c for txt, p in pages for c in chunking.chunk_page(txt, p)]
            local  = vector_store.build_vector_store(chunks, name="tmp")
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
    else:
        if collection is None:
            collection = _load_persisted_collection()
        if collection is None:
            raise HTTPException(503, "No document ingested; call /ingest")
        local = collection

    result = llm.answer_question(local, req.question, k=req.k, stream=False)
    return JSONResponse(content=result)

@app.post("/ingest")
async def ingest(pdf_path: str = Query(..., description="Path to a PDF to ingest")):
    global collection
    try:
        pages = pdf_utils.extract_text(Path(pdf_path).expanduser())
        chunks = [c for txt, p in pages for c in chunking.chunk_page(txt, p)]
        collection = vector_store.build_vector_store(chunks, name=COLL_NAME)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "pages": len(chunks)}

@app.get("/healthz")
async def healthz():
    return {"ok": collection is not None}
