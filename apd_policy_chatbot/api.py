from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import openai
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import pdf_utils, chunking, vector_store, llm


PDF_PATH = Path(os.getenv("CHATBOT_PDF_PATH", "./docs/apd_manual.pdf")).expanduser()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var must be set before starting the API")
openai.api_key = OPENAI_API_KEY


def _build_collection(path: Path):
    pages = pdf_utils.extract_text(path)
    chunks = [c for txt, p in pages for c in chunking.chunk_page(txt, p)]
    return vector_store.build_vector_store(chunks, name="startup_collection")

collection = None  # for demo purposes


@asynccontextmanager  
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global collection
    try:
        collection = _build_collection(PDF_PATH)
    except Exception as exc: 
        print(f"[WARN] Startup preload failed: {exc}")
    yield  


app = FastAPI(title="APD Policy Chatbot", version="0.2.1", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str
    k: int = 5
    pdf_path: Optional[str] = None 

class ChatResponse(BaseModel):
    answer: str
    pages: List[int]


@app.post("/ask", response_model=ChatResponse)
async def ask(req: ChatRequest):
    if req.pdf_path:
        try:
            pages = pdf_utils.extract_text(Path(req.pdf_path).expanduser())
            chunks = [c for txt, p in pages for c in chunking.chunk_page(txt, p)]
            local_collection = vector_store.build_vector_store(chunks, name="tmp")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        if collection is None:
            raise HTTPException(status_code=503, detail="No PDF ingested; call /ingest first")
        local_collection = collection

    result = llm.answer_question(local_collection, req.question, k=req.k)
    return JSONResponse(content=result)


@app.post("/ingest")
async def ingest(pdf_path: str = Query(..., description="Path to a PDF to ingest")):
    global collection
    try:
        pages = pdf_utils.extract_text(Path(pdf_path).expanduser())
        chunks = [c for txt, p in pages for c in chunking.chunk_page(txt, p)]
        collection = vector_store.build_vector_store(chunks, name="ingested")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "pages": len(chunks)}


@app.get("/healthz")
async def healthz():
    return {"ok": collection is not None}