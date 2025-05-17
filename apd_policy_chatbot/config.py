from __future__ import annotations

import os
import openai as _openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var must be set")

OPENAI = _openai.OpenAI(api_key=OPENAI_API_KEY)

USE_LOCAL_EMBEDDER = True
LOCAL_MODEL_NAME = "BAAI/bge-base-en-v1.5"

EMBED_MODEL: str = "text-embedding-3-small"
CHAT_MODEL: str = "gpt-3.5-turbo"
CHUNK_CHAR_SIZE: int = 1_800  # ≈512 tokens
TOP_K: int = 5  # number of chunks to retrieve
EMBED_BATCH: int = 200