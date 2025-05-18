from __future__ import annotations

import re
from typing import List, Dict

from .config import CHUNK_CHAR_SIZE

__all__ = ["sentence_split", "chunk_page"]


def sentence_split(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_page(text: str, page_no: int, chunk_size: int = CHUNK_CHAR_SIZE) -> List[Dict[str, str | int]]:
    buff = ""
    chunks: List[Dict[str, str | int]] = []
    for sent in sentence_split(text):
        if len(buff) + len(sent) + 1 <= chunk_size:
            buff = f"{buff} {sent}".strip()
        else:
            if buff:
                chunks.append({"text": buff, "page": page_no})
            buff = sent
    if buff:
        chunks.append({"text": buff, "page": page_no})
    return chunks