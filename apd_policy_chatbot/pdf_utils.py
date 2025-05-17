from __future__ import annotations

import pathlib
from typing import List, Tuple

import pdfplumber
from tqdm import tqdm

__all__ = ["validate_pdf", "extract_text"]


def validate_pdf(path: pathlib.Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File '{path}' not found")
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF documents are supported in this version")


def extract_text(path: pathlib.Path) -> List[Tuple[str, int]]:
    validate_pdf(path)
    pages: List[Tuple[str, int]] = []
    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) > 1_000:
            raise ValueError("Document exceeds 1 000‑page limit")
        for idx, page in enumerate(tqdm(pdf.pages, desc="Extracting PDF text")):
            pages.append((page.extract_text() or "", idx + 1))
    return pages