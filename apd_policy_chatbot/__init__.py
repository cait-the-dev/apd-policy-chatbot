from importlib.metadata import version as _v, PackageNotFoundError

__all__ = [
    "__version__",
    "load_pdf",
    "chat",
]

try:  
    __version__: str = _v(__name__.replace("_", "-"))
except PackageNotFoundError:  
    __version__ = "0.0.0"

from pathlib import Path
from typing import List, Dict 

from apd_policy_chatbot import pdf_utils, chunking, vector_store, llm
from .config import TOP_K 


def load_pdf(path: str | Path):
    pages = pdf_utils.extract_text(Path(path))
    chunks = [c for text, p in pages for c in chunking.chunk_page(text, p)]
    return vector_store.build_vector_store(chunks)


def chat(collection, question: str, k: int = TOP_K) -> Dict[str, str | List[int]]:
    return llm.answer_question(collection, question, k=k)