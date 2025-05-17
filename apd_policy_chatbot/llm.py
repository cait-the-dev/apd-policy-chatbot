from __future__ import annotations

from typing import List, Dict
import textwrap
import openai

from .config import CHAT_MODEL, TOP_K, OPENAI
from .vector_store import retrieve

__all__ = ["answer_question"]


def answer_question(collection, question: str, *, k: int = TOP_K, model: str = CHAT_MODEL) -> Dict[str, str | List[int]]:
    res = retrieve(collection, question, k=k)
    contexts = res["documents"][0]
    pages = [m["page"] for m in res["metadatas"][0]]

    context_block = "\n---\n".join(contexts)

    prompt = textwrap.dedent(
        f"""
        You answer questions about a policy manual. Cite page numbers in square brackets like [p23].
        If unsure, say "I don't know".

        Excerpts:
        {context_block}

        Question: {question}
        """
    )
    chat = OPENAI.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = chat.choices[0].message.content.strip()
    return {"answer": answer, "pages": pages}
