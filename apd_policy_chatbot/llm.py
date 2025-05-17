from __future__ import annotations

from typing import List, Dict, Generator, Iterable
import textwrap
from .config import CHAT_MODEL, TOP_K, OPENAI
from .vector_store import retrieve

__all__ = ["answer_question"]


def _build_prompt(contexts: Iterable[str], question: str) -> str:
    context_block = "\n---\n".join(contexts)
    return textwrap.dedent(
        f"""
        You answer questions about a policy manual. Cite page numbers in square brackets like [p23].
        If unsure, say "I don't know".

        Excerpts:
        {context_block}

        Question: {question}
        """
    ).strip()


def answer_question(
    collection,
    question: str,
    *,
    k: int = TOP_K,
    model: str = CHAT_MODEL,
    stream: bool = False,
    history: List[Dict[str, str]] | None = None,
) -> Dict[str, str | List[int]] | Generator[dict, None, None]:
    res = retrieve(collection, question, k=k)
    contexts = res["documents"][0]
    pages = [m["page"] for m in res["metadatas"][0]]

    prompt = _build_prompt(contexts, question)
    msgs = (history or []) + [{"role": "user", "content": prompt}]

    chat_resp = OPENAI.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=msgs,
        stream=stream,
    )

    if stream:
        for chunk in chat_resp:
            if token := chunk.choices[0].delta.get("content"):
                yield {"token": token}
        yield {"pages": pages}
        return

    answer = chat_resp.choices[0].message.content.strip()
    return {"answer": answer, "pages": pages}
