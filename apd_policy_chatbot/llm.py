from __future__ import annotations

import logging, textwrap, time
from typing import List, Dict, Generator, Iterable, Any

from .config import CHAT_MODEL, TOP_K, OPENAI
from .vector_store import retrieve

__all__ = ["answer_question"]

log = logging.getLogger("llm")


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
    collection: Any,
    question: str,
    *,
    k: int = TOP_K,
    model: str = CHAT_MODEL,
    stream: bool = False,
    history: List[Dict[str, str]] | None = None,
) -> Dict[str, str | List[int]] | Generator[dict, None, None]:
    t0 = time.perf_counter()

    res = retrieve(collection, question, k=k)
    contexts = res["documents"][0]
    pages = [m["page"] for m in res["metadatas"][0]]

    if not contexts:
        return {"answer": "I don't know.", "pages": []}

    prompt = _build_prompt(contexts, question)
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Prompt chars=%d pages=%s", len(prompt), pages)

    msgs = (history or []) + [{"role": "user", "content": prompt}]

    resp = OPENAI.chat.completions.create(
        model=model, temperature=0.2, messages=msgs, stream=stream
    )

    if stream:
        for chunk in resp:
            if token := chunk.choices[0].delta.get("content"):
                yield {"token": token}
        yield {"pages": pages}
        log.info("Streamed in %.2f s pages=%s", time.perf_counter() - t0, pages)
        return

    answer = resp.choices[0].message.content.strip()
    log.info("Answered in %.2f s pages=%s", time.perf_counter() - t0, pages)
    return {"answer": answer, "pages": pages}