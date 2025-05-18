from __future__ import annotations

import logging, traceback, time
from pathlib import Path
from typing import Optional, List, Dict

import typer
from rich import print
from dotenv import load_dotenv

from . import pdf_utils, chunking, vector_store, llm

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("cli")

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


@app.command()
def chat(
    path: str,
    openai_api_key: Optional[str] = typer.Option(None, envvar="OPENAI_API_KEY"),
    debug: bool = typer.Option(False, "--debug"),
):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("llm").setLevel(logging.DEBUG)

    if not openai_api_key:
        print("[red]Error: OPENAI_API_KEY not set.[/]")
        raise typer.Exit(1)

    import openai

    openai.api_key = openai_api_key

    doc_path = Path(path).expanduser()
    pages = pdf_utils.extract_text(doc_path)
    print(f"[green]Ingested {doc_path.name} with {len(pages)} pages.[/]")
    log.info("Extracted %d pages", len(pages))

    try:
        chunks = [c for txt, p in pages for c in chunking.chunk_page(txt, p)]
        collection = vector_store.build_vector_store(chunks)
        log.info("Vector store ready (%d chunks)", len(chunks))
    except Exception as exc:
        log.error("Failed to build store\n%s", traceback.format_exc())
        print(f"[red]Fatal while building vector store:[/] {exc}")
        raise typer.Exit(1)

    history: List[Dict[str, str]] = []
    print("Ask your questions (type 'exit' to quit):")
    while True:
        try:
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        t0 = time.perf_counter()
        try:
            buffer: List[str] = []
            pages_cited: List[int] = []
            for piece in llm.answer_question(
                collection, q, stream=True, history=history
            ):
                if "token" in piece:
                    buffer.append(piece["token"])
                    print(piece["token"], end="", flush=True)
                elif "pages" in piece:
                    pages_cited = piece["pages"]

            if not buffer:
                fallback = llm.answer_question(collection, q, stream=False, history=history)
                buffer.append(fallback["answer"])
                pages_cited = fallback["pages"]
                print(fallback["answer"])

            print()  
            log.info("Total latency %.2fs pages=%s", time.perf_counter() - t0, pages_cited)
            history.extend(
                [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": "".join(buffer)},
                ]
            )
        except Exception as exc:
            print(f"[red]Error:[/] {exc}")
            log.error(traceback.format_exc())


if __name__ == "__main__":
    app()