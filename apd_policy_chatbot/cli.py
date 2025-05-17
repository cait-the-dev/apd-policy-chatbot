from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict

import typer
from rich import print

from . import pdf_utils, chunking, vector_store, llm

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


@app.command()
def chat(
    path: str,
    openai_api_key: Optional[str] = typer.Option(None, envvar="OPENAI_API_KEY"),
):
    if not openai_api_key:
        print("[red]Error: OPENAI_API_KEY not set.[/]")
        raise typer.Exit(1)

    import openai

    openai.api_key = openai_api_key

    doc_path = Path(path).expanduser()
    pages = pdf_utils.extract_text(doc_path)

    print(f"[green]Ingested {doc_path.name} with {len(pages)} pages.[/]")

    chunks = [c for text, p in pages for c in chunking.chunk_page(text, p)]
    collection = vector_store.build_vector_store(chunks)

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

        try:
            buffer: List[str] = []
            for piece in llm.answer_question(
                collection,
                q,
                stream=True,
                history=history,
            ):
                if "token" in piece:
                    buffer.append(piece["token"])
                    print(piece["token"], end="", flush=True)
                elif "pages" in piece:
                    pages = piece["pages"]
            print()  
            history.extend(
                [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": "".join(buffer)},
                ]
            )
        except Exception as exc:
            print(f"[red]Error:[/] {exc}")


if __name__ == "__main__":
    app()
