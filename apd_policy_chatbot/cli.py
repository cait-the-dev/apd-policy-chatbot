from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print

from . import pdf_utils, chunking, vector_store, llm

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


@app.command()
def chat(path: str, openai_api_key: Optional[str] = typer.Option(None, envvar="OPENAI_API_KEY")):
    if not openai_api_key:
        print("[red]Error: OPENAI_API_KEY not set.[/]")
        raise typer.Exit(1)

    import openai
    openai.api_key = openai_api_key

    pdf_path = Path(path).expanduser()
    pages = pdf_utils.extract_text(pdf_path)

    print(f"[green]Ingested {pdf_path.name} with {len(pages)} pages.[/]")

    chunks = [c for text, p in pages for c in chunking.chunk_page(text, p)]
    collection = vector_store.build_vector_store(chunks)

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
            for chunk in llm.answer_question(
                collection, q, stream=True
            ):      
                if "token" in chunk:
                    print(chunk["token"], end="", flush=True)
            print() 
        except Exception as exc:
            print(f"[red]Error:[/] {exc}")


if __name__ == "__main__":
    app()