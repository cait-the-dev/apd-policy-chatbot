# APD Policy Chatbot 🛂🤖

> A command-line & REST chatbot that answers questions about large policy documents (PDF **or** PNG/JPEG pages) using a Retrieval-Augmented-Generation pipeline.

[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#) 
[![License](https://img.shields.io/badge/license-MIT-blue)](#)

---

## ✨ Features

- **One-liner ingestion** of any PDF ≤ 1000 pages or a single PNG/JPEG image  
- Dense **vector search** (OpenAI *or* local `BAAI/bge-base-en-v1.5`) + Chroma  
- FastAPI **REST API** *and* Typer **CLI**  
- **Streaming** token output in CLI (`openai.ChatCompletion(stream=True)`)  
- **Page citations** – answers include `[p23]` style references + JSON `"pages":[…]`  
- **Chat history** maintained in the REPL for follow-up questions  
- Type-hinted, 100 % offline unit tests, Makefile, Dockerfile

---

## Quick Start

### 0 . Prereqs

| Tool | Win / macOS / Linux |
|------|---------------------|
| Python ≥ 3.10 | ✅ |
| Tesseract ≥ 5.0 (for OCR on images) | `choco install tesseract` / `brew install tesseract` |
| NVIDIA GPU (optional) | for local BGE embeddings |
| **OpenAI API key** | `export OPENAI_API_KEY=sk-…` |

### 1. Install (Poetry path)

```bash
git clone https://github.com/<you>/apd-policy-chatbot.git
cd apd-policy-chatbot
poetry install          # creates .venv and installs deps
```

### 2. Run the CLI
```bash
poetry run apd-chat docs/apd_manual.pdf           # PDF
poetry run apd-chat imgs/APD_policy_page.jpg      # JPEG
```
``` csharp
Ingested apd_manual.pdf with 751 pages.
Ask your questions (type 'exit' to quit):
>>> What is the definition of reasonable suspicion?
Reasonable suspicion is ... [p35]
```

### 3. Run the REST API (Swagger UI)
```bash
OPENAI_API_KEY=sk-... poetry run uvicorn apd_policy_chatbot.api:app --reload
# → http://127.0.0.1:8000/docs
```
### 4. (Alt) Docker in one command
```bash
OPENAI_API_KEY=sk-... make docker-build
OPENAI_API_KEY=sk-... make docker-run
```

## Makefile cheatsheet

| Target | What it does |
|--------|--------------|
| `make install` | Creates local **.venv** and installs deps from `requirements.txt` |
| `make cli FILE=docs/apd_manual.pdf` | Runs the CLI on the given document |
| `make api` | Starts the FastAPI server on <http://127.0.0.1:8000> |
| `make test` | Executes the pytest suite (with coverage) |
| `make lint` | Runs **ruff** and **mypy** linters |
| `make docker-build` / `make docker-run` | Builds the Docker image and runs it on port 8000 |


## Configuration
Env var	Default	Meaning
OPENAI_API_KEY	—	Required for OpenAI embeddings / chat
USE_LOCAL_EMBEDDER	True	Use local BGE if GPU available
LOCAL_MODEL_NAME	BAAI/bge-base-en-v1.5	Swap to bge-large-en etc.
CHATBOT_PDF_PATH	—	Pre-load a doc when starting the API

## Architecture
```pgsql
                ┌────────────┐
CLI / REST  →   │  Question  │
                └─────┬──────┘
                      │
          +-----------▼-----------+
          |   Retrieve top-k      |  ← Chroma (HNSW) + embeddings
          +-----------┬-----------+
                      │
          +-----------▼-----------+
          |   Prompt Builder      |
          +-----------┬-----------+
                      │
          +-----------▼-----------+
          |  OpenAI GPT-3.5 Turbo |  (or any chat model)
          +-----------┬-----------+
                      │
            Stream tokens / JSON
```
**Chunking** – sentence-aware, ≤ 350 tokens each
**Batch embedding** – token-aware, never exceeds 290 k tokens / request
**Tesseract OCR** – PNG/JPEG → text so the same pipeline applies

## Project Structure
```bash
apd_policy_chatbot/
├─ pdf_utils.py          # PDF & image ingestion
├─ chunking.py           # split → chunks
├─ vector_store.py       # embeddings + Chroma
├─ llm.py                # retrieval + LLM glue
├─ cli.py                # Typer REPL  (apd-chat)
└─ api.py                # FastAPI app  (/ask /ingest)
tests/                   # unit + integration (offline)
Dockerfile               # prod image
Makefile                 # dev shortcuts
```

## Design Notes
- Why BGE + Chroma? Offers top-tier recall on legal/policy text while fitting on a laptop GPU; Chroma’s DuckDB backend removes extra infra.
- Chunk size 350 tokens – keeps context under 4 k even for 10 chunks, good balance of locality & overlap.
- Streaming – demonstrates efficient token usage and better UX for long answers.

## Development
```bash
poetry run pytest -q        # 100 % offline tests
poetry run ruff check .     # lint
poetry run mypy apd_policy_chatbot
```

## License
Licensed under the MIT License © 2025 Caitlin Arnspiger.
