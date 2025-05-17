PROJECT_NAME := apd_policy_chatbot
PY := poetry run

.PHONY: help install dev cli api test lint docker-build docker-run clean

help:
	@echo "make install        # deps via Poetry"
	@echo "make cli FILE=...   # CLI run"
	@echo "make api            # FastAPI (requires OPENAI_API_KEY)"
	@echo "make docker-build   # build image"
	@echo "make docker-run     # run image"

install: ; poetry install
dev: ; poetry shell

cli:
	@$(if $(FILE),,echo "FILE var missing (e.g., make cli FILE=docs/policy.pdf)" && exit 1)
	poetry run apd-chat $(FILE)

api:
	poetry run uvicorn $(PROJECT_NAME).api:app --reload --port 8000

test: ; poetry run pytest -q
lint: ; poetry run ruff check $(PROJECT_NAME)

docker-build: ; docker build -t $(PROJECT_NAME):latest .
docker-run: ; docker run --rm --env-file .env -p 8000:8000 $(PROJECT_NAME):latest

clean: ; rm -rf .venv **/__pycache__
