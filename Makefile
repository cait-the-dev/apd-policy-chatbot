ifneq (,$(wildcard .env))
include .env
export $(shell sed -E 's/=.*//' .env)
endif

PROJECT_NAME := apd_policy_chatbot
PYTHON := python

.PHONY: help install dev cli api test lint docker-build docker-run clean

help:
	@echo "Common tasks:"
	@echo "  make install        Install deps via Poetry"
	@echo "  make dev            Activate Poetry shell"
	@echo "  make cli FILE=x.pdf Run CLI chatbot (needs OPENAI_API_KEY)"
	@echo "  make api            Run FastAPI on :8000 (needs OPENAI_API_KEY)"
	@echo "  make test           Run pytest"
	@echo "  make lint           Run Ruff"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-run     Run container using .env for secrets"
	@echo "  make clean          Remove .venv and __pycache__"

install:
	poetry install

dev:
	poetry shell

cli:
	@$(if $(OPENAI_API_KEY),,echo "\n[ERROR] OPENAI_API_KEY not set (see .env)." && exit 1)
	poetry run apd-chat $(FILE)

api:
	@$(if $(OPENAI_API_KEY),,echo "\n[ERROR] OPENAI_API_KEY not set (see .env)." && exit 1)
	poetry run uvicorn $(PROJECT_NAME).api:app --reload --port 8000

test:
	poetry run pytest -q

lint:
	poetry run ruff check $(PROJECT_NAME)

docker-build:
	docker build -t $(PROJECT_NAME):latest .

docker-run:
	@$(if $(OPENAI_API_KEY),,echo "\n[ERROR] OPENAI_API_KEY not set in .env." && exit 1)
	docker run --rm --env-file .env -p 8000:8000 $(PROJECT_NAME):latest

clean:
	rm -rf .venv **/__pycache__