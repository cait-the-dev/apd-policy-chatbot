PROJECT_NAME := apd_policy_chatbot
IMAGE        := apd-chat:latest 
PY           := poetry run

.PHONY: help install dev cli api test lint docker-build docker-run clean

help:
	@echo "make install         # deps via Poetry"
	@echo "make cli FILE=x.pdf  # CLI (requires OPENAI_API_KEY)"
	@echo "make api             # FastAPI on :8000"
	@echo "make docker-build    # build $(IMAGE)"
	@echo "make docker-run      # run  $(IMAGE) with .env"
	@echo "  vars: PORT=, DOC=, GPU=1"

install: ; poetry install
dev:     ; poetry shell

cli:
	@$(if $(FILE),,echo "FILE var missing (e.g., make cli FILE=docs/file.pdf)" && exit 1)
	$(PY) apd-chat $(FILE)

api:
	$(PY) uvicorn $(PROJECT_NAME).api:app --port $(or $(PORT),8000) --reload

test: ; $(PY) pytest -q
lint: ; $(PY) ruff check $(PROJECT_NAME)

docker-build:
	@echo "→ Building $(IMAGE)"
	docker build $(if $(GPU),--build-arg CUDA=1,) -t $(IMAGE) .

docker-run:
	@$(if $(OPENAI_API_KEY),,\
	  echo "[ERROR] Set OPENAI_API_KEY in .env or on the command line" && exit 1)
	docker run --rm -p $(or $(PORT),8000):8000 \
	           --env-file .env \
	           -e OPENAI_API_KEY=$(OPENAI_API_KEY) \
	           $(if $(DOC),-e CHATBOT_PDF_PATH=$(DOC),) \
	           -v $(CURDIR)/docs:/app/docs \
	           $(IMAGE)

clean: ; rm -rf .venv **/__pycache__
