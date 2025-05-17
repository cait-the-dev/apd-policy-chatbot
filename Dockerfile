FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.2

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s "$HOME/.local/bin/poetry" /usr/local/bin/poetry

WORKDIR /app
COPY pyproject.toml poetry.lock* README.md ./

RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi

COPY apd_policy_chatbot ./apd_policy_chatbot

ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "apd_policy_chatbot.api:app", "--host", "0.0.0.0", "--port", "8000"]