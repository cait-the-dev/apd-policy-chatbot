FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "uvicorn[standard]"
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY apd_policy_chatbot ./apd_policy_chatbot
COPY README.md .

ENV PORT=8000
EXPOSE 8000
CMD ["uvicorn", "apd_policy_chatbot.api:app", "--host", "0.0.0.0", "--port", "8000"]
