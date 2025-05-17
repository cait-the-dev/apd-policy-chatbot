set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Error: OPENAI_API_KEY env var not set" >&2
  exit 1
fi

PDF="docs/apd_manual.pdf"

if [[ ! -f $PDF ]]; then
  echo "Error: $PDF not found.  Please place the manual in docs/ first." >&2
  exit 1
fi

python -m apd_policy_chatbot.cli chat "$PDF" <<<'What is the definition of reasonable suspicion?'

CHATBOT_PDF_PATH="$PDF" uvicorn apd_policy_chatbot.api:app --port 8000 &
API_PID=$!
sleep 5

QUESTION="List the circumstances when a body‑worn camera must be activated."
response=$(curl -s -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d "{\"question\": \"$QUESTION\"}")

echo -e "\n[REST] Response:\n$response\n"

kill "$API_PID" 2>/dev/null || true
echo "Demo completed 🎉"