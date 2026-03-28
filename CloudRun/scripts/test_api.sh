#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# test_api.sh – Test the deployed Iris SVM API
# -----------------------------------------------------------------------------
# Usage:
#   ./test_api.sh <BASE_URL>
# or set BASE_URL environment variable.
# -----------------------------------------------------------------------------

BASE_URL="${1:-${BASE_URL:-}}"
BASE_URL="${BASE_URL%/}"  # strip trailing slash

if [[ -z "$BASE_URL" ]]; then
  echo "Usage: $0 <BASE_URL>"
  echo "  e.g. $0 https://iris-svm-api-xxxxx-uc.a.run.app"
  exit 1
fi

echo "==> Testing API at ${BASE_URL}"
echo ""

echo "--- GET /health ---"
curl -s "${BASE_URL}/health" | python3 -m json.tool
echo ""

echo "--- POST /train (default params) ---"
curl -s -X POST "${BASE_URL}/train" \
  -H "Content-Type: application/json" \
  -d '{}' | python3 -m json.tool
echo ""

echo "--- POST /predict (sample setosa) ---"
curl -s -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}' | python3 -m json.tool
echo ""

echo "--- POST /predict (sample versicolor) ---"
curl -s -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [6.0, 2.7, 5.1, 1.6]}' | python3 -m json.tool
echo ""

echo "--- GET /species ---"
curl -s "${BASE_URL}/species" | python3 -m json.tool
echo ""

echo "==> All tests completed."
