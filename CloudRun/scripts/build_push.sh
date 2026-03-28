#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# build_push.sh – Build Docker image and push to Artifact Registry
# -----------------------------------------------------------------------------
# Override defaults via environment variables:
#   PROJECT_ID   – GCP project ID
#   REGION       – GCP region
#   AR_REPO      – Artifact Registry repository name
#   IMAGE_NAME   – Docker image name
#   IMAGE_TAG    – Tag for the image (default: v1)
# -----------------------------------------------------------------------------

PROJECT_ID="${PROJECT_ID:-cloud-run-lab-demo}"
REGION="${REGION:-us-central1}"
AR_REPO="${AR_REPO:-cloud-run-lab}"
IMAGE_NAME="${IMAGE_NAME:-iris-svm-api}"
IMAGE_TAG="${IMAGE_TAG:-v1}"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${SCRIPT_DIR}/../iris_svm_api"

echo "==> Building image: ${IMAGE_URI}"
docker build --platform linux/amd64 -t "${IMAGE_URI}" "${APP_DIR}"

echo "==> Pushing image to Artifact Registry ..."
docker push "${IMAGE_URI}"

echo ""
echo "==> Image pushed: ${IMAGE_URI}"
