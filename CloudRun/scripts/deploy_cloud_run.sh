#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# deploy_cloud_run.sh – Deploy image from Artifact Registry to Cloud Run
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

echo "==> Deploying ${IMAGE_NAME} to Cloud Run (region: ${REGION}) ..."

gcloud run deploy "${IMAGE_NAME}" \
  --image "${IMAGE_URI}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 512Mi \
  --max-instances 1

SERVICE_URL=$(gcloud run services describe "${IMAGE_NAME}" --region "${REGION}" --format='value(status.url)')

echo ""
echo "==> Deployment complete."
echo "    Service URL: ${SERVICE_URL}"
echo ""
echo "Test with:"
echo "  curl ${SERVICE_URL}/health"
