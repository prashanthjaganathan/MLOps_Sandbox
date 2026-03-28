#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# destroy_cloud_run.sh – Remove Cloud Run service (stops all revisions / instances)
# -----------------------------------------------------------------------------
# Same defaults as deploy_cloud_run.sh. Override via environment variables:
#   PROJECT_ID, REGION, IMAGE_NAME (Cloud Run service name), AR_REPO, IMAGE_TAG
#
# Optional:
#   SKIP_PROMPT=1       – skip the confirmation prompt (for CI / automation)
#   DELETE_AR_IMAGE=1   – also delete the image tag in Artifact Registry
#   DELETE_LOCAL_IMAGE=1 – run docker rmi on the local tag (if present)
# -----------------------------------------------------------------------------

PROJECT_ID="${PROJECT_ID:-cloud-run-lab-demo}"
REGION="${REGION:-us-central1}"
AR_REPO="${AR_REPO:-cloud-run-lab}"
IMAGE_NAME="${IMAGE_NAME:-iris-svm-api}"
IMAGE_TAG="${IMAGE_TAG:-v1}"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

if [[ "${SKIP_PROMPT:-0}" != "1" ]]; then
  read -r -p "Delete Cloud Run service '${IMAGE_NAME}' in ${REGION} (project ${PROJECT_ID})? [y/N] " ans
  case "${ans}" in
    y | Y | yes | YES) ;;
    *)
      echo "Aborted."
      exit 1
      ;;
  esac
fi

echo "==> Deleting Cloud Run service: ${IMAGE_NAME} ..."
if gcloud run services describe "${IMAGE_NAME}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" &>/dev/null; then
  gcloud run services delete "${IMAGE_NAME}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --quiet
  echo "    Service deleted (all revisions / instances stopped)."
else
  echo "    Service not found; nothing to delete on Cloud Run."
fi

if [[ "${DELETE_AR_IMAGE:-0}" == "1" ]]; then
  echo "==> Deleting Artifact Registry image: ${IMAGE_URI} ..."
  if gcloud artifacts docker images delete "${IMAGE_URI}" \
    --project "${PROJECT_ID}" \
    --delete-tags \
    --quiet; then
    echo "    Artifact Registry image removed."
  else
    echo "    Could not delete image (missing, or permissions)."
  fi
fi

if [[ "${DELETE_LOCAL_IMAGE:-0}" == "1" ]]; then
  echo "==> Removing local Docker image (if present): ${IMAGE_URI} ..."
  docker rmi "${IMAGE_URI}" 2>/dev/null || echo "    No local image with that tag."
fi

echo "==> Teardown complete."
