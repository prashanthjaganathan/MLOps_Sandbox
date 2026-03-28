#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# setup_gcp.sh – Create GCP project, enable APIs, create Artifact Registry repo
# -----------------------------------------------------------------------------
# Override defaults via environment variables:
#   PROJECT_ID   – GCP project ID (must be globally unique)
#   REGION       – GCP region for Artifact Registry and Cloud Run
#   AR_REPO      – Artifact Registry repository name
# -----------------------------------------------------------------------------

PROJECT_ID="${PROJECT_ID:-cloud-run-lab-demo}"
REGION="${REGION:-us-central1}"
AR_REPO="${AR_REPO:-cloud-run-lab}"

echo "==> Using PROJECT_ID=${PROJECT_ID}, REGION=${REGION}, AR_REPO=${AR_REPO}"

# 1. Create project if it doesn't exist (may fail if ID taken or no org perms)
if gcloud projects describe "$PROJECT_ID" &>/dev/null; then
  echo "Project $PROJECT_ID already exists."
else
  echo "Creating project $PROJECT_ID ..."
  gcloud projects create "$PROJECT_ID" || {
    echo "ERROR: Could not create project. The ID may be taken or you lack org permissions."
    echo "Set a unique PROJECT_ID and re-run."
    exit 1
  }
fi

# 2. Set active project
gcloud config set project "$PROJECT_ID"

# 3. Enable required APIs
echo "Enabling APIs ..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  cloudresourcemanager.googleapis.com

# 4. Create Artifact Registry repository (docker format)
if gcloud artifacts repositories describe "$AR_REPO" --location="$REGION" &>/dev/null; then
  echo "Artifact Registry repo $AR_REPO already exists in $REGION."
else
  echo "Creating Artifact Registry repo $AR_REPO in $REGION ..."
  gcloud artifacts repositories create "$AR_REPO" \
    --repository-format=docker \
    --location="$REGION" \
    --description="Docker images for Cloud Run ML lab"
fi

# 5. Configure Docker authentication for Artifact Registry
echo "Configuring Docker auth for ${REGION}-docker.pkg.dev ..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo ""
echo "==> Setup complete."
echo "    Make sure billing is enabled for project $PROJECT_ID in the GCP Console."
echo "    Your account needs Artifact Registry Writer + Cloud Run Admin roles to push/deploy."
