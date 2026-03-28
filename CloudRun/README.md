# Containerized ML Training using GCP Cloud Run (+ Artifact Registry)

Deploy a simple ML pipeline (Iris + SVM) containerized as a FastAPI service on Google Cloud Run, with full scripted setup for GCP resources, Artifact Registry, and deployment.

**How this lab differs from the [Cloud Run Beginner Lab](https://github.com/raminmohammadi/MLOps/tree/main/Labs/GCP_Labs/Cloud_Runner_Labs/Begineer_Lab)**

| | Beginner lab | This lab |
|---|----------------|----------|
| **App** | Simple Flask hello-world | FastAPI service with a **trainable ML pipeline** (Iris + SVM): `/train`, `/predict`, `/species` |
| **GCP workflow** | Heavier use of **manual Console** steps | **Scripted** project setup, API enablement, Artifact Registry repo, Docker auth (`setup_gcp.sh`) |
| **Registry & deploy** | Introduces the ideas step-by-step in the UI | **Build → push → deploy** from the shell (`build_push.sh`, `deploy_cloud_run.sh`) to Artifact Registry and Cloud Run |
| **Ops extras** | — | Optional `.env.example`, teardown script (`destroy_cloud_run.sh`), and versioning via `IMAGE_TAG` |

---


## What this repo builds

| Component | Description |
|-----------|-------------|
| **ML Pipeline** | `StandardScaler` → `SVC` trained on the Iris dataset |
| **FastAPI** | `/train` to fit the model, `/predict` to classify new samples |
| **Docker** | Containerized app pushed to Artifact Registry |
| **Cloud Run** | Serverless deployment with automatic HTTPS |

---

## Prerequisites

1. **Google Cloud SDK** (`gcloud`) installed and authenticated
2. **Docker** installed and running
3. A **billing account** you can attach to the GCP project (API enablement fails until it is linked; see Step 1)
4. Basic familiarity with Python and REST APIs

---

## Project Structure

```text
CloudRun/
├── README.md
├── .env.example            # Template for shell variables (scripts do not auto-load .env)
├── scripts/
│   ├── setup_gcp.sh         # Create project, enable APIs, create Artifact Registry
│   ├── build_push.sh        # Build and push Docker image
│   ├── deploy_cloud_run.sh  # Deploy to Cloud Run
│   ├── destroy_cloud_run.sh # Remove service (optional registry / local image cleanup)
│   └── test_api.sh          # Test the deployed API
└── iris_svm_api/
    ├── Dockerfile
    ├── requirements.txt
    └── app/
        ├── __init__.py
        ├── main.py         # FastAPI endpoints
        └── pipeline.py     # Iris loading, training, prediction
```

---

## Quick Start

### Step 0 (optional) – Environment variables

The shell scripts **do not read a `.env` file**. They only use variables already exported in your current shell. If you set nothing, each script falls back to the **defaults** in the table below.

**Option A — use defaults:** Skip this step; run the scripts as written.

**Option B — use a file:** Copy `.env.example` to `.env`, edit values, then load them into your shell from the directory that contains `.env` (this lab folder) before running any script:

```bash
git clone https://github.com/prashanthjaganathan/MLOps_Sandbox.git
cd CloudRun
set -a && source .env && set +a
cd scripts
```

| Variable | Default | Notes |
|----------|---------|--------|
| `PROJECT_ID` | `cloud-run-lab-demo` | Globally unique; change if the ID is taken |
| `REGION` | `us-central1` | Artifact Registry and Cloud Run region |
| `AR_REPO` | `cloud-run-lab` | Docker repository name in Artifact Registry |
| `IMAGE_NAME` | `iris-svm-api` | Image name and Cloud Run **service** name |
| `IMAGE_TAG` | `v1` | Docker image tag |

You can also export variables manually instead of using `.env`:

```bash
export PROJECT_ID="your-unique-project-id"
export REGION="us-central1"
export AR_REPO="cloud-run-lab"
export IMAGE_NAME="iris-svm-api"
export IMAGE_TAG="v1"
```

### Step 1 – Setup GCP resources

```bash
cd scripts
./setup_gcp.sh
```

This will:

- Create the GCP project **if it does not exist**
- Enable Cloud Run, Artifact Registry, IAM, and related APIs
- Create the Artifact Registry Docker repository (if missing)
- Configure Docker authentication for the regional Artifact Registry hostname

**First run and billing:** Project creation may succeed, but **enabling APIs** requires a **billing account linked to the project**. If you see an error such as billing not found or APIs failing to enable, open **[Billing in the Google Cloud Console](https://console.cloud.google.com/billing)**, attach a billing account to this project, then run `./setup_gcp.sh` again. The script is safe to re-run: it skips steps that already completed (existing project, existing repository, and so on).

### Step 2 – Build and push the Docker image

```bash
./build_push.sh
```

To tag a new version:

```bash
IMAGE_TAG=v2 ./build_push.sh
```

### Step 3 – Deploy to Cloud Run

```bash
./deploy_cloud_run.sh
```

The script outputs the service URL. Each deploy creates a new **revision** in Cloud Run.

### Step 4 – Test the API

```bash
# Health check
curl https://<SERVICE_URL>/health

# Train the model
curl -X POST https://<SERVICE_URL>/train -H "Content-Type: application/json" -d '{}'

# Predict (setosa example)
curl -X POST https://<SERVICE_URL>/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

or use the swagger-ui at `https://<SERVICE_URL>/docs`

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/train` | POST | Train the SVM model (optional params: `test_size`, `random_state`, `C`, `kernel`) |
| `/predict` | POST | Predict species from features `[sepal_length, sepal_width, petal_length, petal_width]` |
| `/species` | GET | List species names |

### Example `/train` Request

```json
{
  "test_size": 0.2,
  "random_state": 42,
  "C": 1.0,
  "kernel": "rbf"
}
```

### Example `/predict` Request

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Response:

```json
{
  "species_id": 0,
  "species": "setosa"
}
```

---

## Versioning

- **Image tags:** Use `IMAGE_TAG` to push different versions (e.g., `v1`, `v2`, `git-abc123`)
- **Cloud Run revisions:** Each deploy creates a new revision. View them with:

```bash
gcloud run revisions list --service iris-svm-api --region us-central1
```

---

## Scaling Considerations

This lab uses `--max-instances=1` for simplicity. The trained model lives in instance memory (`/tmp`), so:

- With multiple instances, each would need its own `/train` call
- For production, persist models to Cloud Storage and load on startup

This is intentional for a learning lab—discuss trade-offs in class.

---

## Cleanup

From `scripts/`, you can use the teardown helper (prompts for confirmation unless `SKIP_PROMPT=1`):

```bash
./destroy_cloud_run.sh
```

To also remove the Artifact Registry tag and local Docker image: `DELETE_AR_IMAGE=1 DELETE_LOCAL_IMAGE=1 ./destroy_cloud_run.sh` (see script header).

Manual equivalent:

```bash
gcloud run services delete iris-svm-api --region us-central1 --quiet
gcloud artifacts docker images delete \
  ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME} --quiet
```

To delete the entire project:

```bash
gcloud projects delete ${PROJECT_ID}
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `PROJECT_ID already exists` | Choose a unique project ID |
| `Permission denied` pushing image | Run `gcloud auth configure-docker ${REGION}-docker.pkg.dev` |
| `503 Model not trained` | Call `/train` before `/predict` |
| Billing / `UREQ_PROJECT_BILLING_NOT_FOUND` when enabling APIs | Link a billing account to the project in [Billing](https://console.cloud.google.com/billing), then run `./setup_gcp.sh` again |
