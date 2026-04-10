# Terraform — FastAPI on Google Cloud Run (plus VPC + IAM)

## Comparsion with [MLOps Repo Lab](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Terraform_Labs/GCP/Lab1_Beginner)

- the course lab is the **intro** track: fewer resources, usually **local state**, focused on Terraform workflow on GCP.
- **This folder** goes further: **Cloud Run v2** + **Docker image** you build/push, **custom VPC**, **IAM** (service account + scoped `run.invoker`), split across multiple `.tf` files, and optional **GCS remote state**—closer to a small **MLOps** deploy, with the operational gotchas called out in this README.

## What each file is for


| File                       | Role                                                                                          |
| -------------------------- | --------------------------------------------------------------------------------------------- |
| `versions.tf`              | Terraform & Google provider version constraints                                               |
| `backend.tf`               | **Remote state**: GCS `bucket` + `prefix` (see §0 — bucket must exist first)                  |
| `variables.tf`             | Input variables (`project_id`, `region`, `container_image`, …)                                |
| `terraform.tfvars`         | **Your** values (create from example; **gitignored**)                                         |
| `terraform.tfvars.example` | Template to copy                                                                              |
| `main.tf`                  | Google provider, API enablement, **Cloud Run v2** service                                     |
| `network.tf`               | VPC, subnet, firewall (parallel after VPC)                                                    |
| `iam.tf`                   | Service account, **project** IAM for GCS read, **service-level** `run.invoker` for `allUsers` |
| `outputs.tf`               | Prints service URL, SA email, VPC id after apply                                              |
| `app/`                     | Sample **Dockerfile + FastAPI** you build and push                                            |


Terraform **merges all `*.tf` in this directory** into one configuration.

## Prerequisites

- [Terraform](https://developer.hashicorp.com/terraform/install) ≥ 1.0 (see `versions.tf`)
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`)
- [Docker](https://docs.docker.com/get-docker/) (build/push the sample image)
- GCP **project** with **billing** enabled
- **Application Default Credentials** for Terraform + GCS backend (not the same as `gcloud auth login` alone):
  ```bash
  gcloud auth application-default login
  ```
  Or a service account key:
  ```bash
  export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
  ```

## 0. Bootstrap the state bucket (GCS backend)

`backend.tf` points state at a **GCS bucket that must already exist**. Terraform will not create that bucket for you in the same configuration.

1. Pick a **globally unique** bucket name (e.g. `YOUR_PROJECT_ID-tf-state`).
2. Ensure you have a project and billing, then create the bucket (skip `projects create` if you already have a project):
  ```bash
   gcloud config set project YOUR_PROJECT_ID
   gsutil mb -p YOUR_PROJECT_ID -l us-central1 gs://YOUR_BUCKET_NAME
  ```
   If you see **409 already exists**, the bucket is fine — do not create it again.
3. Edit `backend.tf`. Set `**bucket` to the bucket name only** — no `gs://`, no `gs:/`:
  ```hcl
   bucket = "YOUR_BUCKET_NAME"
  ```
   A value like `gs:/my-bucket` or `gs://my-bucket` is wrong and leads to `**bucket doesn't exist**` on `terraform init` even when the bucket is real.
4. Run `**gcloud auth application-default login**` before `terraform init` so the GCS backend can access the bucket.

**Local state first:** Comment out the entire `backend "gcs" { ... }` block in `backend.tf`, run `terraform init`, apply, then migrate using HashiCorp’s [GCS backend](https://developer.hashicorp.com/terraform/language/settings/backends/gcs) docs.

**Changing backend config:** After editing `backend.tf`, use `terraform init -reconfigure`.

## 1. Configure variables

```bash
cd Terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

- `project_id` — GCP project id  
- `region` — e.g. `us-central1`  
- `container_image` — **exact** image URL **including the tag** you pushed (e.g. `:v1`). There is no automatic `latest` tag unless you pushed an image tagged `latest`.

## 2. Build and push the sample image

Cloud Run pulls a **pre-built** image from **Artifact Registry** (or GCR). The tag in `terraform.tfvars` must match what you push.

**Cloud Run expects `linux/amd64`.** On **Apple Silicon (M1/M2/M3)**, a plain `docker build` often produces **arm64**, which fails with a message about **OCI image index** and **amd64/linux**. Use `**--platform linux/amd64`** for everyone building for Cloud Run; on Arm Macs the first build may be slower (emulation).

```bash
export PROJECT_ID=your-project-id
export REGION=us-central1

gcloud services enable artifactregistry.googleapis.com --project "$PROJECT_ID"

gcloud artifacts repositories create fastapi-repo \
  --repository-format=docker \
  --location="$REGION" \
  --project="$PROJECT_ID" \
  --description="FastAPI lab"

gcloud auth configure-docker "${REGION}-docker.pkg.dev"

docker build --platform linux/amd64 \
  -t "${REGION}-docker.pkg.dev/${PROJECT_ID}/fastapi-repo/fastapi-app:v1" \
  ./app

docker push "${REGION}-docker.pkg.dev/${PROJECT_ID}/fastapi-repo/fastapi-app:v1"
```

Set in `terraform.tfvars` (match project, region, and **tag**):

```hcl
container_image = "us-central1-docker.pkg.dev/your-project-id/fastapi-repo/fastapi-app:v1"
```

Use the **same region** as `var.region` to avoid unnecessary cross-region pulls.

**List tags in the repo:**

```bash
gcloud artifacts docker images list "${REGION}-docker.pkg.dev/${PROJECT_ID}/fastapi-repo" --include-tags
```

## 3. Terraform workflow

```bash
cd Terraform

terraform init
# If you changed backend.tf or providers: terraform init -reconfigure
# To refresh providers: terraform init -upgrade

terraform fmt -recursive
terraform validate

terraform plan -out=tfplan
terraform apply tfplan
```

Or:

```bash
terraform apply
```

If a previous **apply failed partway** (e.g. only Cloud Run failed), fix the cause (image URL, platform) and run `**terraform apply` again** — other resources stay in state.

### After apply

- Outputs: `service_url`, `service_account_email`, `vpc_id`, `vpc_name`  
- Or: `terraform output -raw service_url`

## 4. Test the service

```bash
URL=$(terraform output -raw service_url)
curl -sS "${URL}/health"
curl -sS "${URL}/"
```

Expect JSON like `{"healthy":true}` from `/health`.

## 5. Destroy (cleanup)

```bash
terraform destroy
```

Resources are removed in dependency order. The **GCS state bucket** is **not** deleted by this stack.
