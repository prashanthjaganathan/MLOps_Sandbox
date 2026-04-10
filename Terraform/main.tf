provider "google" {
  project = var.project_id
  region  = var.region
}

# -----------------------------------------------------------------------------
# Required APIs (explicit enablement is reliable on fresh projects)
# -----------------------------------------------------------------------------

resource "google_project_service" "run_api" {
  project            = var.project_id
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "compute_api" {
  project            = var.project_id
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "iam_api" {
  project            = var.project_id
  service            = "iam.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "storage_api" {
  project            = var.project_id
  service            = "storage.googleapis.com"
  disable_on_destroy = false
}

# -----------------------------------------------------------------------------
# Cloud Run v2 — listens on 8080 (matches sample FastAPI app)
# -----------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "api" {
  name     = var.cloud_run_service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.cloud_run_sa.email

    containers {
      image = var.container_image

      ports {
        container_port = 8080
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_project_service.run_api,
    google_service_account.cloud_run_sa,
  ]
}
