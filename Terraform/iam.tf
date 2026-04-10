resource "google_service_account" "cloud_run_sa" {
  account_id   = "cloud-run-sa"
  display_name = "Cloud Run microservice SA"

  depends_on = [google_project_service.iam_api]
}

# Least privilege: read objects in GCS (adjust bucket IAM separately as needed)
resource "google_project_iam_member" "cloud_run_storage_viewer" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# Public HTTPS access to this Cloud Run service only (not whole project)
resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  project  = var.project_id
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
