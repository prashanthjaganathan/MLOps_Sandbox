variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for regional resources (Cloud Run, subnet, etc.)"
  type        = string
  default     = "us-central1"
}

variable "container_image" {
  description = "Container image URL (Artifact Registry, GCR, or Docker Hub)"
  type        = string
}

variable "cloud_run_service_name" {
  description = "Cloud Run (v2) service name"
  type        = string
  default     = "fastapi-service"
}
