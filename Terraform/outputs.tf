output "service_url" {
  description = "HTTPS URL of the Cloud Run service"
  value       = google_cloud_run_v2_service.api.uri
}

output "service_account_email" {
  description = "Dedicated service account used by Cloud Run"
  value       = google_service_account.cloud_run_sa.email
}

output "vpc_id" {
  description = "Custom VPC id"
  value       = google_compute_network.vpc.id
}

output "vpc_name" {
  description = "Custom VPC name"
  value       = google_compute_network.vpc.name
}