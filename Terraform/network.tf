# Custom VPC for demonstrating networking. Cloud Run does not attach to this VPC
# unless you add a VPC connector (future extension). Subnet + firewall illustrate
# dependency ordering and parallel resources after the VPC exists.

resource "google_compute_network" "vpc" {
  name                    = "microservice-vpc"
  auto_create_subnetworks = false

  depends_on = [google_project_service.compute_api]
}

resource "google_compute_subnetwork" "subnet" {
  name          = "microservice-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.region
  network       = google_compute_network.vpc.id
}

resource "google_compute_firewall" "allow_http" {
  name    = "allow-http-8080"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["http-server"]
}
