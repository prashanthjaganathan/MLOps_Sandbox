terraform {
  backend "gcs" {
    bucket = "mlops-terraform-labs-tf-state"
    prefix = "microservice/state"
  }
}
