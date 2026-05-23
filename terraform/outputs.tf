output "gke_cluster_name" {
  description = "Provisioned GKE cluster name."
  value       = google_container_cluster.primary.name
}

output "gke_cluster_location" {
  description = "Provisioned GKE cluster location."
  value       = google_container_cluster.primary.location
}

output "gke_get_credentials_command" {
  description = "Command to fetch kubeconfig credentials for kubectl."
  value       = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --zone ${google_container_cluster.primary.location} --project ${var.project_id}"
}

output "artifact_registry_repository" {
  description = "Artifact Registry docker repository path."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.liveness.repository_id}"
}

output "mlflow_artifacts_bucket" {
  description = "GCS bucket for MLflow artifacts."
  value       = google_storage_bucket.mlflow_artifacts.name
}

output "vpc_network_name" {
  description = "VPC network name used by GKE."
  value       = google_compute_network.vpc.name
}
