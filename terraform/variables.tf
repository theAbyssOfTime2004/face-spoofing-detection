variable "project_id" {
  description = "GCP project ID."
  type        = string
}

variable "region" {
  description = "Primary GCP region for Artifact Registry and buckets."
  type        = string
  default     = "asia-southeast1"
}

variable "zone" {
  description = "GKE zone for zonal cluster."
  type        = string
  default     = "asia-southeast1-b"
}

variable "environment" {
  description = "Environment label."
  type        = string
  default     = "dev"
}

variable "cluster_name" {
  description = "GKE cluster name."
  type        = string
  default     = "liveness-gke"
}

variable "network_name" {
  description = "VPC network name."
  type        = string
  default     = "liveness-vpc"
}

variable "subnet_name" {
  description = "Subnetwork name."
  type        = string
  default     = "liveness-subnet"
}

variable "subnet_cidr" {
  description = "Primary CIDR for subnet."
  type        = string
  default     = "10.80.0.0/20"
}

variable "pods_secondary_range_name" {
  description = "Secondary range name for GKE pods."
  type        = string
  default     = "pods-range"
}

variable "pods_secondary_cidr" {
  description = "Secondary CIDR block for GKE pods."
  type        = string
  default     = "10.84.0.0/14"
}

variable "services_secondary_range_name" {
  description = "Secondary range name for GKE services."
  type        = string
  default     = "services-range"
}

variable "services_secondary_cidr" {
  description = "Secondary CIDR block for GKE services."
  type        = string
  default     = "10.88.0.0/20"
}

variable "node_machine_type" {
  description = "Machine type for the GKE node pool."
  type        = string
  default     = "e2-standard-2"
}

variable "node_disk_size_gb" {
  description = "Node boot disk size in GB."
  type        = number
  default     = 100
}

variable "node_disk_type" {
  description = "Node boot disk type."
  type        = string
  default     = "pd-standard"
}

variable "node_count_min" {
  description = "Minimum node count for autoscaling pool."
  type        = number
  default     = 1
}

variable "node_count_max" {
  description = "Maximum node count for autoscaling pool."
  type        = number
  default     = 3
}

variable "artifact_registry_repo" {
  description = "Artifact Registry repository name for docker images."
  type        = string
  default     = "liveness-repo"
}

variable "artifact_registry_format" {
  description = "Artifact Registry format."
  type        = string
  default     = "DOCKER"
}

variable "mlflow_artifacts_bucket_name" {
  description = "GCS bucket name for MLflow artifacts."
  type        = string
}

variable "deletion_protection" {
  description = "Set true to protect GKE cluster from accidental deletion."
  type        = bool
  default     = false
}
