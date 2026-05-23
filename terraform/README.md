# Terraform Provisioning (Create-First)

This directory provisions the cloud resources needed for:
- GKE cluster (zonal) for app + MLflow deployment
- Artifact Registry repository for container images
- GCS bucket for MLflow artifacts
- Dedicated VPC/subnet with secondary ranges for GKE

## 1) Prerequisites

- `terraform >= 1.5`
- `gcloud` authenticated to the target project
- Project APIs can be enabled by your account

```bash
gcloud auth application-default login
gcloud config set project <PROJECT_ID>
```

## 2) Configure variables

```bash
cp terraform/terraform.tfvars.example terraform/terraform.tfvars
```

Update at least:
- `project_id`
- `mlflow_artifacts_bucket_name` (must be globally unique)

## 3) Initialize backend (recommended GCS remote state)

Create state bucket first (one-time):

```bash
gsutil mb -l asia-southeast1 gs://tfstate-ztf-<random>
gsutil versioning set on gs://tfstate-ztf-<random>
```

Then init:

```bash
terraform -chdir=terraform init \
  -backend-config="bucket=tfstate-ztf-<random>" \
  -backend-config="prefix=face-spoofing-detection"
```

## 4) Plan and apply

```bash
terraform -chdir=terraform fmt
terraform -chdir=terraform validate
terraform -chdir=terraform plan -var-file=terraform.tfvars
terraform -chdir=terraform apply -var-file=terraform.tfvars
```

## 5) Connect to cluster

After apply, run output command:

```bash
terraform -chdir=terraform output gke_get_credentials_command
```

Then use it to fetch kubeconfig and verify:

```bash
kubectl get nodes
```
