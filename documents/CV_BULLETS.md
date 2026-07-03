# CV Bullets (Face Liveness MLOps)

- Productionized a face liveness detection system (89.28% accuracy, F1 90.10%) into a cloud-ready service on GKE using FastAPI, Docker, and Helm.
- Built IaC for GCP infrastructure (GKE, Artifact Registry, storage, IAM) using Terraform with reproducible environment setup.
- Implemented CI/CD with GitHub Actions and Workload Identity Federation, including automated testing, container build/push, and rolling deploys to Kubernetes.
- Added monitoring and SRE-ready visibility with Prometheus metrics and Grafana dashboards (request rate, error rate, and p50/p95/p99 latency).
- Integrated MLflow experiment tracking into training pipelines to log parameters, metrics, and model artifacts for reproducible model development.

## Short version (1-2 bullets)

- Built and deployed an end-to-end MLOps stack for face liveness detection on GKE: FastAPI serving, Docker/Helm delivery, Terraform IaC, MLflow tracking, Prometheus/Grafana monitoring, and GitHub Actions CI/CD.
- Delivered a production-style inference pipeline with automated rollout and observability, turning research code into a portfolio-ready cloud deployment.
