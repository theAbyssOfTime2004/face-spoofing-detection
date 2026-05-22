# MLflow on GKE

This directory deploys a lightweight MLflow tracking server on the same GKE cluster.

## Prerequisites

- GKE cluster already provisioned and reachable via `kubectl`
- Terraform applied at least once after adding `roles/storage.objectAdmin` for node SA
- Artifact bucket exists (default in manifests: `gs://mlflow-artifacts-liveness-dev-99/mlflow`)

## Deploy

```bash
kubectl apply -f k8s/mlflow/namespace.yaml
kubectl apply -f k8s/mlflow/pvc.yaml
kubectl apply -f k8s/mlflow/deployment.yaml
kubectl apply -f k8s/mlflow/service.yaml
```

## Verify

```bash
kubectl -n mlflow get pods
kubectl -n mlflow get pvc
kubectl -n mlflow logs deploy/mlflow --tail=100
```

Port-forward UI:

```bash
kubectl -n mlflow port-forward svc/mlflow 5000:5000
```

Then open: `http://127.0.0.1:5000`

## Train scripts integration

`src/train_global.py` and `src/train_local.py` now auto-log runs if `mlflow` is installed.

Optional environment variables:

- `MLFLOW_TRACKING_URI` (for in-cluster MLflow: `http://mlflow.mlflow.svc.cluster.local:5000`)
- `MLFLOW_EXPERIMENT_NAME` (default: `face-spoofing-detection`)
