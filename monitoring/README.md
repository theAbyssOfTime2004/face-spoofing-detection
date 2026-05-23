# N8 Observability Runbook

This folder contains the manifests and values needed for N8:

- `kube-prometheus-stack-values.yaml`: Prometheus and Grafana overrides for ServiceMonitor discovery and dashboard sidecar.
- `servicemonitor.yaml`: scrape `liveness-api` `/metrics` every 30 seconds.
- `grafana-dashboard-configmap.yaml`: auto-import dashboard through Grafana sidecar.
- `grafana-dashboard.json`: dashboard source file.

## 1) Install kube-prometheus-stack

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  -n monitoring \
  --create-namespace \
  -f monitoring/kube-prometheus-stack-values.yaml
```

## 2) Apply ServiceMonitor and Grafana dashboard

```bash
kubectl apply -f monitoring/servicemonitor.yaml
kubectl apply -f monitoring/grafana-dashboard-configmap.yaml
```

## 3) Enable HPA for liveness-api

```bash
helm upgrade --install liveness ./helm/liveness-chart \
  --set autoscaling.enabled=true \
  --set autoscaling.minReplicas=2 \
  --set autoscaling.maxReplicas=5 \
  --set autoscaling.targetCPUUtilizationPercentage=70
```

## 4) Verify

```bash
kubectl get servicemonitor -n monitoring
kubectl get hpa
kubectl get pods -n monitoring
```

Forward Grafana:

```bash
kubectl port-forward svc/monitoring-grafana 3000:80 -n monitoring
```

Get Grafana admin password:

```bash
kubectl get secret monitoring-grafana -n monitoring -o jsonpath="{.data.admin-password}" | base64 --decode; echo
```

Generate traffic to feed dashboard and Prometheus:

```bash
LB_IP=<your-liveness-loadbalancer-ip>
curl -s -X POST "http://$LB_IP/predict" -F "file=@/absolute/path/to/data/test/normal/1490_4.jpg" > /dev/null
curl -s -X POST "http://$LB_IP/predict" -F "file=@/absolute/path/to/data/test/spoof/151_4.jpg" > /dev/null
```
