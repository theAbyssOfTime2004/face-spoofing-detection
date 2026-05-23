# N9 - GitHub Actions CI/CD (WIF + Auto Deploy)

This project uses GitHub Actions for CI/CD with 4 jobs:

1. `test`: run pytest unit tests only (`-m "not integration"`).
2. `lint`: run Helm lint, Terraform fmt check, and Ruff.
3. `build_push`: build Docker image, enforce image size guard (< 2GB), push to Artifact Registry.
4. `deploy`: run Helm upgrade on GKE and verify rollout.

Workflow file: `.github/workflows/ci.yml`

## Required GitHub Repository Variables

Go to `Settings -> Secrets and variables -> Actions -> Variables` and add:

- `GCP_PROJECT_ID`
- `GCP_REGION` (example: `asia-southeast1`)
- `AR_REPOSITORY` (example: `liveness-repo`)
- `AR_IMAGE_NAME` (example: `liveness-api`)
- `GKE_CLUSTER_NAME`
- `GKE_CLUSTER_LOCATION` (zone or region, must match existing GKE cluster)

## Required GitHub Repository Secrets (WIF)

Go to `Settings -> Secrets and variables -> Actions -> Secrets` and add:

- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_SERVICE_ACCOUNT`

## Trigger behavior

- On `pull_request` to `main`: run `test` + `lint`.
- On `push` to `main`: run `test` + `lint` + `build_push` + `deploy`.
- Manual run supported via `workflow_dispatch`.

## CI safety and anti-flaky policy

- Integration tests are excluded in CI using `pytest -m "not integration"`.
- Coverage threshold is enforced at 80% on `app`.
- Deploy is blocked if build or test/lint fails.

## Verification checklist after enabling variables/secrets

1. Create a test PR and confirm `test` + `lint` pass.
2. Merge PR to `main`.
3. Check workflow:
   - `build_push` succeeded and image tagged by commit SHA.
   - `deploy` succeeded.
4. Verify cluster rollout:
   - `kubectl rollout status deployment/liveness`
   - `curl http://<LB_IP>/health`

- CI dry-run note: trigger PR checks.
