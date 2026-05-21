terraform {
  backend "gcs" {}
}

# Example:
# terraform -chdir=terraform init \
#   -backend-config="bucket=tfstate-ztf-<random>" \
#   -backend-config="prefix=face-spoofing-detection"
