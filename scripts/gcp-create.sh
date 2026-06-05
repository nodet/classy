#!/bin/bash
set -euo pipefail

# Provision a GCP e2-micro VM for gmail-classifier.

GCP_PROJECT="classy-498012"
GCP_ZONE="us-central1-a"
GCP_INSTANCE="gmail-classifier"

# Guard: gcloud must be installed
if ! command -v gcloud &>/dev/null; then
    echo "Error: gcloud CLI not found. Install with: brew install --cask google-cloud-sdk"
    exit 1
fi

# Idempotent: skip if instance already exists
if gcloud compute instances describe "$GCP_INSTANCE" \
    --project="$GCP_PROJECT" --zone="$GCP_ZONE" &>/dev/null; then
    echo "Instance '$GCP_INSTANCE' already exists in $GCP_ZONE."
    echo "Run 'make gcp-deploy' to deploy."
    exit 0
fi

echo "Creating e2-micro instance '$GCP_INSTANCE' in $GCP_ZONE..."
gcloud compute instances create "$GCP_INSTANCE" \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE" \
    --machine-type=e2-micro \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --boot-disk-size=30GB \
    --boot-disk-type=pd-standard \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=gmail-classifier

echo ""
echo "Waiting for SSH readiness..."
for i in $(seq 1 30); do
    if gcloud compute ssh "$GCP_INSTANCE" \
        --project="$GCP_PROJECT" --zone="$GCP_ZONE" \
        --command="echo ok" &>/dev/null; then
        echo "SSH ready."
        echo ""
        echo "Instance created successfully."
        echo "Run 'make gcp-deploy' next."
        exit 0
    fi
    sleep 2
done

echo "Warning: SSH not ready after 60s. The instance was created but may need more time to boot."
echo "Run 'make gcp-deploy' when ready."
