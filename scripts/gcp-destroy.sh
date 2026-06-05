#!/bin/bash
set -euo pipefail

# Destroy the gmail-classifier GCP VM (with confirmation).

GCP_PROJECT="classy-498012"
GCP_ZONE="us-central1-a"
GCP_INSTANCE="gmail-classifier"

# Guard: gcloud must be installed
if ! command -v gcloud &>/dev/null; then
    echo "Error: gcloud CLI not found."
    exit 1
fi

# Check if instance exists
if ! gcloud compute instances describe "$GCP_INSTANCE" \
    --project="$GCP_PROJECT" --zone="$GCP_ZONE" &>/dev/null; then
    echo "Instance '$GCP_INSTANCE' does not exist. Nothing to destroy."
    exit 0
fi

echo "This will permanently delete the VM '$GCP_INSTANCE' in $GCP_ZONE."
echo "All data on the VM will be lost."
echo ""
read -r -p "Are you sure? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 1
fi

echo "Deleting instance '$GCP_INSTANCE'..."
gcloud compute instances delete "$GCP_INSTANCE" \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE" \
    --quiet

echo "Instance deleted."
