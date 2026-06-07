#!/bin/bash
set -euo pipefail

# Disable non-essential services on the GCP e2-micro to free RAM/CPU.

GCP_PROJECT="classy-498012"
GCP_ZONE="us-central1-a"
GCP_INSTANCE="gmail-classifier"

vm_run() {
    gcloud compute ssh "$GCP_INSTANCE" \
        --project="$GCP_PROJECT" --zone="$GCP_ZONE" \
        --command="$1"
}

echo "Disabling non-essential services..."

# OS Config Agent (patch management, inventory)
vm_run "sudo systemctl stop google-osconfig-agent 2>/dev/null || true && \
    sudo systemctl disable google-osconfig-agent 2>/dev/null || true"
echo "  Disabled: google-osconfig-agent"

# Cloud Ops Agent (monitoring/logging)
vm_run "sudo systemctl stop google-cloud-ops-agent 2>/dev/null || true && \
    sudo systemctl disable google-cloud-ops-agent 2>/dev/null || true && \
    sudo systemctl stop google-cloud-ops-agent-opentelemetry-collector 2>/dev/null || true && \
    sudo systemctl disable google-cloud-ops-agent-opentelemetry-collector 2>/dev/null || true && \
    sudo systemctl stop google-cloud-ops-agent-fluent-bit 2>/dev/null || true && \
    sudo systemctl disable google-cloud-ops-agent-fluent-bit 2>/dev/null || true"
echo "  Disabled: google-cloud-ops-agent"

# Apt daily timers (unattended upgrades, package list refresh)
vm_run "sudo systemctl stop apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true && \
    sudo systemctl disable apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true"
echo "  Disabled: apt-daily timers"

# Exim4 mail server (unused — classifier uses Gmail API)
vm_run "sudo systemctl stop exim4 2>/dev/null || true && \
    sudo systemctl disable exim4 2>/dev/null || true"
echo "  Disabled: exim4"

echo ""
echo "Done. Freed ~30-40MB RAM and ~3% CPU."
echo "Kept: google-guest-agent (SSH), systemd basics."
