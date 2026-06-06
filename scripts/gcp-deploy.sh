#!/bin/bash
set -euo pipefail

# Deploy gmail-classifier to GCP e2-micro VM.

GCP_PROJECT="classy-498012"
GCP_ZONE="us-central1-a"
GCP_INSTANCE="gmail-classifier"
SERVICE_USER="gmail-classifier"
INSTALL_DIR="/opt/gmail-classifier"

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_DIR"

# --- Guards ---

if [[ ! -s "data/training.db" ]]; then
    echo "Error: data/training.db missing or empty. Run 'make fetch-training' first."
    exit 1
fi

if [[ ! -s "data/inbox_sample.db" ]]; then
    echo "Error: data/inbox_sample.db missing or empty. Run 'make fetch-inbox' first."
    exit 1
fi

if [[ ! -f "credentials/token.json" ]]; then
    echo "Error: credentials/token.json missing. Run 'make fetch-training' to trigger OAuth."
    exit 1
fi

if [[ ! -f "credentials/client_secret.json" ]]; then
    echo "Error: credentials/client_secret.json missing."
    exit 1
fi

# Helper: run command on the VM
vm_run() {
    gcloud compute ssh "$GCP_INSTANCE" \
        --project="$GCP_PROJECT" --zone="$GCP_ZONE" \
        --command="$1"
}

# Helper: copy file to VM
vm_scp() {
    gcloud compute scp "$1" "$GCP_INSTANCE:$2" \
        --project="$GCP_PROJECT" --zone="$GCP_ZONE"
}

# --- First-deploy detection ---

FIRST_DEPLOY=false
if ! vm_run "id $SERVICE_USER" &>/dev/null; then
    FIRST_DEPLOY=true
    echo "First deploy detected. Setting up VM..."

    vm_run "sudo apt-get update -qq && sudo apt-get install -y -qq curl"

    # Create service user with home at install dir
    vm_run "sudo useradd --system --shell /usr/sbin/nologin --home-dir $INSTALL_DIR --create-home $SERVICE_USER"

    # Install uv for the service user
    vm_run "sudo -u $SERVICE_USER -H bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'"

    echo "VM setup complete."
fi

# --- Sync code ---

echo "Syncing code..."
TARBALL="/tmp/gmail-classifier-code.tar.gz"
tar czf "$TARBALL" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='.venv' \
    --exclude='data' \
    --exclude='credentials' \
    --exclude='.DS_Store' \
    --exclude='*.pyc' \
    .

vm_scp "$TARBALL" "/tmp/gmail-classifier-code.tar.gz"
rm -f "$TARBALL"

vm_run "sudo mkdir -p $INSTALL_DIR && \
    sudo tar xzf /tmp/gmail-classifier-code.tar.gz -C $INSTALL_DIR && \
    sudo mkdir -p $INSTALL_DIR/data $INSTALL_DIR/credentials $INSTALL_DIR/.cache && \
    sudo chown -R $SERVICE_USER:$SERVICE_USER $INSTALL_DIR && \
    rm -f /tmp/gmail-classifier-code.tar.gz"

# --- Sync data ---

echo "Syncing data..."
sync_file() {
    local local_path="$1"
    local remote_path="$2"
    local filename
    filename=$(basename "$local_path")

    local local_size
    # macOS stat vs Linux stat
    if [[ "$(uname)" == "Darwin" ]]; then
        local_size=$(stat -f%z "$local_path")
    else
        local_size=$(stat -c%s "$local_path")
    fi

    local remote_size
    remote_size=$(vm_run "stat -c%s $remote_path 2>/dev/null || echo 0")

    if [[ "$local_size" == "$remote_size" ]]; then
        echo "  $filename: unchanged (${local_size} bytes), skipping."
    else
        echo "  $filename: uploading (${local_size} bytes)..."
        vm_scp "$local_path" "/tmp/$filename"
        vm_run "sudo mv /tmp/$filename $remote_path && sudo chown $SERVICE_USER:$SERVICE_USER $remote_path"
    fi
}

sync_file "data/training.db" "$INSTALL_DIR/data/training.db"
sync_file "data/inbox_sample.db" "$INSTALL_DIR/data/inbox_sample.db"
if [[ -s "data/embeddings.db" ]]; then
    sync_file "data/embeddings.db" "$INSTALL_DIR/data/embeddings.db"
fi

# --- Sync credentials ---

echo "Syncing credentials..."
vm_run "sudo chmod 700 $INSTALL_DIR/credentials"

for cred_file in credentials/token.json credentials/client_secret.json; do
    filename=$(basename "$cred_file")
    vm_scp "$cred_file" "/tmp/$filename"
    vm_run "sudo mv /tmp/$filename $INSTALL_DIR/credentials/$filename && \
        sudo chown $SERVICE_USER:$SERVICE_USER $INSTALL_DIR/credentials/$filename && \
        sudo chmod 600 $INSTALL_DIR/credentials/$filename"
done

# --- Install dependencies ---

echo "Installing Python dependencies..."
vm_run "sudo -u $SERVICE_USER -H bash -c 'cd $INSTALL_DIR && \$HOME/.local/bin/uv sync --locked'"

# --- Pre-warm model (first deploy only) ---

if [[ "$FIRST_DEPLOY" == "true" ]]; then
    echo "Pre-warming FastEmbed model (this may take a minute)..."
    vm_run "sudo -u $SERVICE_USER -H bash -c 'cd $INSTALL_DIR && \$HOME/.local/bin/uv run python -c \"from fastembed import TextEmbedding; TextEmbedding(\\\"sentence-transformers/all-MiniLM-L6-v2\\\")\"'"
fi

# --- Install systemd unit ---

echo "Installing systemd service..."
vm_run "sudo tee /etc/systemd/system/gmail-classifier.service > /dev/null << 'UNIT'
[Unit]
Description=Gmail Semantic Auto-Labeling Classifier
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=gmail-classifier
Group=gmail-classifier
WorkingDirectory=/opt/gmail-classifier
Environment=PYTHONUNBUFFERED=1
Environment=HOME=/opt/gmail-classifier
ExecStart=/opt/gmail-classifier/.local/bin/uv run --locked -- python -u scripts/classify_and_label.py --mode pubsub --exclude-labels XLC XLE XLCap
Restart=on-failure
RestartSec=10
StartLimitIntervalSec=300
StartLimitBurst=5
KillSignal=SIGTERM
TimeoutStopSec=30
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=/opt/gmail-classifier/data
ReadWritePaths=/opt/gmail-classifier/credentials
ReadWritePaths=/opt/gmail-classifier/.cache
ProtectHome=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
UNIT"

vm_run "sudo systemctl daemon-reload && sudo systemctl enable gmail-classifier"

# --- Start or restart ---

if [[ "$FIRST_DEPLOY" == "true" ]]; then
    echo ""
    echo "First deploy complete."
    echo "Start the service with: make gcp-start"
else
    echo "Restarting service..."
    vm_run "sudo systemctl restart gmail-classifier"
    echo ""
    echo "Deploy complete. Service restarted."
    echo "Check status with: make gcp-status"
fi
