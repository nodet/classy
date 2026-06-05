# Plan: GCP e2-micro Deployment via `make gcp-*` Targets

## Context

The gmail-classifier runs successfully as a macOS launchd service (via `make service-install`). The project plan (Phase 4) recommends deploying to a GCP e2-micro VM for always-on operation without keeping a laptop open. The e2-micro is perpetually free in us-central1/us-east1/us-west1, same GCP project (`classy-498012`) as the existing Pub/Sub setup, and 1GB RAM is sufficient (measured ~120MB peak RSS).

Goal: `make gcp-create` provisions the VM, `make gcp-deploy` pushes code+data+credentials and configures systemd, operational targets provide start/stop/status/logs/ssh.

## Files to create/modify

| File | Purpose |
|------|---------|
| `/workspace/scripts/gcp-create.sh` | Provision e2-micro VM via gcloud |
| `/workspace/scripts/gcp-deploy.sh` | Deploy code, data, credentials; configure systemd |
| `/workspace/scripts/gcp-destroy.sh` | Teardown VM (with confirmation) |
| `/workspace/Makefile` | Add gcp-* targets |
| `/workspace/README.md` | Add GCP deployment section |

## Design decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Region | `us-central1-a` | Free tier, good Google API connectivity |
| External IP | Ephemeral (free) | Needed for outbound (Pub/Sub, Gmail API, pip). Cloud NAT costs money. |
| Service user | `gmail-classifier` (system, nologin) | Security isolation |
| Install path | `/opt/gmail-classifier` | Standard for services |
| Python mgmt | `uv` (same as macOS) | Consistent with local dev; `uv run --locked` |
| Code sync | tarball via `gcloud compute scp` | Simple; avoids rsync-over-gcloud complexity |
| Data sync | `gcloud compute scp` with size-check skip | Efficient for the 211MB training.db |
| Logs | journald (`journalctl -u gmail-classifier`) | Standard systemd; no rotation config needed |
| Restart policy | `Restart=on-failure`, 5 bursts / 5 min | Mirrors macOS KeepAlive + crash-loop protection |

## 1. `scripts/gcp-create.sh`

- Guard: check gcloud installed
- Check if instance exists (idempotent)
- `gcloud compute instances create gmail-classifier`:
  - `--project=classy-498012`
  - `--zone=us-central1-a`
  - `--machine-type=e2-micro`
  - `--image-family=debian-12 --image-project=debian-cloud`
  - `--boot-disk-size=30GB --boot-disk-type=pd-standard`
  - `--scopes=https://www.googleapis.com/auth/cloud-platform`
  - `--tags=gmail-classifier`
- Wait for SSH readiness (poll with gcloud ssh)
- Print: "Run 'make gcp-deploy' next."

## 2. `scripts/gcp-deploy.sh`

### Guards
- `data/training.db` exists and non-empty
- `data/inbox_sample.db` exists and non-empty
- `credentials/token.json` exists
- `credentials/client_secret.json` exists

### First-deploy detection
- Check if service user exists on VM (`id gmail-classifier`)
- If not: install deps (curl, rsync), create user, install uv

### Sync code
- Create tarball excluding `.git`, `__pycache__`, `.venv`, `data/`, `credentials/`, `.DS_Store`
- `gcloud compute scp` to `/tmp/` on VM
- Extract to `/opt/gmail-classifier`, chown to service user

### Sync data
- For each db: compare local size (`stat -f%z` macOS) vs remote (`stat -c%s` Linux)
- Skip if same; otherwise scp to `/tmp/` then `sudo mv`

### Sync credentials
- `mkdir -p /opt/gmail-classifier/credentials` with mode 700
- scp each file to `/tmp/`, `sudo mv`, `chmod 600`

### Install deps
- `sudo -u gmail-classifier -H bash -c 'cd /opt/gmail-classifier && $HOME/.local/bin/uv sync --locked'`

### Pre-warm model (first deploy only)
- `uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`

### Install systemd unit (generated inline via heredoc)
```ini
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
```

- `systemctl daemon-reload && systemctl enable gmail-classifier`
- If incremental deploy: `systemctl restart gmail-classifier`
- If first deploy: print "Start with: make gcp-start"

## 3. `scripts/gcp-destroy.sh`

- Interactive confirmation prompt
- `gcloud compute instances delete gmail-classifier --project=... --zone=... --quiet`

## 4. Makefile targets

```makefile
# --- GCP Deployment (e2-micro VM) ---

GCP_PROJECT  := classy-498012
GCP_INSTANCE := gmail-classifier
GCP_ZONE     := us-central1-a

gcp-create:   ## Create GCP e2-micro VM
gcp-deploy:   ## Deploy code, data, and credentials to GCP VM
gcp-destroy:  ## Delete the GCP VM (interactive confirmation)
gcp-start:    ## Start the classifier service on GCP
gcp-stop:     ## Stop the classifier service on GCP
gcp-restart:  ## Restart the classifier service on GCP
gcp-status:   ## Show service status on GCP
gcp-logs:     ## Tail service logs on GCP (last 20 lines + follow)
gcp-ssh:      ## Open SSH session to the GCP VM
```

Operational targets use inline `gcloud compute ssh --command="sudo systemctl ..."` / `sudo journalctl -u gmail-classifier -n 20 -f`.

## 5. Workflow

**First time:**
```
make gcp-create    # provision VM
make gcp-deploy    # install everything + download model
make gcp-start     # start the service
make gcp-logs      # verify
```

**Code update:**
```
make gcp-deploy    # syncs code, skips unchanged data, restarts service
```

**Data update (after make fetch-training on Mac):**
```
make gcp-deploy    # detects size change, uploads new db, restarts
```

## 6. Update README

Add a "GCP deployment" section to `/workspace/README.md` after the macOS service section:

```
## GCP deployment (always-on)

Requires `gcloud` CLI configured with project `classy-498012`.

1. Create the VM:        make gcp-create
2. Deploy everything:    make gcp-deploy
3. Start the service:    make gcp-start
4. Check status:         make gcp-status
5. Tail logs:            make gcp-logs
6. SSH into VM:          make gcp-ssh
7. Destroy VM:           make gcp-destroy
```

## 7. Verification

1. `make gcp-create` completes, instance visible in `gcloud compute instances list`
2. `make gcp-deploy` succeeds (first-time: installs uv, downloads model)
3. `make gcp-start` → service running
4. `make gcp-status` → shows active PID
5. `make gcp-logs` → shows classification output
6. `make gcp-stop` → "Stopped." in journal
7. `make gcp-destroy` → instance deleted

## Notes

- **Token refresh**: The deployed `token.json` has a refresh token. It works headlessly. If it ever expires (6mo inactivity), re-auth on Mac and `make gcp-deploy`.
- **Crash email alert**: The `_send_crash_alert()` in classify_and_label.py works on the VM too (uses same credentials).
- **No Docker**: uv manages the Python env directly, same as macOS. Simpler, less memory overhead.
