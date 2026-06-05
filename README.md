# gmail-classifier

Semantic auto-labeling for Gmail using KNN on email embeddings.

## Quick start: git clone to running service

1. **Clone and install dependencies**

   ```bash
   git clone <repo-url>
   cd gmail-classifier
   make setup
   ```

2. **Set up GCP credentials**

   - Create OAuth2 credentials (see [docs/gmail-setup.md](docs/gmail-setup.md))
   - Place `client_secret.json` in `credentials/`
   - Create Pub/Sub topic + subscription (see [docs/gmail-setup.md](docs/gmail-setup.md))

3. **Authenticate**

   ```bash
   make fetch-training    # triggers OAuth flow on first run
   ```

4. **Fetch training data**

   ```bash
   make fetch-training    # downloads labeled emails
   make fetch-inbox       # downloads inbox as skip examples
   ```

5. **Verify it works interactively**

   ```bash
   make watch-pubsub      # Ctrl+C to stop
   ```

6. **Install as macOS service**

   ```bash
   make service-install   # generates runner, plist, control script
   ```

7. **Start the service**

   ```bash
   gmail-classifierctl start
   gmail-classifierctl logs   # watch output
   ```

For detailed launchd configuration, see [mac_uv_launchd_service_plan.md](mac_uv_launchd_service_plan.md).
For GCP/Gmail API setup, see [docs/gmail-setup.md](docs/gmail-setup.md).

## GCP deployment (always-on)

Deploy to a free-tier e2-micro VM for always-on operation without keeping a laptop open.

### Prerequisites: install and configure gcloud CLI

1. Install the Google Cloud CLI:

   ```bash
   brew install --cask google-cloud-sdk
   ```

2. Authenticate:

   ```bash
   gcloud auth login
   ```

3. Set the project:

   ```bash
   gcloud config set project classy-498012
   ```

4. Enable Compute Engine API (first time only):

   ```bash
   gcloud services enable compute.googleapis.com
   ```

5. Verify:

   ```bash
   gcloud config list
   ```

### Deploy

```bash
make gcp-create    # 1. Create the VM
make gcp-deploy    # 2. Deploy code, data, credentials, install deps
make gcp-start     # 3. Start the service
make gcp-status    # 4. Check status
make gcp-logs      # 5. Tail logs (Ctrl+C to stop)
make gcp-ssh       # 6. SSH into VM (for debugging)
make gcp-destroy   # 7. Destroy VM (when no longer needed)
```

### Updating

After code changes, just run `make gcp-deploy` -- it syncs code, skips unchanged data files, and restarts the service automatically.

After retraining (`make fetch-training` on Mac), `make gcp-deploy` detects the size change and uploads the new database.
