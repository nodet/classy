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
