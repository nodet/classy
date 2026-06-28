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

4. Ensure billing is enabled (required even for free-tier resources):

   ```bash
   gcloud billing accounts list
   gcloud billing projects link classy-498012 --billing-account=<BILLING_ACCOUNT_ID>
   ```

   If you don't have a billing account, create one at the
   [GCP billing console](https://console.cloud.google.com/billing)
   (credit card required, but e2-micro in us-central1 is free).

5. Enable Compute Engine API (first time only):

   ```bash
   gcloud services enable compute.googleapis.com
   ```

6. Verify:

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

### Debugging

[Google Cloud console](https://console.cloud.google.com/compute/instancesDetail/zones/us-central1-a/instances/gmail-classifier?project=classy-498012)
Access to the log: ``sudo journalctl -u gmail-classifier -f``

```text
$ vmstat 2
procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----
 r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st
 2  5      0  64932    952  41588    0    0  1815    21  220  258  0  1 60 38  0
 1  6      0  64932    952  41588    0    0  4856     0  854  884  0  1  0 99  0
 1  7      0  64932    952  41588    0    0  4772     2  962  976  0  2  0 98  0
 3  7      0  64932    952  41588    0    0  4462     6  902  959  1  1  0 98  0
 1  9      0  64932    952  41588    0    0  4542     2  755  876  0  1  0 99  0
 1  6      0  64932    952  41588    0    0  4440     0  720  834  1  1  0 99  0
 1  6      0  64932    952  41588    0    0  5016     4  651  784  0  1  0 99  0
 1  6      0  64932    952  41588    0    0  4430     0  783  904  5  2  0 93  0
 2  5      0  64932    952  41588    0    0  4724     0  806  840 17  5  0 79  0
 2  5      0  64932    952  41588    0    0  5200     0  823  906  1  1  0 98  0
 2  6      0  64932    952  41588    0    0  4796     0  925 1031  0  1  0 99  0
 ```

- bi (blocks in): 4000-7000 KB/s — that's your ~5 MB/s constant disk read, matching the GCP console
- wa (I/O wait): 79-99% — CPU is almost entirely idle waiting for disk
- b (blocked processes): 5-18 — many threads blocked on I/O
- us (user CPU): 0-5% — barely any actual computation happening
- si/so (swap): 0/0 — no swapping (good news: it's not a RAM problem)
- free: 64 MB — low but stable; cache is 41 MB

Once the service has been stopped:

```text
$ vmstat 2
procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----
 r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st
 2  2      0 665060   4864 138984    0    0  1825    21  224  262  0  1 59 40  0
 1  1      0 665060   4864 138984    0    0  3872     0 1136 1392  4  3 39 54  0
 1  2      0 665060   4864 138984    0    0  3560   222 1894 2641  6  8  8 78  0
 1  0      0 665060   4864 138984    0    0  3688     2  901 1103  4  5 47 44  0
 1  0      0 665060   4864 138984    0    0     0     0  124  109  1  1 99  0  0
 1  0      0 665060   4864 138984    0    0     0    38   98   89  0  0 99  0  0
 1  0      0 665060   4864 138984    0    0    56     2  145  125  1  1 99  0  0
 1  0      0 665060   4864 138984    0    0    96    90  126  125  0  1 98  1  0
 1  0      0 665060   4864 138984    0    0     0    18  117  114  0  1 99  0  0
 1  0      0 665060   4864 138984    0    0     0   826  154  135  1  1 98  1  0
 1  0      0 665060   4864 138984    0    0     6    14  106   89  0  1 99  1  0
 ```
