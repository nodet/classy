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

3. **Fetch training data** (the first run also triggers the OAuth flow)

   ```bash
   make fetch-training    # downloads labeled emails (opens browser on first run)
   make fetch-inbox       # downloads inbox as skip examples
   ```

   By default this excludes the labels listed in `config.toml` — edit that
   first if there are labels you don't want auto-applied (see
   [Configuration](#configuration)).

4. **Verify it works interactively**

   ```bash
   make watch-pubsub      # Ctrl+C to stop
   ```

5. **Install as macOS service**

   ```bash
   make service-install   # generates runner, plist, control script
   ```

6. **Start the service**

   ```bash
   make service-start
   make service-logs      # watch output
   ```

For detailed launchd configuration, see [mac_uv_launchd_service_plan.md](mac_uv_launchd_service_plan.md).
For GCP/Gmail API setup, see [docs/gmail-setup.md](docs/gmail-setup.md).

## How it works

The classifier never reads rules you write. It learns purely by example from
the emails you have already labeled, by comparing a new email to past ones and
copying the label of its closest matches. Three ideas make that work.

### 1. Embeddings — turning an email into a vector

An **embedding** is a fixed-length list of numbers (a vector) that captures the
*meaning* of a piece of text, produced by a machine-learning model. The key
property: texts that mean similar things get vectors that point in similar
directions, even when they share no words. "Your flight is confirmed" and "Booking
reference for your trip" land near each other; a tech newsletter lands far away.

This project uses **`all-MiniLM-L6-v2`**, a small sentence-embedding model from the
[sentence-transformers](https://www.sbert.net/) family, run locally via
[FastEmbed](https://github.com/qdrant/fastembed) (so no text is sent to any
external API). It maps each email to a **384-dimensional, unit-length vector**.
It is small (~90 MB), fast on CPU, and good enough for short texts like emails —
which is why it runs comfortably on a free-tier VM.

Before embedding, each email is reduced to one text string
(`preprocessing.py`): the sender, the subject, and a cleaned body (HTML
stripped, quoted replies / forwarded blocks / signatures removed, truncated to
~400 words), plus the mailing-list id if present. That string — not the raw
HTML — is what gets embedded.

### 2. KNN — classifying by nearest neighbors

Classification uses **k-nearest-neighbors (KNN)**, which has no separate
"training" step in the usual sense: the model *is* the set of past examples.
To classify a new email:

1. Embed it into a vector.
2. Find the **k = 5** most similar labeled emails, where similarity is
   **cosine similarity** (the angle between two vectors — 1.0 means identical
   direction, 0.0 means unrelated). Unit-length vectors make this just a dot
   product.
3. Each of those 5 neighbors votes for its own label, weighted by how similar
   it is. Summing the weights per label gives a score per label.
4. **Confidence** = winning label's score ÷ total score (0–1). High confidence
   means the neighbors agree strongly.

A label is only eligible to win if it has at least **5 training examples**, so a
single oddball email can't create a category.

### 3. Confidence thresholds and the skip pool

The confidence decides what actually happens (`classifier.py`):

| Confidence | Action |
|---|---|
| ≥ 0.80 | apply the label and archive the email |
| < 0.80 | do nothing (leave it unlabeled in the inbox) |

There is a finer internal distinction at **0.95**: predictions in the
0.80–0.95 band are tagged `LABEL_WITH_REVIEW` rather than `LABEL`. The live
service treats both the same — it applies and archives either way — so in normal
operation there is nothing to review. The split only surfaces in `make dry-run`,
which groups its output into "sure" (≥0.95), "review" (0.80–0.95), and "low"
(<0.80) so you can eyeball where the borderline calls fall before trusting them.

To stop the classifier from labeling mail that *should* just stay in the inbox,
a sample of inbox messages is loaded as negative examples under a special
`__skip__` label. These vote like any other neighbor; if `__skip__` wins, or
even just dilutes the confidence below threshold, the email is left alone. This
is why the README talks about a "skip pool" alongside the training data.

### Learning continuously

Because the model is just the example set, it adapts the moment you correct it.
When you move a message to a label (or out of one), the service re-embeds it and
updates its in-memory index immediately — the next similar email benefits right
away, with no retraining step. Gmail labels are the single source of truth; the
local databases are just a cache of those examples and their vectors.

## Configuration

Tunable settings live in `config.toml` at the repo root — edit it directly, no
code or Makefile changes needed.

```toml
[labels]
# Gmail label names to exclude everywhere: never fetched, never trained on,
# never auto-applied. Replace with your own labels, or leave empty to
# classify into every user label.
excluded = ["XLC", "XLE", "XLCap"]
```

Every command reads exclusions from this file — change the list here to change
what gets fetched, trained on, and auto-applied everywhere, including the macOS
and GCP services.

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
make embed         # 1. Build the embedding cache locally (avoids OOM on the VM)
make gcp-create    # 2. Create the VM
make gcp-deploy    # 3. Deploy code, data, credentials, install deps
make gcp-start     # 4. Start the service
make gcp-status    # 5. Check status
make gcp-logs      # 6. Tail logs (Ctrl+C to stop)
make gcp-ssh       # 7. SSH into VM (for debugging)
make gcp-destroy   # 8. Destroy VM (when no longer needed)
```

`make gcp-deploy` stops the service before syncing and does **not** restart it;
run `make gcp-start` afterwards (already step 4 above). To restart an
already-deployed service without redeploying, use `make gcp-restart`.

### Updating

After code changes, run `make gcp-deploy` then `make gcp-start`. Deploy stops the
service (to avoid corrupting the SQLite files mid-sync), syncs code, skips
unchanged data files, and installs dependencies, but leaves the service stopped --
you must start it again with `make gcp-start`.

After retraining on the Mac (`make fetch-training` / `make fetch-inbox`), rebuild
the embedding cache with `make embed` before deploying, otherwise the VM has to
embed the newly-fetched messages at startup and can run out of memory.
`make gcp-deploy` detects changed databases by size/mtime and uploads them.

### Debugging

[Google Cloud console](https://console.cloud.google.com/compute/instancesDetail/zones/us-central1-a/instances/gmail-classifier?project=classy-498012)
Access to the log: ``sudo journalctl -u gmail-classifier -f`` (or `make gcp-logs`).

The service prefixes every per-message log line with current RSS, so memory
behavior is visible directly in the log. Expected steady-state is ~220 MB on the
e2-micro; startup briefly peaks higher (transient, returned to the OS by
`malloc_trim`).

#### Historical: I/O thrashing (resolved)

The VM was once pinned at 79-99% I/O wait. This was traced *not* to swapping
(`si/so = 0`) but to ~600 MB held resident causing kernel memory pressure and
constant disk reads. Fixed by excluding unused labels, deferring text prep to
cache misses, embedding one message at a time, and trimming the heap after each
batch. Kept here as a diagnostic example of reading `vmstat` on this VM:

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
