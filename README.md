# gmail-classifier

Semantic auto-labeling for Gmail using KNN on email embeddings.

## Quick start: git clone to running service

1. **Clone and install dependencies**

   ```bash
   git clone <repo-url>
   cd gmail-classifier
   make setup
   ```

2. **Set up Gmail credentials**

   - Create OAuth2 credentials (see [docs/gmail-setup.md](docs/gmail-setup.md))
   - Place `client_secret.json` in `credentials/`
   - Create Pub/Sub topic + subscription (see [docs/gmail-setup.md](docs/gmail-setup.md))

3. **Run the classifier** (first run triggers OAuth and bootstraps from Gmail)

   ```bash
   make watch             # Ctrl+C to stop
   ```

   The first boot fetches your labeled emails from Gmail and builds an
   embedding index locally (`data/state.db`). This takes 10–20 minutes; the
   service is live from the first second (new mail arriving during bootstrap
   is parked and classified once the index matures).

   Edit `config.toml` first if there are labels you don't want auto-applied
   (see [Configuration](#configuration)).

4. **Install as macOS service** (optional)

   ```bash
   make service-install   # generates runner, plist, control script
   make service-start
   make service-logs      # watch output
   ```

For GCP/Gmail API setup, see [docs/gmail-setup.md](docs/gmail-setup.md).

### macOS service notes

`make service-install` generates three files under `~/bin` and
`~/Library/LaunchAgents`:

- **Runner** (`~/bin/gmail-classifier-runner`) — sets up the environment,
  combines stdout+stderr into one log, and `exec`s `uv run`.
- **Plist** (`~/Library/LaunchAgents/com.xnodet.gmail-classifier.plist`) —
  `KeepAlive=true` so launchd restarts it on crash; `RunAtLoad` starts it at
  login.
- **Control script** (`~/bin/gmail-classifierctl`) — wraps `launchctl`
  bootstrap/bootout/kickstart into `start`/`stop`/`restart`/`reload`/`status`/
  `logs`/`rotate-log`/`enable`/`disable`.

Use `bootout` (via `make service-stop`) to stop — a plain `kill` under
`KeepAlive=true` just respawns the process. `make service-restart` uses
`kickstart -k` for an atomic restart.

**Troubleshooting** if the service starts in Terminal but not from launchd:

- Validate the plist: `plutil -lint ~/Library/LaunchAgents/com.xnodet.gmail-classifier.plist`
- Check the log: `make service-logs` (or `tail -F ~/Library/Logs/com.xnodet.gmail-classifier.log`)
- Check launchd state: `launchctl print gui/$(id -u)/com.xnodet.gmail-classifier`
- Common causes: wrong `uv` path (use `command -v uv`), missing credentials,
  environment variables that exist in your shell but not under launchd (make
  them explicit in the runner or plist), or macOS privacy controls blocking
  access to files.

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
what gets auto-applied everywhere, including the macOS and GCP services.
`config.toml` ships with the code, so editing exclusions is a normal code
redeploy: the next boot reconciles membership (drops now-excluded labels,
bootstraps newly-included ones) **without** a full re-embed of unchanged
messages.

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
make gcp-create        # 1. Create the VM (if not already created)
make gcp-deploy        # 2. Deploy code + credentials (no DB upload)
make gcp-start         # 3. Start the service — it bootstraps from Gmail
make gcp-logs          # 4. Watch the first-boot bootstrap progress
make gcp-state-status  # 5. Inspect the store (counts, cursor, fingerprint)
```

Deploy ships **code + credentials only** — no data upload. The VM fetches
labeled mail from Gmail and builds its derived `state.db` on first boot. That
first boot is slow (roughly 10–20 min while it embeds the corpus), but the
service is *live and safe from the first second*: it never labels or archives
the pre-existing backlog, only mail that arrives after it starts. Restarts
afterward are fast (it reads `state.db`, no re-fetch).

Because bodies are never persisted, **changing the embedding model or the text
representation forces a re-fetch**: the next boot notices the fingerprint changed
and rebuilds `state.db` from Gmail automatically (carrying the history cursor
forward, no live mail skipped). A pure `config.toml` exclusion change is cheaper —
it reconciles membership without re-embedding.

To wipe and re-bootstrap from scratch: `make gcp-reset-state` (stops the service
and removes `state.db` + its SQLite sidecars). It leaves the service **stopped**;
run `make gcp-start` when you want it to bootstrap fresh from Gmail. Locally,
`make reset-state` does the same to `data/state.db`.

`state.db` is private derived data: it holds no bodies/subjects/senders but does
contain message ids, label ids/names, and embeddings. It is never uploaded from
the VM by default.

### Updating

After code changes, redeploy with `make gcp-deploy` then `make gcp-start`.
Deploy stops the service (to avoid corrupting SQLite files mid-sync), syncs code,
and installs dependencies, but leaves the service stopped — start it again with
`make gcp-start`. A redeploy preserves `state.db` (fast restart, no
re-bootstrap).

### Debugging

[Google Cloud console](https://console.cloud.google.com/compute/instancesDetail/zones/us-central1-a/instances/gmail-classifier?project=classy-498012)
Access to the log: ``sudo journalctl -u gmail-classifier -f`` (or `make gcp-logs`).

The service prefixes every per-message log line with current RSS, so memory
behavior is visible directly in the log. Expected steady-state is ~220 MB on the
e2-micro; startup briefly peaks higher (transient, returned to the OS by
`malloc_trim`).

