# Gmail Semantic Auto-Labeling Service

## Goal

Automatically classify incoming Gmail messages into approximately a dozen user-defined categories (implemented as Gmail labels), using semantic similarity rather than sender-based rules.

Examples:

- Technology
- Optimization
- Conferences
- Customers
- Travel
- Personal
- Newsletters

Messages that don't clearly belong to any category remain unlabeled. Unlabeled messages are the priority tier: they surface in the inbox for direct attention. The system's job is to file away the noise so important emails are what remain.

The system should learn from existing labeled messages and from future manual corrections.

---

## High-Level Architecture

```text
Gmail
  |
  v
Gmail API
  |
  v
Classifier Service (Python)
  |
  +--> Embedding Model (local, sentence-transformers)
  |
  +--> Similarity Index (brute-force cosine similarity)
  |
  +--> Local Cache (SQLite)
  |
  +--> Gmail Labels
```

The source of truth is Gmail itself.

The local cache is disposable and can be rebuilt at any time.

---

## Initial Training

1. Create and maintain Gmail labels corresponding to categories.
2. Fetch historical messages carrying those labels.
3. For each message:
   - Extract sender
   - Extract subject
   - Extract relevant body text (see Preprocessing below)
4. Compute embeddings.
5. Build the similarity index.

The labeled historical messages become the training set.

A label should have at least 5-10 examples before the system attempts auto-classification for that category. Below that threshold, classification for that label is disabled.

---

## Email Preprocessing

Raw email bodies require cleaning before embedding:

1. Strip HTML (extract text content only).
2. Remove quoted replies (lines starting with `>`).
3. Remove forwarded message headers.
4. Trim signatures (detect `-- ` separator or common patterns).
5. Remove legal disclaimers and boilerplate footers.
6. Truncate to the first ~512 tokens (embedding model input limit).

The goal is to embed the meaningful content of the message, not the noise.

---

## Embedding Model

Use a local sentence-transformers model such as `all-MiniLM-L6-v2`:

- Free, no API dependency
- Fast inference (~5ms per message on CPU)
- ~80MB model size
- 384-dimensional embeddings
- Good quality for short text similarity

No external API calls needed for embeddings. The model runs inside the container.

If quality is insufficient, upgrade to a larger local model (e.g., `all-mpnet-base-v2`) or switch to an API-based model later.

---

## Classification Strategy

### Input Features

For each message, construct a text representation from:

- Sender address
- Sender name
- Subject
- Mailing-list headers (if present)
- First portion of the cleaned body

### Similarity-Based Classification

#### Cosine Similarity

For a new message with embedding `v` and a stored embedding `e_i`:

```
sim(v, e_i) = (v · e_i) / (||v|| * ||e_i||)
```

sentence-transformers outputs are unit-normalized, so this simplifies to the dot product:

```
sim(v, e_i) = v · e_i
```

Result is between -1 and 1. Higher means more semantically similar.

#### KNN Classification (K-Nearest Neighbors)

Labels often cover diverse subtopics. "Technology" might include hardware news, AI research, and developer tools — several clusters within one label. A single centroid per label would average these out and land close to nothing real.

KNN handles this naturally: it looks at individual training examples, not averages. If the new message is closest to 4 AI research emails and 1 hardware email, all labeled "Technology", it correctly classifies as Technology regardless of how diverse that label is overall.

**Algorithm:**

Find the K most similar training examples. Each neighbor votes for its label, weighted by similarity:

```
score(L) = sum of sim(v, e_i) for the K nearest neighbors that have label L
```

K=5 is a reasonable default. The system does not need to know or discover how many subclusters exist within a label — it works regardless.

**Why not centroids:**

Centroid-based classification (one mean vector per label) is simpler and faster, but assumes each label forms a single coherent cluster in embedding space. In practice, labels span multiple senders, topics, and writing styles. Centroids would require explicit sub-clustering (choosing how many clusters per label), which adds complexity and parameters to tune. KNN avoids this entirely.

**Why not all-vote:**

Having all training examples vote (weighted by similarity) biases toward labels with more examples. Normalizing by label count is mathematically equivalent to centroid comparison, which has the same single-cluster problem.

#### Confidence Score

```
confidence = score(winning_label) / sum of score(L) for all labels L
```

This gives a probability-like value between 0 and 1. High confidence means the nearest neighbors strongly agree on one label. Low confidence means the neighbors are split across multiple labels.

Alternative (margin-based):

```
confidence = score(winning_label) - score(second_best_label)
```

Both work. The ratio-based approach maps more naturally to the percentage thresholds (95%, 80%) defined in the confidence levels section.

#### Workflow

For a new message:

1. Compute its embedding `v`.
2. Compute cosine similarity against all stored training embeddings.
3. Take the top K=5 most similar examples.
4. Sum their similarity scores per label.
5. Compute confidence for the winning label.
6. If confidence is above threshold, apply the label.
7. Otherwise, leave unlabeled.

At MVP scale (a few thousand training examples), brute-force cosine similarity against all embeddings is fast enough with NumPy. No approximate nearest-neighbor index is needed. If the training set grows to tens of thousands of examples, add FAISS for faster lookup.

#### Example

```text
New message: "Rust 1.80 release notes and migration guide"

Top 5 nearest neighbors:
  1. "Rust 1.75 changelog"          → Technology  (sim: 0.91)
  2. "Go 1.22 release announcement" → Technology  (sim: 0.84)
  3. "LLVM weekly newsletter"       → Technology  (sim: 0.79)
  4. "RustConf 2025 CFP"            → Conferences (sim: 0.76)
  5. "Cargo workspace tips"         → Technology  (sim: 0.74)

Scores:
  Technology:  0.91 + 0.84 + 0.79 + 0.74 = 3.28
  Conferences: 0.76

Confidence: 3.28 / (3.28 + 0.76) = 81%
Prediction: Technology (medium confidence → apply + AI-Predicted)
```

---

## Confidence Levels and the Unlabeled Tier

Unlabeled messages are intentionally unlabeled. They represent the emails the user wants to see first. A false positive (wrong label hides an important email) is much worse than a false negative (a newsletter stays in the inbox). Thresholds should be aggressive.

### High Confidence

```text
Confidence >= 95%
```

Action:

- Apply label automatically.

### Medium Confidence

```text
80% <= Confidence < 95%
```

Action:

- Apply predicted label.
- Add auxiliary label: `AI-Predicted`

### Low Confidence

```text
Confidence < 80%
```

Action:

- Leave unlabeled.
- The message stays visible in the inbox.

Thresholds can be tuned based on observed false-positive rates.

---

## Coexistence with Gmail Filters

Some labels are best handled by Gmail's built-in filters rather than content-based classification. For example, labels defined by recipient address (e.g., "move everything sent to alias+xyz@gmail.com to label XYZ") are not content-learnable — the same email content could arrive at different addresses and need different labels.

**Policy:** Leave such Gmail filters active. The classifier only acts on messages that arrive with no user label. Since filters run server-side at delivery time, filter-labeled messages are already labeled before the classifier ever sees them.

**Why this works:**

- Gmail applies filters synchronously during message delivery.
- By the time the classifier queries a message (whether via polling or push notification), filters have already run.
- The classifier's rule is simple: "if a message already has a user label, skip it."
- Filter-based labels still contribute to the training set (their content is embedded and used as neighbors), but the classifier never *predicts* into those labels for new messages — because messages destined for those labels arrive pre-labeled.

**Excluding filter-based labels from predictions:**

Filter-based labels must be explicitly excluded from the classifier's predictions via configuration (e.g., `--exclude-labels XLC XLE XLCap`). Auto-detection via per-label precision is unreliable: a filter-based label with significantly more examples than its siblings will appear to have high precision (it wins by neighbor count, not content distinctiveness). The user knows which labels are filter-based and configures the exclusion list once.

**Validation with push notifications:**

When using the Gmail Watch API (push), there is a theoretical race condition: could a push notification arrive before the filter has applied its label? In practice, no — Gmail processes filters as part of message delivery, before updating the mailbox history that triggers the push. However, to be safe:

1. When the classifier receives a push notification for a new message, it fetches the message's *current* label state from the API.
2. If the message already carries any user-defined label, it is skipped (assumed handled by a filter or manually).
3. Only truly unlabeled messages proceed to classification.

This "check before classifying" step is both the correct behavior and a safety net against any edge-case timing issues.

---

## Learning from Label State

Gmail is the source of truth. The current label state of any message IS the ground truth.

- A label present on a message = that message is a training example for that category.
- A label removed from a message = that message is no longer a training example for that category.
- A label changed from one category to another = the message moves between training sets.
- A message with no category labels = not a training example (and intentionally unlabeled).

It does not matter whether the user or the system originally applied the label. The current state is what counts. No provenance tracking is needed.

---

## Gmail History API

Use the Gmail History API to track changes.

Benefits:

- Detect new messages.
- Detect label additions.
- Detect label removals.

Workflow:

```text
Last History ID
      |
      v
History API
      |
      v
Mailbox Changes
```

Store the latest processed history ID.

### History ID Staleness

If the stored history ID is too old (Gmail returns 404), fall back to a partial re-sync: scan messages from the last 30 days, reconcile label state, and resume from the new history ID. A full rebuild is only needed if the local cache is entirely lost.

---

## Notification Strategy

### Current (Phase 2): Polling loop

The classifier loops every 5 minutes, fetching inbox messages and classifying new ones. Simple but makes unnecessary API calls when idle.

### Target (Phase 3): Push via Pub/Sub

Gmail Watch API notifies a Cloud Pub/Sub topic on mailbox changes. The classifier pulls from the subscription (blocking, instant delivery). Zero API calls when idle, reacts within seconds. See Phase 3 implementation plan for details.

---

## OAuth2 Setup

Gmail API uses OAuth2. For a headless container, the setup is:

1. Create a Google Cloud project with Gmail API enabled.
2. Configure an OAuth consent screen (internal or external with test users).
3. Create OAuth2 client credentials (desktop application type).
4. Run a one-time interactive authorization flow to obtain a refresh token.
5. Store the refresh token in a mounted secret/volume accessible to the container.

At runtime, the service uses the refresh token to obtain short-lived access tokens automatically. The refresh token is the only credential that needs to persist outside the container.

---

## Rate Limits

Gmail API has quotas (250 quota units/second per user for most operations).

During initial training (fetching historical messages):

- Use `messages.list` with `labelIds` filter to find relevant messages.
- Batch `messages.get` requests (up to 100 per batch).
- Implement exponential backoff on 429 responses.
- Pace requests to stay within quota.

During normal polling, quota usage is minimal (a few API calls per cycle).

---

## Deployment

### Recommended: Docker Container

```text
Docker
  |
  v
Python + sentence-transformers
  |
  v
Gmail API
```

Run anywhere Docker runs: a home server, a $5 VPS, a NAS, or a spare machine. The service uses minimal resources (polling every 5 minutes, embedding computation only for new messages).

Advantages:

- Easy deployment
- Easy upgrades
- Reproducible environment
- No cloud vendor lock-in

### Scaling Up (if needed later)

- ECS Fargate: no VM management, automatic restart
- AWS Lambda: pay-per-use, but less convenient for the persistent polling model

These are not needed for a personal email classifier.

---

## Local Cache

### SQLite

Store:

- Gmail message ID
- Label
- Embedding vector
- Timestamp
- Last processed history ID

### Purpose

- Fast classification (avoid re-fetching and re-embedding known messages)
- Fast startup (load embeddings from disk rather than recomputing)

The cache is not authoritative. Gmail is.

---

## Recovery Philosophy

No backups are required.

If local state is lost:

1. Read labeled messages from Gmail.
2. Recompute embeddings.
3. Rebuild similarity index.
4. Continue processing.

Because Gmail is the source of truth, the system is always recoverable.

---

## Rebuild Scenarios

### Normal Operation

No rebuild required.

Only incremental updates.

### Service Restart

Load SQLite cache.

Continue from latest history ID.

### History ID Expired

Partial re-sync from last 30 days.

Resume normal polling.

### Catastrophic Cache Loss

Perform full rebuild from Gmail.

### Embedding Model Upgrade

Recompute all embeddings.

Rebuild similarity index.

This is expected to be rare.

---

## Implementation Phases

The primary risk is ML effectiveness, not infrastructure. Validate classification quality first, on a laptop, before building a service.

---

### Phase 1: Validate the ML Approach (laptop, no side effects) [DONE]

Goal: answer the question "does this actually work on my email?" before writing any service code.

**Result: YES.** 99.9% precision at 91.7% coverage on content-based labels, with skip examples preventing false positives on inbox messages.

#### Step 1.1: Project Skeleton and Unit Tests [DONE]

Set up the Python project with unit tests using synthetic data (95 tests, all passing).

#### Step 1.2: OAuth2 Setup and Gmail Fetch [DONE]

- `scripts/fetch_training_data.py`: fetches labeled messages (500/label, stores in `data/training.db`)
- `scripts/fetch_inbox.py`: fetches 500 recent inbox messages (stores in `data/inbox_sample.db`)
- Incremental: re-running fetches only new messages.

#### Step 1.3: Train + Cross-Validation [DONE]

- Embeddings module (sentence-transformers, lazy-loaded)
- Training pipeline: messages → preprocess → embed → (embeddings, labels)
- Leave-one-out cross-validation with optional `__skip__` examples
- Evaluation metrics: precision/coverage at threshold, per-label precision
- CLI: `scripts/train_and_evaluate.py`

#### Step 1.4: Dry-Run + Skip Discovery [DONE]

- CLI: `scripts/dry_run.py` — classifies inbox messages without modifying Gmail

**Key discovery: the false positive problem.**

Without negative examples, the classifier labels 54% of inbox messages (all incorrectly) because KNN confidence only measures neighbor agreement, not absolute fit. Bank alerts, booking confirmations, etc. have no training examples but land near "Pub" in embedding space — and all 5 neighbors agree, giving 100% confidence.

**Solution: `__skip__` pseudo-label.**

Unlabeled inbox messages are used as negative training examples with a `__skip__` label. When `__skip__` wins the KNN vote, the message is left unlabeled. When it's among neighbors but doesn't win, its score dilutes confidence in the real label.

**Results with skip examples (6 content-based labels, excluding filter-based XLC/XLE/XLCap):**

```
Threshold  Precision  Coverage
0.95       99.9%      91.7%
0.80       99.8%      95.9%
0.60       99.6%      98.8%
```

- Inbox false positives: 3.8% (19/500) — down from 54% without skip
- Per-label precision: 99.1%–100% for all 6 content-based labels
- Only 5 errors in LOO (4 are RO↔Gurobi which are legitimately related)

#### Phase 1 Lessons Learned

1. **Confidence alone is insufficient.** KNN confidence = neighbor agreement, not absolute similarity. A message far from all training data still gets 100% confidence if all distant neighbors share a label.

2. **Negative examples are essential.** The unlabeled inbox provides the "don't label" signal. Without it, the classifier has no concept of "none of the above."

3. **Filter-based labels are not content-learnable.** Labels defined by recipient address (not by content) confuse the classifier. These should remain handled by Gmail filters. Per-label precision in evaluation naturally surfaces them.

4. **Labeling must be exhaustive.** When creating a new label, label ALL matching messages in the inbox — leaving some unlabeled sends contradictory signal (same content in both label and skip pools).

5. **The inbox IS the negative training set** (when Gmail filters handle everything that should be labeled). No bootstrapping problem in this case.

---

### Phase 2: Apply Labels (laptop, writes to Gmail)

Goal: let the system actually modify Gmail, with guardrails.

#### Steps:

1. Add a `classify_and_label.py` script.
2. Build training index: labeled messages + inbox as `__skip__` examples.
3. Fetch recent unlabeled inbox messages via API.
4. For each message: if it already has a user label, skip (filter handled it).
5. Classify against the training index.
6. For messages above threshold: apply the label via `messages.modify`.
7. For medium-confidence messages: also apply `AI-Predicted` label.
8. Log every action taken.

#### Training data refresh:

Before each run:
- Re-fetch training data (`make fetch-training`) — picks up newly labeled messages.
- Re-fetch inbox (`make fetch-inbox`) — refreshes the `__skip__` pool, removing messages that were labeled since last fetch.

#### Exclude non-content labels:

Labels handled by Gmail filters (XLC, XLE, XLCap in current setup) must be excluded from predictions via `--exclude-labels`. This is explicit configuration — the user knows which labels are filter-based.

#### Guardrails:

- Start with the 0.95 threshold (99.9% precision validated).
- Run manually (not on a schedule) for the first few days.
- Review the log after each run.
- Add a `--dry-run` flag that shows what would happen without modifying anything.

#### Feedback loop:

- After a few manual runs, check for corrections (labels you changed in Gmail).
- Re-fetch training data to incorporate corrections.
- The system self-improves: more labeled messages = better training, more inbox history = better skip signal.

---

### Phase 3: Push Notifications (laptop, near-real-time) [DONE]

Goal: replace polling with push notifications via Gmail Watch API + Cloud Pub/Sub. React instantly to new mail and manual label changes.

**Result:** Working. Reacts to new mail within seconds. Label changes auto-update training/skip DBs.

#### Background: Google Cloud Pub/Sub

**What is Pub/Sub?** A message queue service from Google Cloud. Publishers send messages to a *topic*; subscribers read from a *subscription* attached to that topic. Decouples producers from consumers.

**What is a topic?** A named channel. In our case, Gmail publishes "something changed in your mailbox" messages to the topic. You never write to it directly — Gmail does.

**What is a subscription?** A consumer endpoint attached to a topic. Messages accumulate until acknowledged. A *pull* subscription means your code fetches messages on demand (no public URL needed). A *push* subscription would require a public HTTPS endpoint.

**Permissions model:**

- The topic needs exactly one special permission: `gmail-api-push@system.gserviceaccount.com` as **Publisher**. This is a Google-owned service account that Gmail uses to push notifications. It cannot read your messages — it only writes small notification payloads (`{"emailAddress": "...", "historyId": "..."}`).
- Your OAuth user credentials (with `pubsub` scope) have **Subscriber** access by default as the project owner. No additional IAM configuration needed for pull.
- Do NOT grant broader permissions (Editor, Owner) to the Gmail push service account. Publisher is sufficient and minimal.

#### Architecture: Pub/Sub pull subscription

```text
Gmail
  |  (mailbox change)
  v
users.watch() → Cloud Pub/Sub topic
                      |
                      v
              Pull subscription
                      |
                      v
              Classifier process (local)
                      |
                      v
              history.list() → classify / update training
```

The classifier process pulls from the subscription (blocking, instant delivery). No public URL needed. The model stays loaded in memory. Reacts within seconds, makes zero API calls when idle.

#### GCP setup (completed)

Project: `classy-498012`

1. Enabled Pub/Sub API in the Google Cloud project.
2. Created topic: `projects/classy-498012/topics/gmail-notifications`.
3. Granted publish rights to `gmail-api-push@system.gserviceaccount.com` (Pub/Sub Publisher role, topic-level only).
4. Created pull subscription: `projects/classy-498012/subscriptions/gmail-notifications-sub` (never-expire).
5. Added `https://www.googleapis.com/auth/pubsub` to OAuth scopes.
6. Installed `google-cloud-pubsub` Python package.

#### How it works

1. **On startup**: loads training data, embeddings, authenticates. Calls `users.watch()` to register for notifications. Does an initial inbox check to catch anything missed while offline.
2. **Main loop**: blocks on Pub/Sub pull (60s timeout). On notification, calls `history.list()` to get what changed.
3. **New inbox message**: classifies it (same KNN logic). Labels+archives if confident, adds to skip pool if not.
4. **Label change**: updates training DB (label added → training, label removed → skip pool).
5. **Watch renewal**: checked before each pull, renewed if expiring within 1 hour.
6. **History expired**: falls back to full inbox scan, re-watches to get fresh historyId.

#### Reacting to label changes

When the user manually labels or unlabels a message:

- **Label added**: fetch the message, add to training DB under that label. Remove from skip pool (if present).
- **Label removed**: remove from training DB for that label. Add to skip pool (message is now an unlabeled inbox message that shouldn't be labeled).
- **Label moved** (remove A + add B): remove from training under A, add under B. Not a skip example (it still has a label).

This eliminates the need for manual `make fetch-training` / `make fetch-inbox` during normal operation.

#### Known limitations

- Pub/Sub pull still needs a running process on the laptop.

#### Usage

```bash
make watch-pubsub   # run with push notifications (preferred)
make watch          # fallback: poll every 5 minutes
make classify       # one-shot: classify once and exit
```

---

### Phase 4: Deploy as a Service (always-on, no laptop needed)

Goal: run the classifier permanently without keeping a laptop open.

#### Monitoring

Beyond "is the process alive":

- **Correction tracking**: when `process_label_changes` sees the user relabel something the classifier labeled, that's a misclassification. Count these. Alert if the rate spikes.
- **Activity heartbeat**: "last successful classification at T" / "N messages processed in last 24h". Alert if too old.
- **Watch renewal**: log renewals; alert if time-since-last-renewal exceeds 24h.
- **Confidence drift**: track confidence distribution over time. A shift toward low-confidence signals training data staleness.

#### Logging

Replace `print()` with Python `logging` (JSON lines to stdout). The container runtime captures stdout. Structured logs enable querying after crashes.

#### Hosting options evaluated

**AWS (already a user)**

| Option | Specs | $/month |
|--------|-------|---------|
| Lightsail | 1GB RAM, 2 vCPU, flat pricing | $5 |
| EC2 t4g.micro | 1GB RAM, 2 vCPU ARM (tight) | ~$6 |
| EC2 t4g.small | 2GB RAM, 2 vCPU ARM (comfortable) | ~$12 |

Lightsail is simplest: flat bill, built-in monitoring, SSH access, no VPC/SG fiddling.

**GCP (already have project classy-498012 with Pub/Sub)**

| Option | Specs | $/month |
|--------|-------|---------|
| e2-micro | 1GB RAM, 2 shared vCPU | **Free** (perpetual, 1/account, us-central1/us-west1/us-east1) |
| e2-small | 2GB RAM, 2 shared vCPU | ~$13 |
| Cloud Run (push) | Scale to zero, pay per request | ~free for low volume |

The e2-micro free tier is perpetual (not 12-month trial). 30GB disk also free. 1GB RAM is tight but might fit. Pub/Sub is in the same project (no egress cost).

Cloud Run alternative: flip to a push subscription (Pub/Sub POSTs to Cloud Run). Service scales to zero when idle. Essentially free at personal email volume. Catch: cold start reloads the model (~5-10s), needs min-instances=1 to avoid that (~$15/month).

**Oracle Cloud (free tier)**

4 ARM cores, 24GB RAM, perpetually free. Catches: capacity hard to provision (retry for days), idle instances get reclaimed (bad for a mostly-sleeping classifier), no SLA, account closure risk.

**Raspberry Pi 4/5**

~50 EUR one-time, ~5 EUR/year electricity. No vendor dependency. 4GB+ RAM runs the model fine. Needs physical setup + network config.

**Synology NAS (DS213j)**

Not viable. ARM (Marvell Armada 370), 512MB RAM, no Docker. Too underpowered for the embedding model.

#### Recommendation

Try **GCP e2-micro** first (free, same project as Pub/Sub). If 1GB RAM is too tight, upgrade to e2-small ($13/month) or switch to AWS Lightsail ($5/month). Both run Docker with restart policies and capture stdout logs.

#### TODO when ready to deploy

1. Switch prints to `logging` module (JSON lines).
2. Optimize Dockerfile for low memory (slim base image, no dev deps).
3. Add health check endpoint or script.
4. Add correction-rate counter (label_change_handler already has the data).
5. Set up alerting on: process down, no activity in 24h, correction rate spike.
6. Persist historyId to disk so restarts don't trigger full inbox scan.

---

### Phase 5: Dynamic Label Discovery (no restart on new labels)

Goal: when the user creates a new Gmail label and moves messages to it, the running classifier picks this up automatically and starts classifying into the new label — no restart needed.

#### Current state (broken without restart)

Three data structures are built once at startup and never refreshed:

1. `label_name_to_id` — maps label names to Gmail IDs. Used when applying a predicted label.
2. `label_id_to_name` — maps Gmail IDs to names. Used in `process_label_changes` to look up label names from history events.
3. `user_label_ids` — set of all user label IDs. Used to check "does this message already have a user label?"

When a new label is created:
- `labelsAdded` events arrive with an unknown label ID.
- `label_id_to_name.get(lid)` returns `None` → event silently skipped.
- Training index never learns the new label.
- Even if it did, `label_name_to_id` wouldn't have it, so predictions couldn't be applied.

#### Design

**Trigger:** detect an unknown label ID in a history event.

When `process_label_changes` encounters a label ID not in `label_id_to_name`:
1. Refresh the label maps from Gmail (`client.list_user_labels()`).
2. If the unknown ID is now in the refreshed map (confirming a new label exists):
   - Update all three data structures in place.
   - Log: "New label discovered: {name}".
   - Check if the new label is in the excluded set. If so, skip it.
   - Otherwise, proceed with normal `labelsAdded` processing (fetch message, embed, add to training index).
3. If the unknown ID is still not found after refresh (deleted between event and refresh?), skip it.

This is a lazy refresh — we only hit the API when we see something unexpected. During normal operation (no new labels), zero extra API calls.

#### Implementation plan

**Step 1: Extract label maps into a mutable container**

Currently the three dicts are passed as separate arguments throughout the call chain. Refactor into a single `LabelRegistry` object:

```python
class LabelRegistry:
    def __init__(self, client: GmailClient, excluded: Set[str]):
        self._client = client
        self._excluded = excluded
        self.refresh()

    def refresh(self):
        """Re-fetch label list from Gmail."""
        user_labels = self._client.list_user_labels()
        self.name_to_id = {name: lid for lid, name in user_labels}
        self.id_to_name = {lid: name for lid, name in user_labels}
        self.user_label_ids = {lid for lid, name in user_labels}

    def is_known(self, label_id: str) -> bool:
        return label_id in self.id_to_name

    def is_excluded(self, label_id: str) -> bool:
        name = self.id_to_name.get(label_id)
        return name in self._excluded if name else False

    def get_name(self, label_id: str) -> Optional[str]:
        return self.id_to_name.get(label_id)

    def get_id(self, label_name: str) -> Optional[str]:
        return self.name_to_id.get(label_name)
```

Passed by reference, so all callers see updates immediately.

**Step 2: Add unknown-label detection to `process_label_changes`**

Before the current filtering loop, check if any label ID in the event is unknown:

```python
# Check for unknown label IDs → trigger refresh
unknown_ids = set()
for event in events:
    if event.type in ("labelsAdded", "labelsRemoved"):
        for lid in event.label_ids:
            if lid not in registry.user_label_ids and not lid.startswith(("CATEGORY_", "IMPORTANT", "INBOX", "SENT", "DRAFT", "SPAM", "TRASH", "UNREAD", "STARRED")):
                unknown_ids.add(lid)

if unknown_ids:
    registry.refresh()
    new_labels = [registry.get_name(lid) for lid in unknown_ids if registry.is_known(lid)]
    for name in new_labels:
        print(f"{now()} New label discovered: {name}")
```

**Step 3: Update callers**

Replace all `label_name_to_id`, `label_id_to_name`, `user_label_ids` parameters with a single `registry: LabelRegistry` parameter throughout:
- `_run_pubsub_mode`
- `_run_poll_mode`
- `_check_inbox`
- `process_label_changes`
- `process_history_events`

**Step 4: Handle excluded labels**

When a new label is discovered, it's NOT excluded by default (the excluded set is explicit configuration). If the user creates a filter-based label and wants it excluded, they restart with the updated `--exclude-labels`. This is acceptable: new content-based labels (the common case) work immediately; new filter-based labels (rare) need a restart.

**Step 5: Tests**

- Test that `LabelRegistry.refresh()` picks up new labels.
- Test that `process_label_changes` with an unknown label ID triggers refresh and processes the event.
- Test that a still-unknown ID after refresh is silently skipped.
- Test that excluded labels discovered via refresh are skipped.
- Integration: new label → move messages → verify training index updated → verify classification uses new label.

---

## Technology Stack

- Python 3.11+
- uv (package/project management)
- sentence-transformers (all-MiniLM-L6-v2)
- NumPy
- BeautifulSoup4 (HTML parsing)
- google-api-python-client + google-auth
- google-cloud-pubsub (push notifications)
- SQLite (local cache)
- pytest (testing)

---

## Project Structure

```
gmail-classifier/
  src/gmail_classifier/
    __init__.py
    preprocessing.py    # HTML strip, quote removal, signature trim, text repr
    embeddings.py       # sentence-transformers wrapper
    classifier.py       # KNN logic, confidence calculation, decision
    gmail_client.py     # API wrapper: fetch, label, history
    storage.py          # SQLite read/write for messages + embeddings
  tests/
    __init__.py
    fixtures/emails.json
    test_preprocessing.py
    test_text_representation.py
    test_classifier.py
  credentials/          # .gitignored
  data/                 # .gitignored
  pyproject.toml
  Makefile
  .gitignore
```

## Development Setup

```
git clone <repo>
make setup   # creates .venv, installs project + dev dependencies
make test    # runs pytest
make clean   # removes .venv and build artifacts
```
