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

**Auto-detection of non-content-learnable labels:**

During evaluation (leave-one-out cross-validation), per-label precision naturally reveals which labels are not content-learnable. Labels whose precision falls significantly below others (e.g., <95% when others are >99%) are candidates for filter-based handling rather than ML classification. The evaluation script reports per-label precision to surface this.

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

## Polling Strategy

Every 5 minutes:

1. Query Gmail History API.
2. Retrieve changes since the previous history ID.
3. Process new messages (classify if unlabeled).
4. Process label changes (update training set).
5. Update local cache.

This is efficient because only mailbox deltas are retrieved.

---

## Future Optimization: Push Notifications

Possible future enhancement:

```text
Gmail Watch API
        |
        v
Google Pub/Sub
        |
        v
Classifier Service
        |
        v
History API
```

This removes periodic polling and provides near-real-time updates.

Not required for the first version.

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

### Phase 1: Validate the ML Approach (laptop, no side effects)

Goal: answer the question "does this actually work on my email?" before writing any service code.

#### Step 1.1: Project Skeleton and Unit Tests [DONE]

Set up the Python project with unit tests using synthetic data.

**Implemented (39 tests, all passing):**

- Preprocessing pipeline: strip_html, remove_quoted_replies, remove_forwarded, trim_signature, truncate, preprocess_email_body
- Text representation: build_text_representation (sender, subject, body, optional list-id)
- Classifier: cosine_similarity, find_neighbors (KNN), aggregate_scores, compute_confidence, decide_action, classify
- ClassificationResult dataclass, Action enum, MIN_EXAMPLES_PER_LABEL threshold
- All tests use plain strings or synthetic NumPy vectors — no embedding model needed, suite runs in <200ms

**Not tested with fake data (by design):**

- ML quality. Synthetic text embeds differently than real email. Thresholds will be tuned on real data in Step 1.3.

#### Step 1.2: OAuth2 Setup and Gmail Fetch

One-time interactive setup, then fetch and cache real data locally.

**Steps:**

1. Create Google Cloud project, enable Gmail API, create OAuth credentials.
2. Write a CLI script: `fetch_training_data.py`
   - Authenticate (opens browser, stores refresh token in `credentials/token.json`)
   - For each target label: call `messages.list(labelIds=[label_id])`, then batch `messages.get`
   - Extract: message ID, sender, subject, headers, body (raw or parsed)
   - Store in a local SQLite database or JSON file (not committed to git)
3. Write a second script or flag: `fetch_inbox.py`
   - Fetch the last N messages from INBOX that have no category label
   - Same extraction logic
   - These become the "dry run" test set

**Output:** a local file like `data/training.db` with a few hundred to a few thousand labeled messages, and `data/inbox_sample.db` with N recent unlabeled messages.

**Privacy:** add `data/` and `credentials/` to `.gitignore`. Real email never leaves the laptop.

**Rate limit handling:** batch fetches, exponential backoff, progress output. The initial fetch might take a few minutes depending on volume.

#### Step 1.3: Train (Build the Index)

**Steps:**

1. Load training messages from the local store.
2. Preprocess each (HTML strip, quote removal, signature trim, truncate).
3. Construct the text representation: `"{from_name} <{from_address}> | {subject} | {body_snippet}"` (experiment with format).
4. Compute embeddings using sentence-transformers.
5. Store embeddings + labels in memory (NumPy arrays).

**Validation via leave-one-out cross-validation:**

Before testing on new messages, measure how well the system classifies its own training data:

- For each training message, remove it from the index.
- Classify it against the remaining messages.
- Record: predicted label, true label, confidence.

This gives a confusion matrix and per-threshold accuracy without needing separate test data.

**Key metrics:**

- **Precision at threshold T**: of messages where confidence >= T, what fraction are correctly labeled?
- **Coverage at threshold T**: what fraction of messages have confidence >= T?

The tradeoff: higher threshold = higher precision but lower coverage (more messages left unlabeled). This is acceptable — the system should be conservative.

**Expected output:**

```
Threshold  Precision  Coverage
0.95       98%        45%
0.90       95%        62%
0.80       88%        78%
0.70       82%        87%
```

This table directly informs threshold tuning.

#### Step 1.4: Dry-Run Classification

**Steps:**

1. Load the trained index (all training embeddings + labels).
2. Load the inbox sample (N recent unlabeled messages).
3. For each inbox message:
   - Preprocess and embed.
   - Run KNN classification.
   - Record: predicted label, confidence, top 5 nearest neighbors (with their subjects/senders for interpretability).
4. Output a report sorted by confidence (highest first).

**Report format:**

```
Message: "Rust 1.80 release notes" from releases@rust-lang.org
  Prediction: Technology (confidence: 0.89)
  Neighbors:
    1. "Rust 1.75 changelog" [Technology] (sim: 0.91)
    2. "Go 1.22 release" [Technology] (sim: 0.84)
    3. "LLVM weekly" [Technology] (sim: 0.79)
    4. "RustConf CFP" [Conferences] (sim: 0.76)
    5. "Cargo tips" [Technology] (sim: 0.74)
  Verdict: WOULD LABEL (medium confidence)

Message: "Re: dinner tonight?" from spouse@gmail.com
  Prediction: Personal (confidence: 0.42)
  Neighbors:
    1. "Weekend plans" [Personal] (sim: 0.55)
    2. "Team lunch Friday" [Work] (sim: 0.51)
    ...
  Verdict: WOULD NOT LABEL (low confidence)
```

**Manual review:** read the report, count how many predictions are correct, wrong, or debatable. This is the real test.

#### Phase 1 Success Criteria

- Precision >= 95% at the chosen threshold (i.e., when the system would label, it's almost always right)
- Coverage is reasonable (the system labels at least 40-50% of fileable messages, not just the trivially obvious ones)
- The failure modes make sense (ambiguous messages stay unlabeled, not mislabeled)

If these aren't met, iterate: try a different embedding model, adjust preprocessing, change K, experiment with the text representation format.

---

### Phase 2: Apply Labels (laptop, writes to Gmail)

Goal: let the system actually modify Gmail, with guardrails.

#### Steps:

1. Add a `classify_and_label.py` script (or mode flag).
2. Fetch recent unlabeled inbox messages via API.
3. Classify each.
4. For messages above threshold: apply the label via `messages.modify`.
5. For medium-confidence messages: also apply `AI-Predicted` label.
6. Log every action taken.

#### Guardrails:

- Start with a very high threshold (95%+).
- Run manually (not on a schedule) for the first few days.
- Review the log after each run. If mislabeling occurs, lower the threshold or investigate why.
- Add a `--dry-run` flag that shows what would happen without modifying anything (reuse Phase 1 logic).

#### Feedback loop:

- After a few manual runs, check for corrections (labels you changed).
- Feed corrections back into the training set (re-fetch labeled messages, rebuild index).
- Track precision over time: is it improving as the training set grows?

---

### Phase 3: Autonomous Service (remote, runs unattended)

Goal: move from manual laptop runs to a self-running service.

Only start this phase after Phase 2 has been running successfully for a few weeks with stable precision.

#### Steps:

1. Package into a Docker container.
2. Add the polling loop (History API every 5 minutes).
3. Add incremental index updates (label changes update the training set without full rebuild).
4. Add the history ID persistence and staleness handling.
5. Deploy to any always-on machine (home server, VPS, or cloud).

#### What changes from Phase 2:

- Polling replaces manual invocation.
- History API replaces "fetch last N inbox messages".
- Incremental index updates replace full rebuilds.
- Needs to handle restarts, token refresh, and error recovery gracefully.

---

## Technology Stack

- Python 3.11+
- uv (package/project management)
- sentence-transformers (all-MiniLM-L6-v2)
- NumPy
- BeautifulSoup4 (HTML parsing)
- google-api-python-client + google-auth
- SQLite (local cache)
- pytest (testing)
- Docker (Phase 3 only)

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
