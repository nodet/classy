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

For a new message:

1. Compute its embedding.
2. Compute cosine similarity against all stored labeled embeddings.
3. Weight votes by similarity.
4. Identify the most likely label.
5. Compute a confidence score.

At MVP scale (a few thousand training examples), brute-force cosine similarity over all embeddings is fast enough with NumPy. No approximate nearest-neighbor index is needed.

If the training set grows to tens of thousands of examples, add FAISS for faster lookup.

Example:

```text
Technology (94%)
Nearest examples:
- ACM newsletter
- NVIDIA developer update
- Google AI announcement
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

## Suggested MVP

Technology Stack:

- Python
- sentence-transformers (all-MiniLM-L6-v2)
- Gmail API + History API
- SQLite
- NumPy (cosine similarity)
- Docker

Workflow:

1. One-time OAuth2 setup to obtain refresh token.
2. Fetch historical labeled messages (with rate limit handling).
3. Preprocess and embed messages.
4. Build similarity index in memory.
5. Poll History API every 5 minutes.
6. Classify new unlabeled messages (high threshold to avoid false positives).
7. Apply Gmail labels for high-confidence predictions.
8. Update training set when label state changes.
9. Leave uncertain messages unlabeled (they surface in the inbox).

This provides semantic email filing with minimal operational complexity, a recoverable nearly-stateless architecture, and a strong bias toward precision over recall.
