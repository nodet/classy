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
  +--> Embedding Model
  |
  +--> Similarity Index (FAISS)
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
   - Extract relevant body text
4. Compute embeddings.
5. Build a nearest-neighbor index (FAISS).

The labeled historical messages become the training set.

---

## Classification Strategy

### Input Features

For each message:

- Sender address
- Sender name
- Subject
- Mailing-list headers (if present)
- First portion of the body

### Embedding-Based Classification

For a new message:

1. Compute its embedding.
2. Find the nearest labeled messages.
3. Weight votes by similarity.
4. Assign the most likely label.
5. Compute a confidence score.

Example:

```text
Technology (94%)
Nearest examples:
- ACM newsletter
- NVIDIA developer update
- Google AI announcement
```

---

## Confidence Levels

### High Confidence

Example:

```text
Confidence >= 90%
```

Action:

- Apply label automatically.

### Medium Confidence

Example:

```text
70% <= Confidence < 90%
```

Action:

- Apply predicted label.
- Add auxiliary label such as:
  - AI-Predicted
  - NeedsReview

### Low Confidence

Example:

```text
Confidence < 70%
```

Action:

- Leave unclassified.
- Wait for manual classification.

Thresholds can be tuned later.

---

## Learning from Corrections

The system should treat manual label changes as training feedback.

Example:

```text
Predicted:
  Technology

User changes to:
  Optimization
```

This is interpreted as a correction.

The corrected message becomes part of the Optimization training set.

---

## Gmail History API

Use the Gmail History API to track changes.

Benefits:

- Detect new messages.
- Detect label additions.
- Detect label removals.
- Detect manual corrections.

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

This is the only truly important piece of persistent state.

---

## Polling Strategy

Simple version:

Every 5 minutes:

1. Query Gmail History API.
2. Retrieve changes since the previous history ID.
3. Process new messages.
4. Process label changes.
5. Update local cache.

This remains polling, but it is efficient because only mailbox changes are retrieved.

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

## Deployment Options

### Recommended First Version

Containerized Python service.

```text
Docker
  |
  v
Python
  |
  v
Gmail API
```

Advantages:

- Easy deployment
- Easy upgrades
- Reproducible environment

### AWS Options

#### EC2

Pros:

- Familiar
- Flexible

Cons:

- VM maintenance
- OS updates

#### ECS Fargate

Recommended cloud deployment.

Pros:

- No VM management
- Runs containers directly
- Automatic restart
- Easy upgrades

#### AWS Lambda

Possible later.

Pros:

- Very low operational burden
- Pay-per-use

Cons:

- Less convenient for experimentation
- Additional AWS concepts

---

## Local Cache

### SQLite

Store:

- Gmail message ID
- Label
- Embedding vector
- Timestamp
- Last processed history ID

### FAISS

Store:

- Embedding index for nearest-neighbor search

Purpose:

- Fast classification
- Fast startup

The cache is not authoritative.

---

## Recovery Philosophy

No backups should be required.

If local state is lost:

1. Read labeled messages from Gmail.
2. Recompute embeddings.
3. Rebuild FAISS.
4. Continue processing.

Because Gmail is the source of truth, the system remains recoverable.

---

## Rebuild Scenarios

### Normal Operation

No rebuild required.

Only incremental updates.

### Service Restart

Load SQLite and FAISS.

Continue from latest history ID.

### Catastrophic Cache Loss

Perform full rebuild from Gmail.

### Embedding Model Upgrade

Recompute all embeddings.

Rebuild FAISS.

This is expected to be rare.

---

## Suggested MVP

Technology Stack:

- Python
- Gmail API
- Gmail History API
- SQLite
- FAISS
- Docker

Workflow:

1. Fetch historical labeled messages.
2. Build embedding index.
3. Poll History API every 5 minutes.
4. Classify new messages.
5. Apply Gmail labels.
6. Learn from manual corrections.

This provides semantic email filing with minimal operational complexity and a recoverable, nearly stateless architecture.
