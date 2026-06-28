# Plan: Make `nodet/classy` Run Reliably on GCP with Lower-Memory Email Classification

Date: 2026-06-28
Repository reviewed: https://github.com/nodet/classy
Primary objective: keep automatic Gmail labeling useful while reducing runtime memory enough for a GCP deployment.

---

## 0. Measurement update — 2026-06-28 (reframes this plan)

Phase 0 instrumentation (`log_mem()` RSS checkpoints, commit `cf4ee18`) was
deployed to the e2-micro and measured. The numbers **partly overturn this
document's premise** and are recorded here as the source of truth.

### What was measured (VM, 1 GB e2-micro, pubsub mode)

| Startup stage | RSS | Note |
|---|---|---|
| before training DB load | 72 MB | interpreter + imports |
| after training DB load | 519 MB | **+447 MB** — raw `Message` objects (full bodies) |
| after skip DB load | 513 MB | — |
| after `Embedder()` load | 631 MB | **+118 MB** — FastEmbed/onnxruntime is *not* the big consumer |
| `prepare_texts` | — | ran **327 s** (BeautifulSoup over all 4331 msgs) |
| after `build_training_data` | 636 MB | peak 649 MB |
| **after `del` + `malloc_trim`** | **214 MB** | `malloc_trim(0)` works on Linux (no-op on macOS) — returns ~420 MB |
| steady-state: pubsub loop ready | **220 MB** | the resting footprint |

### Conclusions that change the plan

1. **Steady-state is ~220 MB, not ~600 MB.** The 519/636 MB figures are a
   *startup transient*; `malloc_trim` returns most of it. The README's
   "~600 MB resident pins the VM" framing measured the transient, not the
   resting state. At 220 MB on a 1 GB box, the embedding runtime is **not** a
   steady-state memory problem.
2. **The OOM was a startup transient, not a leak.** Crash cause: peak ~649 MB
   resident + an `embed_batch` spike of 455 *uncached* messages, colliding
   under 1 GB. The misses existed because `fetch-training`/`fetch-inbox`
   refreshed the DBs without rebuilding `embeddings.db`. Fix: `make embed`
   before deploy → misses dropped 460 → 5 → startup cleared.
3. **FastEmbed is only +118 MB.** This plan's central lever ("remove FastEmbed
   to reclaim ~500 MB") does not hold. The largest single chunk is the
   **+447 MB raw-corpus load**, which is *transient* and trimmed.
4. **The real operational wound is startup time, not memory.** `prepare_texts`
   ran **327 s every restart** — BeautifulSoup over all 4331 messages — before
   the service could serve, even on a 99.9%-cached corpus. With
   `Restart=on-failure`, a crash-loop burns minutes per cycle.

### Cheap fixes applied (no new classifier needed for these)

- **Exclude XL* permanently** (`cf4ee18`): XLC/XLE/XLCap were **1560 / 5134
  rows = 30%** of the training corpus, loaded into RAM then discarded at
  filter time. Now excluded at fetch (`fetch_training_data.py --exclude-labels`,
  wired into `make fetch-training`) and deleted from the existing DB. ~30% off
  the transient corpus load.
- **Defer text prep to cache misses** (`1616593`): `build_training_data` now
  extracts cheap labels/ids for all messages but only runs `_message_text`
  (BeautifulSoup) for cache *misses*. Warm-cache restart preps ~5 messages
  instead of 4331 → startup drops from ~5.5 min to seconds, and the transient
  peak shrinks. Guarded by a test asserting cache hits are never text-prepped.
- **Live RSS in logs** (`80845fa`): `now()` prefixes every per-message line
  with current RSS so memory creep is visible in the service log.

### Revised recommendation on the lightweight classifier

The lightweight classifier (Phases 1–3 of the approved plan) is **still
defensible, but for a different reason than this document argues**: it would
eliminate the startup transient and the BeautifulSoup parse entirely (no
onnxruntime, no corpus load, no text prep at boot), making the VM robust to
corpus growth. It is **no longer justified by steady-state memory** — that is
already ~220 MB. Decision: re-measure after the startup fixes above; build the
lightweight classifier only if the transient peak or restart cost remains a
problem as the corpus grows. The §-by-§ design below stays valid if/when built,
subject to the corrections in the approved plan's review verdict (preserve
online adaptation; reuse existing `classifier.py`/`evaluation.py`; drop hybrid
mode, per-label JSON, Cloud Run, SGD from committed scope).

---

## 1. Executive summary

The current version of `nodet/classy` uses semantic KNN over email embeddings. That is a good quality baseline, but it is a poor fit for a very small always-on GCP VM because the runtime loads:

1. all training messages;
2. all skip/inbox-negative examples;
3. a local FastEmbed embedding model;
4. processed text representations;
5. cached or newly computed embedding arrays;
6. a final in-memory KNN index.

The repository has already moved in the right direction by using FastEmbed, embedding cache support, `malloc_trim`, and a GCP-specific deployment flow. The remaining memory issue is structural: the GCP process still behaves like a local semantic ML runtime.

Recommended direction:

- Keep the current semantic KNN classifier as the local/high-quality baseline.
- Add a new `lightweight` classifier mode for GCP.
- Train that mode locally from the same training and skip databases.
- Deploy only a small model artifact to GCP.
- Accept lower coverage before accepting lower precision.

The first lightweight model should be a hashed lexical text classifier:

```text
HashingVectorizer + ComplementNB
```

or, if probability calibration is weak:

```text
HashingVectorizer + SGDClassifier(loss="log_loss")
```

This avoids loading FastEmbed on GCP, avoids an embedding matrix at runtime, and avoids full training database loading when the service starts. Quality will likely degrade mainly as lower recall/coverage, which is acceptable if thresholds are conservative.

---

## 2. Current repository state

### 2.1 Project shape

The repository currently describes itself as:

> Semantic auto-labeling for Gmail using KNN on email embeddings.

Source: https://github.com/nodet/classy

Important current repository files:

```text
README.md
gcp-deploy-plan.md
gmail-semantic-labeling-plan.md
scripts/classify_and_label.py
src/gmail_classifier/classifier.py
src/gmail_classifier/embeddings.py
src/gmail_classifier/storage.py
src/gmail_classifier/training.py
src/gmail_classifier/training_index.py
pyproject.toml
Makefile
```

The README currently documents a deployment path to a free-tier GCP `e2-micro` VM using targets such as:

```text
make gcp-create
make gcp-deploy
make gcp-start
make gcp-status
make gcp-logs
make gcp-ssh
make gcp-destroy
```

The same README states that after retraining on the Mac, `make gcp-deploy` detects database size changes and uploads the new database. That is convenient, but it is not ideal for a constrained GCP runtime. In the new plan, GCP should receive a compact model artifact, not the raw training corpus unless debugging requires it.

### 2.2 Current runtime behavior

The current service starts in `scripts/classify_and_label.py`. It loads the training database, loads the skip database, merges them, creates an embedding cache, creates an `Embedder`, builds training embeddings, then deletes large message lists and calls `malloc_trim` when available.

Relevant source:

- https://raw.githubusercontent.com/nodet/classy/main/scripts/classify_and_label.py

The current storage layer has `MessageStore.load_all()`, which selects full message rows including `body_html` and returns a list of `Message` objects.

Relevant source:

- https://raw.githubusercontent.com/nodet/classy/main/src/gmail_classifier/storage.py

The training path converts all messages into preprocessed text and labels before embedding. This means large Python lists can coexist during startup:

- original `Message` objects;
- processed body strings;
- text representations;
- labels;
- ids;
- cached embedding dict;
- missing embedding batch;
- final `float32` embedding matrix.

Relevant source:

- https://raw.githubusercontent.com/nodet/classy/main/src/gmail_classifier/training.py

The embedder wraps FastEmbed and loads `sentence-transformers/all-MiniLM-L6-v2`:

- https://raw.githubusercontent.com/nodet/classy/main/src/gmail_classifier/embeddings.py

The current classifier computes cosine similarities against the full training embedding matrix and sorts to find nearest neighbors:

- https://raw.githubusercontent.com/nodet/classy/main/src/gmail_classifier/classifier.py

### 2.3 Existing quality baseline

The project plan notes strong results from semantic KNN with skip examples, specifically very high precision with substantial coverage. It also notes that the `__skip__` negative class is essential because KNN confidence alone can be misleading.

Relevant source:

- https://raw.githubusercontent.com/nodet/classy/main/gmail-semantic-labeling-plan.md

This matters because the new lightweight mode should not replace the semantic classifier everywhere. It should become the GCP runtime mode while semantic KNN remains the local evaluation and quality reference.

---

## 3. Problem statement

The classifier uses too much memory on GCP because the current design optimizes for semantic quality, not memory-bounded deployment.

Likely memory pressure points:

1. Full training and skip databases are loaded into Python objects.
2. HTML bodies are loaded even when only compact features are needed.
3. Processed text representations are built for every message before embedding.
4. FastEmbed and ONNX runtime state must stay resident to embed incoming messages.
5. Cached embeddings may be loaded into intermediate dicts before matrix assembly.
6. The final KNN index remains in memory for the lifetime of the service.
7. Each new message classification performs allocation-heavy vector operations.

The current fixes reduce peak memory but do not remove the biggest cost: a local embedding model running inside the production GCP process.

---

## 4. Goals and non-goals

### 4.1 Goals

1. Run reliably on the intended GCP deployment target.
2. Keep precision high, even if fewer messages are automatically labeled.
3. Preserve the current semantic classifier as a benchmark and optional local mode.
4. Avoid loading FastEmbed in the GCP lightweight runtime.
5. Avoid loading full training and skip databases during GCP service startup.
6. Keep the implementation easy to evaluate against the existing corpus.
7. Make rollback simple.

### 4.2 Non-goals

1. Do not chase maximum recall at the expense of false positives.
2. Do not rebuild the entire project around a hosted LLM unless local lightweight models fail.
3. Do not require a vector database for this small personal-email use case.
4. Do not replace Gmail filters for deterministic sender/list/recipient rules.
5. Do not remove semantic KNN until the lightweight model has been measured.

---

## 5. Decision: add classifier modes

Introduce an explicit classifier mode setting:

```text
CLASSY_CLASSIFIER_MODE=semantic
CLASSY_CLASSIFIER_MODE=lightweight
CLASSY_CLASSIFIER_MODE=hybrid
```

Initial behavior:

- `semantic`: current FastEmbed + KNN behavior.
- `lightweight`: new hashed lexical classifier; recommended for GCP.
- `hybrid`: optional later mode; lexical first, semantic fallback only when local resources permit it.

Recommended default by environment:

```text
local development: semantic
local evaluation: semantic and lightweight
GCP e2-micro: lightweight
larger paid VM: semantic or hybrid
```

---

## 6. Architecture proposal

### 6.1 New package layout

Add a classifier abstraction without over-engineering:

```text
src/gmail_classifier/classifiers/
  __init__.py
  base.py
  semantic_knn.py
  lightweight.py
```

Suggested `base.py`:

```python
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

class Action(Enum):
    LABEL = "label"
    LABEL_WITH_REVIEW = "label_with_review"
    NO_LABEL = "no_label"

@dataclass
class ClassificationResult:
    label: str
    confidence: float
    action: Action
    explanation: dict

class EmailClassifier(Protocol):
    def classify_message(self, message) -> ClassificationResult:
        ...
```

Notes:

- The current `Action`, `ClassificationResult`, and threshold logic can be migrated or re-exported for compatibility.
- The existing `classifier.py` can remain as semantic-specific implementation during transition.
- `explanation` replaces semantic-only `neighbors` with a generic payload: top labels, probabilities, margins, matched sender/list features, etc.

### 6.2 Runtime flow in GCP lightweight mode

New GCP startup should do this:

```text
1. Parse CLI/env configuration.
2. Load Gmail credentials.
3. Load label registry.
4. Load lightweight model artifact.
5. Start poll/pubsub loop.
6. For each incoming message:
   a. parse email metadata/body;
   b. build compact text features;
   c. classify using the model artifact;
   d. apply label only if threshold and margin rules pass;
   e. otherwise store as skip/unlabeled if current behavior requires it.
```

It should not do this:

```text
1. load all training messages;
2. load all skip messages;
3. instantiate FastEmbed;
4. rebuild embeddings;
5. build KNN matrix.
```

### 6.3 Lightweight features

Build a deterministic feature representation from cheap email fields:

```text
FROM_ADDRESS=<full normalized address>
FROM_DOMAIN=<domain>
FROM_NAME=<normalized display name>
LIST_ID=<list id>
SUBJECT=<subject text>
BODY=<cleaned/truncated body text>
```

Weight reliable fields by repetition rather than by custom math:

```text
FROM_DOMAIN=example.com FROM_DOMAIN=example.com FROM_DOMAIN=example.com
LIST_ID=github.com LIST_ID=github.com LIST_ID=github.com
FROM_ADDRESS=notifications@example.com FROM_ADDRESS=notifications@example.com
SUBJECT=...
BODY=...
```

Rationale:

- Sender domain and list-id are often more stable than body content.
- Subject and body capture semantic-ish content for labels that are not deterministic sender/list labels.
- Field-prefix tokens let a lexical classifier distinguish `github.com` in a sender field from the same token in body text.

### 6.4 Text preprocessing constraints

Use the existing HTML/body preprocessing, but cap body size hard:

```text
max_body_chars: 4000 initially
max_subject_chars: 300
max_sender_chars: 200
```

This prevents pathological newsletters and HTML emails from dominating memory and CPU.

### 6.5 Model artifact

Start with a `joblib` artifact for speed of implementation:

```text
data/models/lightweight_classifier.joblib
data/models/lightweight_classifier_meta.json
```

The metadata file should include:

```json
{
  "created_at": "2026-06-28T00:00:00Z",
  "classifier_type": "hashing_vectorizer_complement_nb",
  "n_features": 262144,
  "ngram_range": [1, 2],
  "labels": ["..."],
  "skip_label": "__skip__",
  "high_threshold": 0.98,
  "review_threshold": 0.90,
  "min_margin": 0.20,
  "training_db_hash": "...",
  "skip_db_hash": "...",
  "metrics": {
    "precision": null,
    "coverage": null
  }
}
```

Later, if `joblib` dependency or artifact size is undesirable, move to a custom `.npz` artifact for a pure NumPy ComplementNB runtime.

---

## 7. Model options

### 7.1 Option A: HashingVectorizer + ComplementNB

Recommended first implementation.

Properties:

- Very small runtime memory.
- No vocabulary stored.
- Fast classification.
- Good baseline for text classification.
- Naturally produces class scores/probabilities.
- Usually robust for imbalanced text categories compared with standard MultinomialNB.

Suggested vectorizer:

```python
HashingVectorizer(
    n_features=2**18,
    alternate_sign=False,
    norm="l2",
    ngram_range=(1, 2),
    dtype=np.float32,
)
```

Suggested classifier:

```python
ComplementNB(alpha=0.1)
```

External reference:

- HashingVectorizer documentation: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html
- ComplementNB documentation: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html

Pros:

- Lowest implementation complexity.
- Very low memory.
- Works with sparse matrices.
- Easy to retrain locally.

Cons:

- Worse semantic generalization than embeddings.
- Feature collisions are possible.
- Probability estimates may require conservative thresholds.
- Explanations are less intuitive because hashing discards feature names.

### 7.2 Option B: HashingVectorizer + SGDClassifier(log_loss)

Recommended if ComplementNB does not calibrate well.

Suggested classifier:

```python
SGDClassifier(
    loss="log_loss",
    penalty="l2",
    class_weight="balanced",
    alpha=1e-5,
    max_iter=20,
    random_state=42,
)
```

External reference:

- SGDClassifier documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
- Out-of-core text classification example: https://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html

Pros:

- Better decision boundaries than NB in some cases.
- Supports incremental/out-of-core training via `partial_fit`.
- Often strong for sparse text classification.

Cons:

- More hyperparameters.
- Probability calibration may still need tuning.
- Training can be less deterministic if not carefully configured.

### 7.3 Option C: Rule-first + lightweight classifier

This is not an alternative to the model; it is a companion strategy.

Use deterministic Gmail filters or local rules for:

```text
from_domain
from_address
list_id
to/cc alias
known automated notifications
known newsletters
bank/vendor notifications
GitHub/GitLab/Jira/Linear notifications
```

Then reserve ML classification for ambiguous content-based labels.

Pros:

- Zero ML memory for deterministic cases.
- Better precision for sender/list-driven labels.
- Reduces workload on the classifier.

Cons:

- Needs maintenance.
- Rules can become messy if overused.

External reference:

- Gmail filter creation help: https://support.google.com/mail/answer/6579

### 7.4 Option D: Remote LLM or hosted embedding API

Use a hosted model only for uncertain messages or as a fallback.

Pros:

- Minimal VM memory.
- Better semantic understanding.
- Can handle labels with little local training data.

Cons:

- Cost.
- Latency.
- Privacy implications.
- Requires careful prompt/version management.
- More moving parts.

Use this only if local lightweight classification fails to provide acceptable coverage.

### 7.5 Option E: Semantic index precomputed locally

Build an offline semantic artifact locally:

```text
runtime_embeddings.npy
runtime_labels.npy
runtime_ids.txt
runtime_meta.json
```

Deploy only these artifacts to GCP.

Pros:

- Reduces startup peak.
- Keeps semantic KNN quality for the existing training set.

Cons:

- GCP still needs an embedder for every new message unless embedding is done remotely.
- FastEmbed remains resident.
- Does not fully solve e2-micro memory pressure.

This is worth doing as a semantic-mode improvement, but it should not be the primary GCP plan.

---

## 8. Implementation plan

### Phase 0: Measure baseline memory and quality

Purpose: avoid guessing and create before/after evidence.

#### Tasks

1. Add a memory profiling helper:

```text
scripts/profile_startup.py
```

2. Add logging checkpoints to current startup:

```text
before loading training DB
after loading training DB
after loading skip DB
after creating Embedder
after prepare_texts
after cache lookup
after embedding misses
after final embedding matrix
after malloc_trim
```

3. Capture these metrics locally and on GCP:

```text
RSS MB
startup seconds
number of training messages
number of skip examples
embedding matrix shape
embedding cache hit rate
number of labels
```

4. Freeze current semantic evaluation numbers from the existing corpus.

#### Suggested implementation

Add optional dependency:

```text
psutil
```

or use standard library `resource` on Unix for a minimal version.

Example helper:

```python
import os
import psutil

def log_memory(stage: str) -> None:
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print(f"[mem] {stage}: {rss_mb:.1f} MB", flush=True)
```

#### Acceptance criteria

- A local and GCP memory log exists.
- The biggest startup memory jumps are visible.
- Current semantic precision/coverage is captured as the baseline.

---

### Phase 1: Introduce classifier abstraction

Purpose: separate Gmail orchestration from classification implementation.

#### Tasks

1. Create:

```text
src/gmail_classifier/classifiers/base.py
src/gmail_classifier/classifiers/semantic_knn.py
src/gmail_classifier/classifiers/lightweight.py
```

2. Move or wrap current KNN behavior in `semantic_knn.py`.

3. Change `scripts/classify_and_label.py` so it does not hard-code semantic startup.

4. Add CLI flag:

```text
--classifier-mode semantic|lightweight
```

5. Add environment fallback:

```text
CLASSY_CLASSIFIER_MODE=lightweight
```

6. Keep current behavior as default for local compatibility during the transition.

#### Target CLI examples

```bash
uv run scripts/classify_and_label.py --classifier-mode semantic --once --dry-run
uv run scripts/classify_and_label.py --classifier-mode lightweight --once --dry-run
```

#### Acceptance criteria

- Semantic mode produces the same results as before.
- Lightweight mode can be selected but may initially fail with a clear "model missing" error.
- Gmail fetching/labeling code is not duplicated.

---

### Phase 2: Implement lightweight feature extraction

Purpose: create a deterministic representation used by both training and runtime.

#### New file

```text
src/gmail_classifier/lightweight_features.py
```

#### Suggested API

```python
def normalize_email_address(address: str) -> str:
    ...

def extract_domain(address: str) -> str:
    ...

def build_lightweight_text(message, max_body_chars: int = 4000) -> str:
    ...
```

#### Feature format

Example output:

```text
FROM_ADDRESS=notifications@github.com FROM_ADDRESS=notifications@github.com
FROM_DOMAIN=github.com FROM_DOMAIN=github.com FROM_DOMAIN=github.com
LIST_ID=github.com LIST_ID=github.com LIST_ID=github.com
FROM_NAME=GitHub
SUBJECT=Pull request opened in repo
BODY=...
```

#### Important details

- Normalize case for email addresses and domains.
- Strip angle brackets and display-name formatting.
- Preserve enough subject/body text to classify content.
- Hard-truncate body text before vectorization.
- Represent missing fields explicitly only if helpful, for example `NO_LIST_ID`.

#### Acceptance criteria

- Same message produces same lightweight text in training and runtime.
- Unit tests cover from-address parsing, domain extraction, list-id handling, empty body, and truncation.

---

### Phase 3: Add local training script for lightweight model

Purpose: train locally and create a compact artifact for GCP.

#### New script

```text
scripts/train_lightweight_classifier.py
```

#### Inputs

```text
--training-db data/training.db
--skip-db data/inbox_sample.db
--output data/models/lightweight_classifier.joblib
--meta-output data/models/lightweight_classifier_meta.json
--exclude-labels ...
--body-chars 4000
--n-features 262144
--model complement_nb|sgd_log_loss
```

#### Training logic

1. Load training messages.
2. Load skip messages if present.
3. Assign skip examples label `__skip__`.
4. Exclude configured labels.
5. Build lightweight feature text for each message.
6. Train vectorizer + classifier pipeline.
7. Evaluate with holdout or cross-validation.
8. Save artifact and metadata.

#### Initial model pipeline

```python
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    HashingVectorizer(
        n_features=2**18,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        dtype=np.float32,
    ),
    ComplementNB(alpha=0.1),
)
```

#### Metadata metrics to save

```text
message_count
skip_count
label_count
per_label_counts
train/test split strategy
precision_at_thresholds
coverage_at_thresholds
false_positive_count
false_positive_examples_count
artifact_size_bytes
created_at
repo_commit_if_available
```

#### Acceptance criteria

- Running the script creates `lightweight_classifier.joblib` and metadata.
- Training completes locally without requiring FastEmbed.
- The artifact can be loaded by a separate Python process and classify one message.

---

### Phase 4: Add evaluation script comparing semantic and lightweight modes

Purpose: quantify the quality degradation and tune thresholds.

#### New script

```text
scripts/evaluate_classifiers.py
```

#### Inputs

```text
--training-db data/training.db
--skip-db data/inbox_sample.db
--model data/models/lightweight_classifier.joblib
--semantic
--lightweight
--thresholds 0.80,0.85,0.90,0.95,0.98,0.99
--min-margins 0.05,0.10,0.20,0.30
```

#### Required metrics

Measure for each mode:

```text
precision
coverage
false positives
false negatives
per-label precision
per-label coverage
confusion matrix
skip false-positive rate
number of messages left unlabeled
```

The primary metric should be precision at a given coverage, not raw accuracy.

#### Threshold policy

Initial recommendation:

```text
Apply real label only if:
  predicted_label != "__skip__"
  predicted_probability >= 0.98
  probability_margin_to_second_label >= 0.20

Otherwise:
  no label
```

Medium confidence optional behavior:

```text
If probability >= 0.90 and margin >= 0.10:
  optionally apply AI-Predicted / Review label
else:
  no label
```

Do not apply a final Gmail label at medium confidence until observed false positives are acceptable.

#### Acceptance criteria

- A report shows semantic baseline and lightweight result side by side.
- The recommended threshold is selected based on measured precision and coverage.
- Lightweight mode has acceptable precision, even if coverage is lower.

---

### Phase 5: Implement lightweight runtime

Purpose: make GCP service classify without FastEmbed or full corpus load.

#### New class

```text
src/gmail_classifier/classifiers/lightweight.py
```

#### Suggested API

```python
class LightweightEmailClassifier:
    def __init__(self, model_path: str, high_threshold: float, review_threshold: float, min_margin: float):
        ...

    def classify_message(self, message) -> ClassificationResult:
        ...
```

#### Runtime classification logic

1. Build lightweight text from message.
2. Get probabilities or scores from model.
3. Sort top labels.
4. If top label is `__skip__`, return no label.
5. If top probability and margin pass threshold, return label.
6. Otherwise return no label or review action.

#### Explanation payload

Include:

```json
{
  "mode": "lightweight",
  "top_labels": [
    ["LabelA", 0.991],
    ["LabelB", 0.041],
    ["__skip__", 0.020]
  ],
  "margin": 0.950,
  "threshold": 0.980,
  "min_margin": 0.200
}
```

#### Acceptance criteria

- GCP lightweight startup does not import `fastembed`.
- GCP lightweight startup does not call `MessageStore.load_all()` for training or skip data.
- The service can classify new inbox messages using only the model artifact.

---

### Phase 6: Add deployment target for lightweight GCP mode

Purpose: deploy only what the small VM needs.

#### New Make targets

```makefile
train-lightweight:
	uv run scripts/train_lightweight_classifier.py

eval-lightweight:
	uv run scripts/evaluate_classifiers.py --lightweight --semantic

gcp-deploy-lightweight:
	# sync source, credentials, and lightweight model artifact only
	# install lightweight dependency set
	# set CLASSY_CLASSIFIER_MODE=lightweight
	# restart service
```

#### Deployment payload

Deploy:

```text
src/
scripts/classify_and_label.py
scripts/service runner files
pyproject.toml
uv.lock
credentials/
data/models/lightweight_classifier.joblib
data/models/lightweight_classifier_meta.json
```

Avoid deploying by default:

```text
data/training.db
data/inbox_sample.db
data/embeddings.db
raw exported messages
local evaluation outputs
```

#### Service config

In the systemd service environment:

```text
CLASSY_CLASSIFIER_MODE=lightweight
CLASSY_LIGHTWEIGHT_MODEL=data/models/lightweight_classifier.joblib
CLASSY_HIGH_THRESHOLD=0.98
CLASSY_REVIEW_THRESHOLD=0.90
CLASSY_MIN_MARGIN=0.20
```

#### Acceptance criteria

- `make gcp-deploy-lightweight` does not upload the raw training database by default.
- `journalctl` logs show lightweight mode and model metadata at startup.
- Startup memory is materially lower than semantic mode.

---

### Phase 7: Tune and harden

Purpose: improve quality while keeping memory low.

#### Tuning knobs

```text
n_features: 2**16, 2**18, 2**20
ngram_range: (1,1), (1,2), (1,3)
body_chars: 1000, 2000, 4000, 8000
field weights: sender/list repetition counts
ComplementNB alpha: 0.01, 0.1, 0.5, 1.0
threshold: 0.95, 0.98, 0.99
margin: 0.10, 0.20, 0.30
```

#### Hardening tasks

1. Save model metadata and print it at service startup.
2. Refuse to start if model artifact is missing or incompatible.
3. Add a dry-run GCP smoke test target.
4. Log every applied label with mode, confidence, and margin.
5. Log skipped predictions only in debug mode to reduce noise.
6. Add a rollback target that sets mode back to semantic or stops service.

#### Acceptance criteria

- False-positive examples are easy to inspect.
- Threshold changes do not require retraining.
- Model retraining is repeatable.

---

## 9. Changes to current semantic mode

Even if GCP uses lightweight mode, semantic mode should be made less fragile.

### 9.1 Avoid full training object load

Add streaming storage methods:

```python
def iter_all(self, batch_size: int = 500):
    ...

def iter_lightweight_rows(self, batch_size: int = 500):
    ...
```

This reduces local and semantic-mode peak memory.

### 9.2 Build semantic runtime index offline

Add:

```text
scripts/build_semantic_index.py
```

Artifacts:

```text
data/models/semantic_embeddings.npy
data/models/semantic_labels.json
data/models/semantic_ids.json
data/models/semantic_meta.json
```

Then semantic runtime can load an index directly rather than rebuilding it from DBs at every start.

### 9.3 Pre-normalize semantic embeddings

Store unit-normalized embeddings and classify with dot product:

```python
sims = embeddings @ query_unit
idx = np.argpartition(sims, -k)[-k:]
idx = idx[np.argsort(sims[idx])[::-1]]
```

This avoids recomputing norms and sorting the full array for every incoming email.

### 9.4 Cap skip examples

Keep skip examples representative rather than unlimited.

Suggested policy:

```text
max_skip_examples_total: 5000
max_skip_examples_per_sender_domain: 100
max_skip_examples_per_month: configurable
sampling: reservoir or recency-weighted
```

This benefits both semantic and lightweight training.

---

## 10. Quality strategy

### 10.1 Preserve precision first

For automatic email labeling, false positives are more damaging than missed labels. The lightweight model should therefore prefer:

```text
high precision
medium coverage
many no-label decisions
```

over:

```text
high coverage
medium precision
many incorrect labels
```

### 10.2 Use `__skip__` consistently

The existing project insight remains important: negative examples are essential. Continue treating inbox/no-label examples as `__skip__` during training.

Rules:

1. If `__skip__` wins, do not label.
2. If a real label wins but `__skip__` is close, do not label.
3. Include `__skip__` in threshold/margin calculations.

### 10.3 Evaluate labels separately

Some labels may be content-learnable; others may be sender/list driven or too ambiguous.

Classify labels into groups:

```text
A. safe for lightweight auto-labeling
B. safe only with higher thresholds
C. should be Gmail filters/rules
D. should remain manual
```

The lightweight model should support per-label thresholds later:

```json
{
  "Receipts": {"threshold": 0.99, "margin": 0.30},
  "Newsletters": {"threshold": 0.95, "margin": 0.10},
  "Travel": {"threshold": 0.98, "margin": 0.20}
}
```

---

## 11. GCP runtime strategy

### 11.1 Preferred target

Keep the current always-on VM model if the lightweight runtime fits comfortably.

GCP free-tier reference:

- https://docs.cloud.google.com/free/docs/free-cloud-features

### 11.2 Memory budget target

Set explicit budgets:

```text
startup RSS target: less than 250 MB
steady-state RSS target: less than 300 MB
classification memory spike: less than 50 MB
startup time target: less than 15 seconds after dependencies installed
```

These are engineering targets, not guarantees. The exact budget should be adjusted after measurement.

### 11.3 Cloud Run alternative

Cloud Run becomes more attractive only after the runtime is lightweight. A semantic local embedding runtime is a poor Cloud Run fit because cold starts and model memory will still hurt.

Cloud Run reference:

- https://docs.cloud.google.com/run/docs/configuring/services/memory-limits

Potential Cloud Run shape:

```text
Gmail Pub/Sub -> Cloud Run endpoint -> classify one message -> apply label -> exit/scale down
```

Use this later if the always-on VM remains annoying to operate.

---

## 12. Suggested pull request breakdown

### PR 1: Instrumentation

Files:

```text
scripts/profile_startup.py
src/gmail_classifier/memory.py
```

Outcome:

- Current memory behavior is visible.

### PR 2: Classifier abstraction

Files:

```text
src/gmail_classifier/classifiers/base.py
src/gmail_classifier/classifiers/semantic_knn.py
scripts/classify_and_label.py
```

Outcome:

- Existing semantic behavior still works.
- Runtime can select classifier mode.

### PR 3: Lightweight features and model training

Files:

```text
src/gmail_classifier/lightweight_features.py
scripts/train_lightweight_classifier.py
tests/test_lightweight_features.py
pyproject.toml
```

Outcome:

- Local training creates a model artifact.

### PR 4: Evaluation

Files:

```text
scripts/evaluate_classifiers.py
src/gmail_classifier/evaluation.py
```

Outcome:

- Semantic and lightweight modes are compared with precision/coverage metrics.

### PR 5: Lightweight runtime

Files:

```text
src/gmail_classifier/classifiers/lightweight.py
scripts/classify_and_label.py
```

Outcome:

- GCP service can run without FastEmbed.

### PR 6: GCP lightweight deployment

Files:

```text
Makefile
gcp-deploy-plan.md
README.md
service/systemd templates if present
```

Outcome:

- `make gcp-deploy-lightweight` deploys only compact runtime assets.

### PR 7: Semantic-mode cleanup

Files:

```text
src/gmail_classifier/storage.py
scripts/build_semantic_index.py
src/gmail_classifier/classifiers/semantic_knn.py
```

Outcome:

- Semantic mode has lower startup peak and faster classification.

---

## 13. Acceptance criteria for the full effort

### 13.1 Runtime acceptance

The lightweight GCP service is acceptable if:

```text
- it starts reliably on the target GCP VM;
- it does not import FastEmbed in lightweight mode;
- it does not load training.db or inbox_sample.db at service startup;
- it processes new mail without OOM/restart loops;
- steady-state RSS is comfortably below the VM memory limit;
- logs clearly identify classifier mode and model version.
```

### 13.2 Quality acceptance

The lightweight model is acceptable if:

```text
- precision remains high at the selected threshold;
- skip false-positive rate is very low;
- coverage degradation is understood and documented;
- unsafe labels can be excluded or given stricter thresholds;
- no-label behavior is common and acceptable.
```

Concrete initial target:

```text
precision: >= 97 percent on held-out/evaluation messages
coverage: best achievable while keeping precision target
skip false positives: near zero, manually inspected
```

If this target is not met, try per-label thresholds and rule-first filtering before using a remote LLM.

---

## 14. Risks and mitigations

### Risk 1: Lightweight model labels too aggressively

Mitigation:

- Raise threshold.
- Add margin requirement.
- Add per-label thresholds.
- Treat `__skip__` proximity as veto.
- Use review label before final label.

### Risk 2: Lightweight model has poor coverage

Mitigation:

- Add Gmail filters for deterministic labels.
- Increase body character cap.
- Tune n-grams and field weights.
- Try SGDClassifier.
- Add remote LLM fallback for uncertain messages.

### Risk 3: Hashing collisions hurt quality

Mitigation:

- Increase `n_features` from `2**18` to `2**20`.
- Compare against CountVectorizer locally for diagnosis.
- Use per-label evaluation to identify affected labels.

### Risk 4: Model artifact still pulls heavy dependencies

Mitigation:

- Check import graph in lightweight mode.
- Keep `fastembed` out of the lightweight runtime path.
- Consider a custom `.npz` ComplementNB runtime if scikit-learn import cost is too high.

### Risk 5: Training/evaluation split leaks near-duplicate emails

Mitigation:

- Split by sender/domain/date where possible.
- Evaluate on recent messages not used in training.
- Manually inspect false positives.

---

## 15. Fallback plan

If lightweight classification is not good enough:

1. Add deterministic Gmail filters/rules for high-volume easy labels.
2. Use lightweight classifier for safe labels only.
3. Use hosted LLM fallback only for uncertain messages.
4. Move from `e2-micro` to a slightly larger VM if zero-cost is less important than quality.
5. Continue using semantic KNN locally for periodic offline labeling/batch review.

---

## 16. Proposed command workflow

### Local training and evaluation

```bash
make fetch-training
make fetch-inbox
make train-lightweight
make eval-lightweight
```

### Local dry run

```bash
uv run scripts/classify_and_label.py \
  --classifier-mode lightweight \
  --once \
  --dry-run
```

### GCP deploy

```bash
make gcp-deploy-lightweight
make gcp-start
make gcp-logs
```

### Rollback

```bash
make gcp-stop
# either redeploy semantic mode on a larger machine or restore previous service config
make gcp-start
```

---

## 17. Suggested implementation details

### 17.1 pyproject dependency options

Option 1: add scikit-learn globally:

```toml
dependencies = [
  "numpy",
  "beautifulsoup4",
  "google-api-python-client",
  "google-auth-oauthlib",
  "google-cloud-pubsub",
  "fastembed",
  "scikit-learn",
  "joblib",
]
```

Option 2: split extras:

```toml
[project.optional-dependencies]
semantic = ["fastembed"]
lightweight = ["scikit-learn", "joblib"]
dev = ["pytest"]
```

The cleaner long-term design is option 2, but option 1 is faster if packaging complexity is not worth it yet. The critical requirement is that lightweight runtime code must not import semantic modules that import FastEmbed.

### 17.2 Avoid accidental semantic imports

Bad:

```python
from gmail_classifier.embeddings import Embedder
```

at top level in `classify_and_label.py`.

Better:

```python
if args.classifier_mode == "semantic":
    from gmail_classifier.classifiers.semantic_knn import SemanticKnnClassifier
else:
    from gmail_classifier.classifiers.lightweight import LightweightEmailClassifier
```

### 17.3 Threshold helper

Implement one shared helper for score decisions:

```python
def decide_from_scores(top_label, top_score, second_score, high_threshold, review_threshold, min_margin):
    margin = top_score - second_score
    if top_label == SKIP_LABEL:
        return Action.NO_LABEL
    if top_score >= high_threshold and margin >= min_margin:
        return Action.LABEL
    if top_score >= review_threshold and margin >= min_margin:
        return Action.LABEL_WITH_REVIEW
    return Action.NO_LABEL
```

This makes semantic and lightweight behavior easier to compare.

---

## 18. First-week execution checklist

1. Add memory logging to current startup.
2. Run current semantic mode locally and on GCP once.
3. Add `lightweight_features.py` and unit tests.
4. Add `train_lightweight_classifier.py`.
5. Train ComplementNB artifact locally.
6. Add `evaluate_classifiers.py` with threshold sweep.
7. Pick initial threshold/margin based on false positives.
8. Add lightweight runtime mode.
9. Add `gcp-deploy-lightweight`.
10. Deploy to GCP and verify startup RSS.
11. Run dry-run for 24-48 hours or equivalent sample size.
12. Enable label application only after inspecting proposed labels.

---

## 19. Recommended final state

The best final shape is:

```text
Local Mac:
  - fetch/retrain data
  - evaluate semantic and lightweight modes
  - build lightweight model artifact
  - optionally build semantic benchmark index

GCP VM:
  - run Gmail watcher
  - load lightweight artifact only
  - classify conservatively
  - apply labels only when confidence and margin are high
  - record skipped/uncertain messages

Gmail:
  - deterministic filters handle obvious sender/list/alias cases
  - classifier handles ambiguous content-based labels
```

This preserves the quality work already done in `classy`, while changing the production runtime to match the memory constraints of the GCP environment.

---

## 20. Reference links

Repository and project files:

- Repository: https://github.com/nodet/classy
- README: https://github.com/nodet/classy#readme
- Semantic labeling plan: https://raw.githubusercontent.com/nodet/classy/main/gmail-semantic-labeling-plan.md
- Runtime script: https://raw.githubusercontent.com/nodet/classy/main/scripts/classify_and_label.py
- Storage: https://raw.githubusercontent.com/nodet/classy/main/src/gmail_classifier/storage.py
- Training: https://raw.githubusercontent.com/nodet/classy/main/src/gmail_classifier/training.py
- Embedder: https://raw.githubusercontent.com/nodet/classy/main/src/gmail_classifier/embeddings.py
- Current classifier: https://raw.githubusercontent.com/nodet/classy/main/src/gmail_classifier/classifier.py
- Project dependencies: https://raw.githubusercontent.com/nodet/classy/main/pyproject.toml

External references:

- Google Cloud free tier: https://docs.cloud.google.com/free/docs/free-cloud-features
- Google Cloud Run memory limits: https://docs.cloud.google.com/run/docs/configuring/services/memory-limits
- scikit-learn HashingVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html
- scikit-learn ComplementNB: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html
- scikit-learn SGDClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
- scikit-learn out-of-core text classification example: https://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html
- Gmail filters help: https://support.google.com/mail/answer/6579
