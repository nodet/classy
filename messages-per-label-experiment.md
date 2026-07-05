# Plan: Experiment — how many messages per label the classifier actually needs

Date: 2026-07-05
Status: proposed

## Goal and the decision it informs

Find the smallest **per-set example count** at which the KNN classifier still works correctly,
so the live service stores no more vectors than it needs. This directly sets the default for
**`--max-per-label`** in `stateless-bootstrap-plan.md` ("Bounding the bootstrap total"), where
the cap applies uniformly to every set including the `__skip__` pool. It also feeds issue #1's
per-set coreset target — the eventual runtime bound on index growth.

Output: a per-label and aggregate **learning curve** (quality vs. examples/label) with a
recommended cap and the evidence behind it, not a single guessed number.

## What already exists (reuse map — do NOT rebuild)

- **Capture:** `scripts/fetch_training_data.py --max-per-label 500` → `data/training.db`
  (one label per message, `--max-per-label` default already 500). `scripts/fetch_inbox.py
  --count 500` → `data/inbox_sample.db` (the skip pool; `--count` default is already 500 and it
  slices the INBOX listing, so "500 inbox" needs no code change).
- **Embedding:** `training.build_training_data(messages, embedder)` → `(embeddings, labels,
  ids)`. `embeddings.Embedder` is the FastEmbed runtime — the expensive step; embed once.
- **KNN + decision:** `classifier.find_neighbors` / `aggregate_scores` / `compute_confidence`
  / `decide_action`, thresholds `HIGH=0.95` / `MEDIUM=0.80`, `MIN_EXAMPLES_PER_LABEL=5`,
  `SKIP_LABEL="__skip__"` (`classifier.py:22-25`).
- **Metrics:** `evaluation.precision_at_threshold` / `coverage_at_threshold` /
  `compute_metrics_table` / `per_label_precision`; `cross_validation.PredictionResult`.
- **Existing driver to mirror:** `scripts/train_and_evaluate.py` (loads DBs, excludes configured
  labels, prints the metrics table + per-label precision + error samples).

The **only** genuinely new code is a holdout evaluator (fixed test set, capped training set) and
a sweep driver. Everything downstream of a `List[PredictionResult]` is reused verbatim.

## Why the naive "re-run LOO at each cap" is not quite right

Re-running leave-one-out with N messages/label changes *two* variables at once: the training
size **and** the set of messages being tested (and the denominator of every metric). Curves
across N are then not comparable, and small-N points look artificially good or bad depending on
which messages happened to survive the cap.

Fix instead a **held-out test set per label**, identical across every N, and vary only the
**training pool** the model is built from. That is a textbook *learning curve* and answers the
question directly: "with N examples/label, how many of the same held-out messages does the
classifier get right?"

## Design

### Phase 0 — Capture and embed once (store for later, as you suggested)

- Fetch **500 per user label** into `data/training.db` (already the default) and **500 inbox
  messages** into `data/inbox_sample.db` via `fetch_inbox --count 500` (already the default; it
  slices the listing). 500 is the ceiling of the sweep; keep the raw DBs as the durable
  "just in case" corpus.
- Embed **every** captured message exactly once via `build_training_data` and persist vectors to
  `data/experiments/vectors.npz` keyed by message id (plus a sidecar `labels.json`:
  `id → user-label` and `id → "__skip__"`). All sweeps below are then **pure NumPy over cached
  vectors** — no FastEmbed, no re-fetch — so hundreds of trials are cheap and reproducible.
- Record the embedding-model fingerprint alongside the cache so a model change invalidates it
  (mirrors the bootstrap plan's `ml_fingerprint`).

### Phase 1 — Fixed test split, then a training-size sweep

Per **user label** (the skip pool is handled in Phase 3):

- **Split once, deterministically** (fixed seed) into a held-out **test set** and a **train
  pool**:
  `test_n(label) = min(TEST_CAP=50, floor(0.30 * label_size))`; the remaining messages are the
  train pool. Persist the split (list of test ids per label) so every N and every seed evaluates
  the *same* test messages. Labels with fewer than `MIN_TEST=10` test messages are flagged
  low-confidence in the report, not silently averaged in.
- **Sweep** `N ∈ {5, 10, 20, 35, 50, 75, 100, 150, 200, 350, 500}` (denser near the expected
  knee than your original 20/50/100/200/500 — a learning curve's interesting region is the
  low end). For each N: sample `min(N, pool_size)` from the train pool. When
  `pool_size < N`, the curve for that label **stops** — that exhaustion point is itself the
  answer for that label ("cannot benefit from a higher cap").
- **Repeat** each (label, N) over `R = 8` random seeds (different training subsamples, same test
  set) and report **mean ± std**. Low-N variance is large; a single draw would be misleading.

New function `holdout_predict(train_embs, train_labels, test_embs, test_labels, k)` →
`List[PredictionResult]`: mirrors the inner body of `leave_one_out` (eligibility gate, neighbors,
aggregate, confidence, skip→no-label, `decide_action`) but over an explicit test set instead of
index `i`. It reuses the classifier primitives directly; no new metric or decision logic.

### Phase 2 — What "correctly classified" means (make it operational)

The service's costly mistake is **mislabeling**, not abstaining. Define per test message:

- **True user-label message:** *correct* iff `predicted_label == true_label` **and**
  `confidence ≥ operating threshold`. `NO_LABEL`/below-threshold is a **miss** (safe, hurts
  coverage). A confident **wrong** label is a **mislabel** (the expensive error).
- **Skip message** (true `__skip__`): *correct* iff the classifier does **not** apply a user
  label (predicts skip or stays below threshold). Applying a label is a **false positive**.

Report, as functions of N, at both operating thresholds (**0.80** and **0.95**):

- **Precision** = correct / labeled (via `precision_at_threshold`) — the primary curve.
- **Coverage** = labeled / total (via `coverage_at_threshold`) — the secondary curve.
- **Mislabel rate** and **skip false-positive rate** broken out (these are what precision hides).
- **Per-label precision/coverage** (via `per_label_precision`) — the aggregate curve hides that
  some labels plateau at 20 and others need 200; the cap must satisfy the *limiting* label.

### Phase 3 — Skip-pool size as a second axis

`__skip__` is "one more set" (our committed framing), but it behaves differently: it is the
reject mass and sits in the confidence denominator, so it trades coverage for precision rather
than adding a class. Two sub-questions:

1. **Lockstep (primary):** run Phase 1 with the skip pool also capped at N. This is the
   apples-to-apples "one cap for everything" curve that sets `--max-per-label`.
2. **Skip-only mini-sweep (secondary):** fix label N at the Phase-1 knee, vary skip
   ∈ {0, 20, 50, 100, 200, 500}. Shows how much skip mass precision needs and whether the skip
   pool can be capped *lower* than labels — an input to whether one cap or two is justified
   (revisit the "one value for both" decision only if the data demands it).

### Phase 4 — Pick the cap from the curves

Operationalize "works correctly" as a stopping rule, applied per label and in aggregate:

- **Recommended cap = smallest N such that** precision@0.80 ≥ `PRECISION_TARGET` (start at
  **0.97**, matching the bootstrap/eval target) **and** coverage@0.80 ≥ `coverage(N=500) − ε`
  (within, say, 2 pts of the asymptote). I.e. the knee where more examples stop buying quality.
- Report the aggregate cap **and** the per-label knees; the deployed `--max-per-label` should
  cover the limiting useful label, while noting any label whose own pool is already the bound.
- If the knee sits well below 500 (expected), that is the headline result: the live service can
  store far fewer vectors per set at no measurable quality cost.

## Deliverables

- `scripts/experiment_sample_size.py` — orchestrates Phase 0–4, writes:
  - `data/experiments/vectors.npz` + `labels.json` (durable embedded corpus).
  - `data/experiments/learning_curve.csv` — rows `(scope, label, N, seed, threshold, precision,
    coverage, mislabels, skip_fp, n_test)` (raw, for re-plotting).
  - `data/experiments/summary.md` — aggregate + per-label curves (mean ± std), the recommended
    cap, and the skip-only mini-sweep.
- `make experiment-sample-size` target (mirrors `make evaluate`).
- A short written conclusion feeding the `--max-per-label` default back into
  `stateless-bootstrap-plan.md`.

## Threats to validity (state them in the report)

- **The 500-cap sample is itself truncated.** Labels with >500 messages never expose their tail;
  the curve describes "up to 500," and a cap ≥ the knee is safe only within that window.
- **Historical, single-mailbox, point-in-time.** Learning-curve quality may not survive topic
  drift; re-run periodically rather than treating the cap as permanent.
- **LOO/holdout optimism.** Held-out test messages come from the same capture as training; real
  future mail is strictly harder. Prefer the *conservative* (higher-N) end of a flat knee.
- **Class imbalance.** Aggregate precision is prediction-weighted, so a few big labels dominate;
  this is why per-label curves and the limiting-label rule matter.
- **Skip staleness (wrinkle #7)** is a *runtime* effect and out of scope here; this experiment
  sizes the *initial* pool, not its long-run refresh.

## Verification

- Determinism: fixed seeds → identical splits and identical curves across runs; the persisted
  test-id set is stable across all N and seeds (assert same ids evaluated at N=20 and N=500).
- Cache correctness: `vectors.npz` round-trips; a stored fingerprint mismatch forces re-embed;
  no FastEmbed import on the sweep path once vectors are cached.
- Harness reuse: `holdout_predict` produces `PredictionResult`s that flow through the **existing**
  `compute_metrics_table` / `per_label_precision` unchanged; a unit test cross-checks that at
  "N = all-but-one" `holdout_predict` on a single held-out message agrees with `leave_one_out`
  for that message.
- Sanity: precision/coverage are monotone-ish and non-decreasing on average with N (allowing
  noise); a label whose pool < N stops rather than extrapolating; empty/low test sets are flagged
  not averaged.
- Small unit tests on the "correct/mislabel/skip-FP" tallying (Phase 2) with hand-built results.
