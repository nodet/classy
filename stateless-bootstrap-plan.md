# Plan: Self-bootstrapping GCP deployment — bootstrap from Gmail, persist only derived state

Date: 2026-06-30
Status: proposed (supersedes the lightweight-classifier direction for the GCP goal)

## Goal

Treat **Gmail as the single source of truth**. Make a fresh GCP deploy require only
*code + credentials* — no locally-built `training.db` / `inbox_sample.db` / `embeddings.db`
to upload. This is not fully stateless: the VM persists a local derived cache,
`data/state.db`, so restarts do not re-fetch the mailbox. The important invariant is:

> `state.db` contains only derived runtime state and may be discarded and rebuilt from Gmail
> at any time.

Trade a slow first boot for trivial deployment:

> create VM → push code + credentials → start. No user-specific training step.

As a bonus this **eliminates the ~447 MB startup transient** (the raw-corpus `load_all()`),
which was the last remaining memory fragility and the only reason the lightweight
classifier was still on the table.

## Why this works (premise check)

All three databases are *derivable* from Gmail; none is ground truth:

| File | What it is | Source of truth |
|---|---|---|
| `training.db` | labeled messages | Gmail labels — already rebuilt by `fetcher.py:22-39` |
| `inbox_sample.db` | skip pool (inbox = don't label) | current Gmail inbox |
| `embeddings.db` | `id → vector` cache | pure derivative (the model) |

At **runtime** the KNN index needs only `id → vector → label` (`TrainingIndex`,
`training_index.py:10-15`). Message **bodies are dead weight once embedded** — they exist
in `training.db` only to be re-embeddable. A self-bootstrapping design never persists bodies.

### The memory prize

Today's startup peak (~606 MB on the VM) is dominated by `MessageStore.load_all()` holding
every body in RAM at once (+447 MB), because *fetch* and *embed* are separate phases with a
DB between them. If bootstrap instead **fetches → embeds → caches → discards each message
one at a time** (the path we already adopted in `build_training_data` and that live mail
uses via `embedder.embed`), bodies never accumulate. Bootstrap peak ≈ model + base
≈ 200–250 MB. The transient doesn't shrink — it's gone.

## Design: bootstrap-on-empty, then persist derived-only

Chosen over "fully stateless / rebuild every boot": a crash-loop under
`Restart=on-failure` must not re-fetch ~4000 messages each cycle. So we persist the
*derived* state and only a fresh VM pays full bootstrap.

The invariants:

- Gmail is authoritative for message contents and labels.
- `state.db` is authoritative only for the local derived runtime cache.
- `state.db` may be deleted at any time and rebuilt from Gmail.
- `state.db` must never contain raw message text, subjects, senders, or body HTML.
- The Gmail history cursor is durable; startup must not skip live changes across crashes or
  deploys.

### Persisted state (derived, no bodies) — one file, `data/state.db`

A single SQLite file replaces all three of today's DBs. The runtime `TrainingIndex` is the
join `embeddings ⋈ labels on message_id`. **No message bodies/subjects/senders are stored
anywhere on the VM.** `state.db` still contains mailbox-derived metadata (message ids,
label ids/names, and embeddings), so it should be treated as private.

| Table | Columns | Role |
|---|---|---|
| `embeddings` | `message_id, vector, fingerprint, created_at, updated_at` | the vector cache |
| `labels` | `message_id, label_id, label_name_snapshot, source, updated_at` | `label_id` = Gmail label id, or `__skip__` for the skip pool |
| `pending_new` | `message_id, first_seen_history_id, reason, created_at` | post-boundary mail seen before the model is mature |
| `meta` | `key, value` | schema/config/ML fingerprint, bootstrap status, Gmail history cursors |

Minimum `meta` keys:

```text
state_schema_version
ml_fingerprint
excluded_labels_hash
bootstrap_status
bootstrap_boundary_history_id
last_processed_history_id
watch_expiration_ms
bootstrap_started_at
bootstrap_completed_at
```

`label_id` is the classifier's internal label identity. `label_name_snapshot` is only for
logs/status/debug output and can be refreshed from `LabelRegistry`. This avoids stale
predictions if a Gmail label is renamed.

(Name `state.db` chosen over keeping `embeddings.db` since it now holds more than
embeddings; one file = one connection, atomic, trivial to reset.)

### Startup logic (`scripts/classify_and_label.py:main`)

1. Open the derived store.
2. Validate `state_schema_version`, `ml_fingerprint`, and `excluded_labels_hash`.
   - If compatible and the store already has a usable index (`embeddings + id→label`),
     load it and enter the warm path.
   - If the ML fingerprint is incompatible, build `state.rebuild.db` from Gmail and
     atomically swap it into place only after validation succeeds. Do not delete the last
     usable `state.db` before the replacement exists. A fingerprint mismatch invalidates
     *vectors only* — the label map and `last_processed_history_id` are still valid, so the
     rebuild **recomputes embeddings but carries the existing cursor forward**. It must not
     re-pin a fresh watch boundary, or live changes during the (possibly long) rebuild are
     skipped — the same backlog-skipping bug the warm path is careful to avoid.
   - If the exclusion config changed, reconcile the derived state: remove now-excluded
     labels from the index, bootstrap newly-included labels, and reuse embeddings for
     unchanged message ids where possible.
3. If empty (fresh VM) → **bootstrap from Gmail**:
   - Call `client.watch(PUBSUB_TOPIC)` first and persist both
     `bootstrap_boundary_history_id` and `last_processed_history_id` from the returned
     `historyId`.
   - `list_user_labels()` minus excluded (XLC/XLE/XLCap).
   - For each label: `list_message_ids(label_id, max_results=--max-per-label)`.
   - For the skip pool: list recent INBOX ids, **minus any id that already carries a
     user label** (see "Labeled wins over skip" below).
   - For each id **not already embedded**: `get_message` → parse → `build_text_representation`
     → `embedder.embed` → `cache.put(id, vec)` + record `id→label_id` (or `__skip__`).
     Discard the raw message. **One at a time** — bounded memory, resumable.
   - Build `TrainingIndex` from the cache + label map.
4. Warm path:
   - Refresh the Gmail watch, but **do not replace** `last_processed_history_id` with the
     new watch id.
   - Process Gmail history from the persisted `last_processed_history_id`.
   - After successful event processing, persist the new Gmail history id.
   - If `history.list` says the cursor is expired/out of range, run a full sync/reconcile
     from Gmail, not an inbox-only poll.
5. Either path → the pubsub loop. The **cold** (bootstrap) path enters a progressive variant —
   see "Progressive bootstrap" below — so the service is live and safe from the first second
   rather than after a 20-min wait.

### Labeled wins over skip (the one semantic rule the single table needs)

The single `labels` table keyed on `message_id` eliminates the *structural* bug we hit
with two stores (a message could be a row in both `training.db` and `inbox_sample.db`,
producing a duplicate KNN vote and an orphaned, uncorrectable row). But one row per id
turns the conflict into **last-write-wins**, which is not automatically correct: a message
can hold a user label *and* still sit in INBOX (labeling doesn't archive it), so the skip
step would otherwise `upsert(id, '__skip__')` over a real label.

Rule: **a message that carries a real user label is a labeled example, never a skip
example.** Equivalently — when building the skip pool, exclude INBOX ids that already
carry a user label. (The reverse is correct and unchanged: an INBOX message with *no*
user label is a skip example.) This is the same guard the immediate two-store fix applies
at load time; the bootstrap applies it at the source so the conflict never reaches the
`labels` table.

Use two names in code to keep this clear:

```text
known_ids = all message ids present in labels, real labels and __skip__
skip_vote_ids = only message ids whose label_id is __skip__
```

Inbox/history processing skips `known_ids`, while the classifier votes with `skip_vote_ids`
as negative examples.

### Resumability (matters during the slow first boot)

Because each vector + label row is committed as computed, a crash at minute 15 of a 20-min
bootstrap **resumes** (step 2 skips already-embedded ids) rather than restarting. This is
why per-message caching (commit `fd0b6d6`) was worth doing.

Persisting `last_processed_history_id` is part of resumability: after a crash, the service
must replay Gmail history from the last durable cursor, not from a new watch boundary.

### Progressive bootstrap (read-only until mature)

A fresh VM has no cache, so step 2 is slow (~10–20 min). Rather than block the service
until it finishes, bootstrap **incrementally** while the pubsub loop is already live. Three
mechanisms make this both *useful early* and *safe early*.

#### Read-only until there is a cache (the hard safety boundary)

The current first boot does an **initial inbox check that labels the backlog**
(`_run_pubsub_mode` → `_check_inbox` → `apply_label`/archive). On a cache-less deploy that
is exactly wrong: the service would wake up and archive hundreds of emails that arrived
*before* it ever ran. Rule:

> When bootstrapping (no cache yet), **everything already in Gmail is read-only.** Bootstrap
> *reads* existing mail only to embed it into the index; it never labels or archives it.
> Only mail that arrives *after* the service starts is eligible to be labeled.

Mechanism: call `client.watch(PUBSUB_TOPIC)` **first**, before reading a single message, and
pin the returned `historyId` as the boundary. Anything at-or-before it = existing = read-only
forever; anything after it = new = classifiable (subject to the maturity gate below). Because
the subscription exists from the start, notifications for genuinely-new mail **accumulate**
during the slow bootstrap and are serviced as we go — none are lost. The labeling initial
inbox check is **removed from the cold path**. On warm restarts, history catch-up from
`last_processed_history_id` is the *only* labeling path: drop the labeling inbox check on the
warm path too. The reason it cannot simply be made boundary-safe is that there is no cheap
per-message signal to enforce the boundary at inbox-list time — `list_message_ids` returns no
per-message `historyId`, and `known_ids` is an incomplete guard because the skip pool is only
*sampled* (~50 + capped round-robin), so pre-boundary INBOX mail outside that sample is absent
from `known_ids` and would be labeled. History replay is inherently safe instead: pre-boundary
mail never appears in post-cursor history, so it stays untouched without any explicit check.

Keep the Gmail watch unfiltered: do not restrict it to INBOX. Label changes outside INBOX are
how the service learns user corrections.

#### Round-robin ordering (so the classifier is broad, not deep, early)

The naive bootstrap order is depth-first: finish label A, then B, then C. That is the worst
order for early usefulness, because `_eligible_labels` (`classifier.py:98`) only lets a label
win once it has **≥5 examples** — so for a long stretch the classifier can recognize A-type
mail and is blind to everything else.

Instead, **round-robin**: process one message from each label per round (and the skip pool —
see below), committing each vector+label as computed. After R rounds every label has ~R
examples and they all cross the eligibility line together. The memory and resumability
properties are order-independent, so this is free; a half-finished round-robin is already a
working *broad* classifier on the next boot.

The **skip pool is loaded similarly, but front-loaded**: take ~50 inbox messages first (the
safety mass — see the maturity gate), then round-robin across both the user labels *and* the
inbox for the remainder.

#### Two gates: read-only boundary vs. maturity

These are independent and must not be conflated:

- **Read-only gate** (above): existing vs. new mail. Existing is *never* labeled, no matter
  how mature the model becomes. Permanent, per-message, decided by the pinned `historyId`
  and enforced by processing only post-boundary history.
- **Maturity gate**: even genuinely-new mail is not labeled until the index is broad enough —
  approximately 20 examples per eligible user label and the skip pool loaded.

Use finite targets so small labels do not block forever:

```text
label_target(label) = min(MATURITY_EXAMPLES_PER_LABEL, available_count_for_label)
skip_target = min(SKIP_MATURITY_TARGET, available_unlabeled_inbox_count)
```

Labels with fewer than `MIN_EXAMPLES_PER_LABEL` available examples do not block maturity;
they remain ineligible to win until they have enough examples.

Confidence is `winning_score / total_score` with `__skip__` neighbors in the denominator
(`classifier.py:73,137`); without the skip mass loaded, early confidence is spuriously high
and the service **over-labels**. Since the live path applies *and archives* at ≥0.80
(`inbox_check.py:83,93`), an early mistake is a semi-irreversible action on the mailbox — so
the gate is conservative by design.

Consequence (accepted): new mail arriving during early warmup, before the maturity gate opens,
stays **unlabeled in the inbox**. It is recorded in `pending_new`; it is **not** written as
`__skip__`, not archived, and not treated as final. When the maturity gate opens, process
`pending_new` through the normal classifier and then remove each row. Only after that normal
pass may a still-no-label result become a `__skip__` example. Because `pending_new` stores no
body (only `message_id` + `history_id`, by design), draining it re-`get_message`s each parked
id — the one place a "new" message is read from Gmail twice. The count is bounded by mail
volume during the warmup window, so it is small in practice.

#### Single-threaded interleave (not a background thread)

Do the bootstrap *in* the pubsub loop, not a side thread. `TrainingIndex.add` reassigns
`self.embeddings` via `np.vstack` and mutates a list + dict (`training_index.py:23-35`); a
concurrent `classify` reading `self.embeddings` mid-`vstack` races on a half-built array, and
two threads would share one FastEmbed model and compete for the e2-micro's single core. Instead,
process one bounded round-robin batch *between* `run_iteration` calls: a batch (for example,
max 25 messages or max 5 seconds), then service any pending notification, repeat until the
corpus is exhausted.

For index growth, avoid one `np.vstack` per bootstrapped message. Add `TrainingIndex.extend(...)`
or publish immutable batch snapshots from a small builder, then classify against the current
complete snapshot between batches.

This is single-threaded (no lock, no index race, one embedder caller) and naturally throttled
(live mail preempts bootstrap between batches). Bootstrap finishes somewhat later in wall-clock —
the tradeoff we already accept, since the goal is early responsiveness, not fast completion.

#### First-boot summary

`watch()` → persist boundary + cursor → load ~50 skip → round-robin labels + inbox, committing
each vector (resumable) → defer genuinely-new mail into `pending_new` until mature → once
finite maturity targets are met, classify **new** mail and drain `pending_new` → existing mail
stays untouched forever. Single-threaded interleave in the pubsub loop.

### Pub/Sub acknowledgement rule

Treat Pub/Sub messages as wakeups, but do not acknowledge them before durable Gmail history
processing.

Required order:

```text
pull notification(s)
read Gmail history from last_processed_history_id
process events idempotently
persist new last_processed_history_id
ack Pub/Sub message(s)
```

If the service crashes before ack, Pub/Sub can redeliver and Gmail history replay remains
idempotent. If it crashes after ack, the durable cursor has already moved.

### Live adaptation (preserve today's behavior)

`label_change_handler.process_label_changes` currently writes bodies to
`training_store`/`skip_store` *and* updates the in-memory index
(`label_change_handler.py:109,117,139,144`). Under the new model the in-memory `index.add`
stays; the persistence target changes from "save body to MessageStore" to "upsert
`id→label_id` + `cache.put(id, vec)` in the derived store." Same learn-on-correction behavior,
no bodies persisted. The `index.add(...)` calls are untouched, or replaced with `index.extend(...)`
where batching is available.

The live update points map 1:1 from today's body-writes to label+vector upserts:
- new inbox msg after maturity → skip (`inbox_check`/`history_processor` save empty-label body)
  ⇒ `labels.upsert(id, '__skip__')` + `embeddings.put(id, vec)`
- new inbox msg before maturity ⇒ `pending_new.upsert(id, history_id, 'immature')`; no label row yet
- label applied/corrected (`label_change_handler.py:109,117`) ⇒
  `labels.upsert(id, label_id)` + `embeddings.put(id, vec)` + index update
- label removed back to inbox (`label_change_handler.py:139,144`) ⇒
  `labels.upsert(id, '__skip__')`

## State lifecycle (the persistence guarantees)

The whole point is that **derived state survives restarts and deploys, and is rebuilt only
when it is stale or explicitly reset**:

| Event | Behavior |
|---|---|
| **Service restart** | `state.db` present → validate, load index, refresh watch, process history from durable `last_processed_history_id`. No full Gmail fetch if the cursor is valid. |
| **Code deploy, ML/config unchanged** | `gcp-deploy.sh` builds the tarball with `--exclude='data'` and untars *over* `INSTALL_DIR`; `tar x` never deletes files absent from the archive, so `data/state.db` is untouched. Deploy + restart → fast, state preserved. |
| **ML changed** (embedding model or `build_text_representation`) | Startup compares `meta.ml_fingerprint` to the current code. Mismatch ⇒ cached vectors are stale. Build `state.rebuild.db` from Gmail, validate counts/fingerprint, then atomically replace `state.db`. Only vectors are stale — carry the existing `last_processed_history_id` forward rather than re-pinning a fresh boundary, so live changes during the rebuild are not skipped. No silent wrong vectors. |
| **Excluded-label config changed** | Startup compares `excluded_labels_hash`. Remove now-excluded labels from the index, bootstrap newly-included labels, and reuse unchanged embeddings where possible. If reconciliation is too complex, use the same two-phase rebuild path. |
| **History cursor expired** | Run a full sync/reconcile from Gmail: label registry, trainable label ids, labeled-message ids, skip sample, and `last_processed_history_id`. Do not fall back to inbox-only polling as the sole recovery. |
| **Explicit reset** | `make gcp-reset-state` / `make reset-state` — stop, `rm -f .../data/state.db`, start → next boot bootstraps fresh. The escape hatch "just in case." |

**Fingerprint:** a string or JSON object that includes at least:

```text
state_schema_version
embedding_model_name
embedding_dimension
textrepr-vN
excluded_labels_hash
classifier_params_hash
```

The model name comes from `Embedder`'s `model_name`; the `textrepr-vN` is a manual constant
bumped whenever `build_text_representation`/`preprocess_email_body` changes in a way that
alters embeddings. Stored in `meta` on bootstrap, checked on every startup.

**Design defaults chosen** (flag if you want the alternative):
- *One file, multiple small tables* (not separate vector/label files) — single atomic reset.
- *Two-phase auto-rebuild on fingerprint mismatch* (not "wipe first" and not "refuse to start
  + tell the user to reset") — avoids surprise downtime and preserves the last usable state
  until replacement succeeds; the explicit `reset-state` still exists for force.

## Files

- New `src/gmail_classifier/state_store.py` (or extend `embedding_cache.py`) — the
  `state.db` wrapper: `embeddings` + `labels` + `pending_new` + `meta` tables;
  `upsert_label`, `upsert_embedding`, `get_labels()`, `get_known_ids()`, `iter_index()`
  (join), `get/set_meta()`, `get/set_last_processed_history_id()`, `pending_new` helpers.
  One SQLite connection.
- New `src/gmail_classifier/bootstrap.py` — `bootstrap_index(client, embedder, store, ...)`,
  testable with fakes (no heavy imports), one-at-a-time fetch/embed/persist, resumable
  (skips ids already in `embeddings`). Folds in `fetcher.py`'s list/diff logic and the
  round-robin/front-loaded-skip ordering.
- `scripts/classify_and_label.py` — replace the `MessageStore.load_all` + `build_training_data`
  startup block with: **(a)** open `state.db`; **(b)** validate schema/config/ML fingerprint;
  **(c)** two-phase rebuild if needed; **(d)** if empty, call `watch()` first and bootstrap;
  **(e)** else load index from the join and process Gmail history from the persisted cursor.
  Drop `--training-db`/`--skip-db` defaults for a single `--state-db` path (keep old flags as
  overrides for local use).
- `pubsub.py` / `pubsub_loop.py` — change pull/ack ownership so messages are acknowledged only
  after history events are processed and `last_processed_history_id` is persisted. Preserve the
  existing "backlog across outage" behavior.
- `label_change_handler.py` — retarget persistence from `MessageStore.save_message` to
  `store.upsert_label` + `cache.put`; keep `index.add` untouched or move to `index.extend` when
  batching lands.
- `inbox_check.py` / `history_processor.py` — skip-pool write becomes `upsert_label(id, '__skip__')`
  only after maturity; before maturity, write `pending_new` instead. Inbox/history processors use
  `known_ids`, not just `skip_ids`, to avoid reprocessing labeled-but-still-INBOX messages.
- `training_index.py` — add `extend(...)` or a builder/snapshot path so bootstrap does not
  `np.vstack` one message at a time.
- `scripts/gcp-deploy.sh` — stop shipping `data/*.db`; ship code + credentials only.
  Verify the `--exclude='data'` + untar-over behavior preserves `state.db` across deploys
  (it does today). Optional `--seed-state` to upload a prebuilt `state.db` and skip first
  boot, but not required.
- `Makefile` — add `reset-state` (local: stop, `rm -f data/state.db`), `gcp-reset-state`
  (VM: stop, `rm -f $INSTALL_DIR/data/state.db`, start), and `gcp-state-status` (schema/
  fingerprint/bootstrap status/index size/per-label counts/skip count/pending count/history
  cursor).
- `README.md` — **needs a pass; several current claims become wrong.** Specifically:
  the GCP deploy steps and "Updating" section currently tell the user to run `make embed`
  and say `gcp-deploy` uploads the databases — both go away (deploy ships code + credentials
  only; the VM bootstraps from Gmail). The `# Deploy code, data, credentials` inline comment
  drops "data". Add: first-boot warm-up time, the auto-rebuild-on-ML-change behavior, and
  the `make gcp-reset-state` escape hatch. Quick-start's `make fetch-training`/`fetch-inbox`
  become optional (local-only) rather than prerequisites for deployment.

## Costs / wrinkles (accepted, but explicit)

1. **Slow first boot.** ~4331 messages = that many `get_message` calls + parse + serial
   embed. On the e2-micro, plausibly **10–20 min** for a fresh VM (parse alone was 327 s at
   this corpus size; serial embed adds more). First boot only; restarts are fast.
2. **Credentials still ship.** "Look at Gmail" needs the OAuth token + client secret. Deploy
   is code **+ `credentials/`**, not code alone. One small dir, not user training data.
3. **Model / text-representation changes force a re-fetch.** With no bodies persisted,
   changing the embedding model or `build_text_representation` invalidates the cache and
   requires re-bootstrapping. Consistent with "Gmail is truth," but a real consequence.
4. **Gmail API volume.** ~4–5k reads on first boot; well within daily quota, network-bound
   not quota-bound. Keep the existing `--max-per-label` cap to bound it.
5. **Local dev unaffected.** `make fetch-training`/`fetch-inbox` + the DB files can remain
   for local runs; this plan changes the *GCP/runtime* startup, not the local workflow. (Or
   converge local onto the same bootstrap later — out of scope here.)
6. **`state.db` is private derived data.** It has no bodies/subjects/senders, but it still
   contains message ids, label ids/names, and embeddings. Use `chmod 700 data/`,
   `chmod 600 data/state.db`, never upload `state.db` from the VM by default, and keep
   message ids out of normal logs.

## Verification

All unit tests drive fakes (fake Gmail client recording calls, fake embedder, in-memory
SQLite) — no network, no FastEmbed — mirroring the existing `test_pubsub_loop.py` /
`test_training.py` style. Grouped by the behavior each guards; the safety groups
(read-only boundary, maturity gate, history cursor, Pub/Sub ack ordering) are the highest
priority because they gate irreversible archive actions on a fresh mailbox.

### Unit — `state_store.py`
- `upsert_label` is **last-write-wins** on `message_id` (second upsert overwrites).
- `iter_index()` join yields only ids present in **both** `embeddings` and `labels`; an
  embedded id with no label row (or vice versa) is **excluded** — guards the orphaned-row
  class of bug structurally.
- `get_fingerprint`/`set_fingerprint` round-trip; a fresh store returns `None`.
- `get_last_processed_history_id`/`set_last_processed_history_id` round-trip.
- `known_ids` includes both real-label rows and `__skip__` rows.
- empty store reports empty (drives the "fresh VM → bootstrap" branch).
- `pending_new` insert/drain is idempotent.

### Unit — `bootstrap.py`
- Builds the right index from a fake client/embedder (ids → vectors → labels as expected).
- **Resumability:** a second run **skips ids already in `embeddings`** — assert
  `get_message` is *not* called for cached ids (not merely that the result is the same).
- Excludes XL* labels at the source (no excluded-label rows reach `labels`).
- Stores/votes by Gmail `label_id`, with `label_name_snapshot` used only for display/status.
- **One-at-a-time:** the raw message for id *i* is released before id *i+1* is fetched —
  assert no `embed_batch`/bulk path and at most one live body held (e.g. fake records max
  concurrent un-discarded messages == 1).
- **Round-robin ordering:** with labels A/B/C each having ≥R messages, the order of
  `cache.put` cycles A,B,C,A,B,C… (not A,A,…,B,B,…) — so after R rounds every label has ~R
  examples and crosses the ≥5 eligibility line (`classifier.py:98`) together, not serially.
- **Skip pool front-loaded:** the first ~50 persisted rows are skip seeds, *before* the
  round-robin proper begins.
- **Labeled-wins-over-skip at source:** an INBOX id that also carries a user label is
  recorded with that **label**, never `__skip__` — `labels` never holds `__skip__` for it.

### Unit — startup dispatch (schema/config/fingerprint)
- **Match** → load index from the join, `client.get_message` **never called** (no fetch).
- **ML mismatch** → build `state.rebuild.db`, validate it, then atomically swap; crash during
  rebuild leaves old `state.db` untouched. The rebuilt store **carries the prior
  `last_processed_history_id` forward** (assert the cursor is preserved, not re-pinned to a
  fresh watch id).
- **Empty store** → call `watch()` first, persist boundary/cursor, bootstrap runs (fingerprint
  written on completion).
- **Excluded-label mismatch** → now-excluded rows are removed or a two-phase rebuild is
  triggered; newly-included labels are bootstrapped.

### Unit — read-only boundary (cold path safety)
- Bootstrap **never calls `apply_label`/archive** on existing inbox mail — assert
  `client.apply_label` is not called during bootstrap, even for messages that would
  classify with high confidence. (Guards against archiving the pre-existing backlog — the
  current `_check_inbox` labeling behavior must be removed from the cold path.)
- `client.watch()` is invoked **before the first `get_message`** (ordering assertion), so
  the pinned `historyId` boundary reflects start-of-service, not end-of-bootstrap.
- A notification whose `historyId` is **at-or-before** the pinned boundary is treated as
  existing (not labeled); one **after** it is eligible (subject to the maturity gate).
- Warm restart processes history from persisted `last_processed_history_id`; it does not
  replace that cursor with the new watch id and thereby skip backlog.
- Warm restart runs **no labeling inbox check** — assert `apply_label`/archive is reached only
  via history replay, never via an inbox scan that could touch pre-boundary mail.

### Unit — maturity gate
- Below threshold (**label target not met OR skip pool not yet loaded**), a new post-boundary
  message is **not** labeled/archived regardless of confidence.
- Below threshold, the message is inserted into `pending_new`, not `labels('__skip__')`.
- The gate requires **both** conditions: a mature label set with the skip pool *not* loaded
  still blocks labeling (guards the spurious-high-confidence over-labeling described in
  "Two gates").
- Labels with fewer than `MIN_EXAMPLES_PER_LABEL` available examples do not block maturity
  forever and remain ineligible to win.
- Once finite maturity targets are met, a high-confidence new message **is** labeled.
- When maturity opens, `pending_new` is drained through normal classification and removed
  idempotently.

### Unit — Pub/Sub / history cursor safety
- Pub/Sub notification is not acknowledged until events are processed and
  `last_processed_history_id` is persisted.
- Crash after processing but before ack is safe: redelivery replays history idempotently.
- Crash before processing is safe: persisted cursor causes replay on restart.
- History expiration/404 schedules full sync/reconcile, not inbox-only fallback.
- The Gmail watch is not filtered to INBOX; label-add/remove events outside INBOX still reach
  the service.

### Unit — progressive interleave
- A pending notification is **serviced between bootstrap batches**, not after the whole
  corpus — drive the loop with a fake where a notification arrives mid-bootstrap and assert
  it is processed before bootstrap completes (proves the single-threaded interleave, no
  starvation).
- Bootstrap batches obey max message/time budgets.
- `TrainingIndex.extend` or builder snapshots avoid one `np.vstack` per message.

### Memory / Behavior / Deploy (as before)
- Memory: instrumented bootstrap on the VM shows peak materially below the old ~606 MB (no
  +447 MB corpus load); steady-state ~220 MB unchanged. (Idle RSS already held flat by the
  shipped idle-trim fix `be24c59`.)
- Behavior: a correction still shifts a prediction (existing live-adaptation guarantee);
  full suite green.
- Deploy: on a freshly created VM, `gcp-create → gcp-deploy (code+creds) → gcp-start`
  produces a working classifier with **no DB upload**; a redeploy (code change, ML/config
  unchanged) **preserves `data/state.db`** across the `--exclude='data'` untar-over; restart
  is fast (reads derived store, no full fetch).
- Status: `make gcp-state-status` reports schema version, ML fingerprint, bootstrap status,
  index size, per-label counts, skip count, pending count, and last processed history id.

## Relationship to prior plans

- **Supersedes** the lightweight-classifier direction (`optimized-purring-globe.md` Phases
  1–3 and `classy_gcp_memory_classification_plan.md`) *for the GCP goal*: once the transient
  is gone and steady-state is ~220 MB, there's no memory case left for replacing the
  semantic classifier. Quality stays at the semantic baseline (no degradation).
- **Builds on** the Phase 0 findings and the startup fixes already shipped (`cf4ee18`,
  `1616593`, `fd0b6d6`, `80845fa`).
