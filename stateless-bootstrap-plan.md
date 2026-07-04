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

## Two independent backends (coexistence invariant)

This is delivered as a **second storage backend alongside the existing one**, not a rewrite of
it. The two must be fully isolated so either can be deployed and the other resumed later with
no cleanup:

- **`legacy` backend** — today's three files: `data/training.db`, `data/inbox_sample.db`,
  `data/embeddings.db`. Bodies persisted. Unchanged by this plan.
- **`state` backend** — this plan's single `data/state.db` (+ `data/state.rebuild.db` during a
  rebuild). Derived-only, no bodies.

Hard invariants:

1. **Disjoint files.** The two file sets never overlap by name. The `state` backend reads and
   writes only `state.db`/`state.rebuild.db` and their SQLite sidecars; it **never opens**
   `training.db`, `inbox_sample.db`, or `embeddings.db` — not to read, migrate, or delete. The
   `legacy` backend never opens `state.db`. (There is deliberately **no migration path** between
   them:
   the `state` backend's behavior does not depend in any way on whether the legacy files are
   present, and it never reuses or converts their contents.)
2. **Behavior independent of the other's presence.** A `state`-backend boot on a VM that still
   has all three legacy files behaves *identically* to a boot on a clean VM. The only signal it
   reads is its own `state.db`. Likewise for `legacy`.
3. **Deploy selects, never destroys.** Deploying one backend sets which one the service starts
   with; it does **not** remove or rewrite the other's data files. Switching back is: redeploy
   the other backend (or flip the config) and restart — the other's database contents are as they
   were left. Deploy may still normalize owner/mode so the service can read its files; that is
   not a backend state change.
4. **Reset is backend-scoped.** `reset-state` touches only `state.db`/`state.rebuild.db` and
   their SQLite sidecars; it never deletes legacy files, and there is no combined "reset
   everything" that crosses the boundary.

The backend is chosen at startup by an explicit selector (see "Backend selection"), defaulting
to `legacy` so an unflagged/again-old deploy is byte-for-byte today's behavior. Backend switches
are exclusive: the deploy path stops the current service, rewrites the selector, and starts one
classifier process. Running `legacy` and `state` services concurrently against the same mailbox is
not supported.

### Switching backends: what each side sees on restart

Because Gmail is the source of truth and the two backends share no files, a switch is safe but
**stale** — the incoming backend runs whatever it last persisted, ignorant of what the other did
in between. The one non-obvious hazard is the history cursor.

**Legacy restarting after `state` ran (and archived new mail out of the inbox):**

Legacy only ever acts on what is **currently in the INBOX** (`inbox_check.py:49`), and for each
such message it treats *labeled* mail as settled truth and *unlabeled* mail as work
(`inbox_check.py:57-59`):
- **Mail `state` labeled + archived** — out of the inbox (and labeled). Legacy never lists it,
  and even if it resurfaced, the user label makes legacy **skip it as truth**. Left exactly as
  `state` set it. This is the "existing = truth, not to-be-classified" behavior you'd expect.
- **Mail `state` left unlabeled in the inbox** (low confidence, below maturity, or pre-boundary)
  — still in the inbox with no user label, so legacy **re-classifies it** with its own model and
  may label it. Legacy makes no distinction between "arrived during the state era" and "always
  been here": any unlabeled inbox message is work. (`state`'s *first* boot, by contrast, is
  read-only over the pre-existing mailbox and leaves such mail alone.) So switching to legacy
  labels the inbox backlog.
- **This is not a switching-specific effect.** It is identical to restarting legacy after *any*
  downtime: legacy always re-`watch()`es fresh and sweeps the current INBOX, classifying whatever
  unlabeled mail accumulated while it was down. Whether the gap was filled by `state` running, or
  by nothing running at all, legacy sees the same thing — a current INBOX — and behaves the same.
  `state` running in the interim changes nothing about it.
- Legacy runs with its **frozen** `training.db`/`inbox_sample.db` (as of its last fetch) — none
  of the examples `state` learned. Expected staleness; run `make fetch-training` to refresh.
- A later `labelsRemoved` on a message `state` handled but legacy never stored is safe: the
  delete-from-training is a no-op for an unknown id.

**Why the seam must keep legacy's cursor non-durable:** legacy today re-`watch()`es from a
**fresh** `historyId` every boot and keeps **no durable history cursor** (`_run_pubsub_mode`) —
it never replays past history, only watches forward from start-of-service. The `StorageBackend`
interface exposes `get/set_last_processed_history_id()`; the legacy adapter must implement these
as **process-local only**: the current loop may remember a cursor in memory, but nothing is
written to disk and a fresh legacy adapter starts with no cursor. This preserves the fresh-watch
behavior byte-for-byte. This is not about preventing corruption — a replay would mostly be
redundant work (and often just hits `HistoryExpiredError` → inbox-poll fallback anyway) — it is
about legacy staying *exactly* today's behavior after the refactor. Only the `state` backend
persists a cursor, in its own `state.db`.

**`state` restarting after legacy ran:** `state` replays history from its own persisted cursor,
which now spans the legacy period, so it ingests the labels legacy applied (including legacy's
*automated* labels) as training truth. This is **consistent with the existing model** — Gmail
labels are authoritative and `make fetch-training` already folds the classifier's own auto-labels
into the training set — so it is accepted, not a new defect. (If the cursor has aged out, the
"history cursor expired" path does a full reconcile instead.)

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
bootstrap_status              # absent | in_progress | complete
bootstrap_boundary_history_id
last_processed_history_id
watch_expiration_ms
bootstrap_started_at
bootstrap_completed_at
```

`bootstrap_status` drives the cold/warm branch and the progressive-vs-normal loop variant:
`absent` (fresh VM) → cold bootstrap; `in_progress` (crashed mid-bootstrap) → resume the
progressive path; `complete` → warm path.

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
     - For the swap to be truly atomic: build `state.rebuild.db` **in `data/`** (same
       filesystem as `state.db`, so `os.replace` is a rename, not a copy); **close the SQLite
       connections to both files before renaming**; and move the WAL sidecars too, or
       checkpoint+`journal_mode=DELETE` before the swap so no `-wal`/`-shm` is orphaned.
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

This read-only property is scoped to the **cold first boot** — the mailbox that pre-existed the
service's very first start. It is *not* a permanent "state never touches the backlog": a **warm
restart** after downtime does catch up on the gap, but via history replay from the persisted
`last_processed_history_id` (step 4), not an inbox scan. That replay is post-cursor only, so it
classifies mail that arrived during the downtime while still never touching pre-first-boot mail;
if the cursor has aged out of Gmail's history window, it falls back to a full reconcile.

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

For index growth, avoid one `np.vstack` per bootstrapped message. Use
`TrainingIndex.add_many(items)` (already implemented, `4cf66a8`), which appends a whole
round-robin batch of `(id, vector, label)` tuples with a single reallocation, then classify
against the grown index between batches.

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
idempotent. If it crashes after ack on the `state` backend, the durable cursor has already moved.
On the `legacy` backend, a shared implementation may ack after processing, but the cursor remains
non-durable by design so a restart keeps today's fresh-watch behavior.

Note the crash-safety guarantee here (crash-before-ack → idempotent history replay) is
**state-only**. Legacy does not inherit it: after a restart legacy fresh-`watch()`es and does not
persist the pre-crash history cursor. Pub/Sub redelivery alone is therefore not relied on to replay
pre-crash history. Legacy's safety net is instead its startup inbox scan, which re-examines whatever
is currently unlabeled in the INBOX — the same catch-up-after-downtime behavior described in
"Switching backends." This is today's legacy behavior, unchanged.

### Live adaptation (preserve today's behavior)

`label_change_handler.process_label_changes` currently writes bodies to
`training_store`/`skip_store` *and* updates the in-memory index
(`label_change_handler.py:109,117,139,144`). Under the new model the in-memory `index.add`
stays; the persistence target changes from "save body to MessageStore" to "upsert
`id→label_id` + `cache.put(id, vec)` in the derived store." Same learn-on-correction behavior,
no bodies persisted. At the `StorageBackend` seam, callers still pass the parsed `Message` plus
its vector: the legacy adapter uses the `Message` to perform today's body-preserving
`save_message`, while the state adapter stores only `message.id`, `label_id`, and `vec`, then
forgets the raw content. The `index.add(...)` calls are untouched; a batch correction can use
`index.add_many(...)` (`4cf66a8`) where several updates land together.

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
| **Code deploy (state backend), ML/config unchanged** | `gcp-deploy-state` builds the tarball with `--exclude='data'` and untars *over* `INSTALL_DIR`; `tar x` never deletes files absent from the archive, so `data/state.db` (and any leftover legacy DBs) is untouched. Deploy + restart → fast, state preserved. |
| **ML changed** (embedding model or `build_text_representation`) | Startup compares `meta.ml_fingerprint` to the current code. Mismatch ⇒ cached vectors are stale. Build `state.rebuild.db` from Gmail, validate counts/fingerprint, then atomically replace `state.db`. Only vectors are stale — carry the existing `last_processed_history_id` forward rather than re-pinning a fresh boundary, so live changes during the rebuild are not skipped. No silent wrong vectors. |
| **Excluded-label config changed** | Startup compares `excluded_labels_hash` (checked independently of the ML fingerprint). Reuse all unchanged embeddings: remove now-excluded labels from the index and bootstrap only the newly-included labels' ids — no re-embed of retained ids. The two-phase rebuild path is a last-resort fallback only if incremental reconciliation proves too complex to implement safely, not the expected path. |
| **History cursor expired** | Run a full sync/reconcile from Gmail: label registry, trainable label ids, labeled-message ids, skip sample, and `last_processed_history_id`. Do not fall back to inbox-only polling as the sole recovery. |
| **Explicit reset** | `make gcp-reset-state` / `make reset-state` — stop, remove `state.db` plus SQLite sidecars (`state.db-wal`, `state.db-shm`, `state.rebuild.db*`), start → next boot bootstraps fresh. Legacy DBs are untouched. The escape hatch "just in case." |

**Fingerprint:** a string or JSON object that includes at least:

```text
state_schema_version
embedding_model_name
embedding_dimension
textrepr-vN
classifier_params_hash
```

The model name comes from `Embedder`'s `model_name`; the `textrepr-vN` is a manual constant
bumped whenever `build_text_representation`/`preprocess_email_body` changes in a way that
alters embeddings. Stored in `meta` on bootstrap, checked on every startup.

**`excluded_labels_hash` is deliberately *not* in the fingerprint.** The fingerprint governs
*vector validity* (is a cached `id→vector` still correct?); `excluded_labels_hash` governs
*membership* (which rows participate in the index). They are orthogonal: changing the excluded
set does not invalidate a single vector, only which rows are loaded — so it must take the cheap
reconcile path (remove now-excluded, bootstrap newly-included, reuse unchanged embeddings), not
the full vector rebuild. Folding it into the fingerprint would make that reconcile path dead
code and force a needless ~4000-message re-embed on a pure config change. It is checked
independently on startup as its own `meta` key.

**Design defaults chosen** (flag if you want the alternative):
- *One file, multiple small tables* (not separate vector/label files) — single atomic reset.
- *Two-phase auto-rebuild on fingerprint mismatch* (not "wipe first" and not "refuse to start
  + tell the user to reset") — avoids surprise downtime and preserves the last usable state
  until replacement succeeds; the explicit `reset-state` still exists for force.

## Code organization (common core, backend-specific edges)

The goal is that both backends share one runtime and one classification path, differing only in
*how the index is loaded/persisted*. Extract the shared logic; keep each backend's storage code
in its own file so neither imports the other.

**Common (backend-agnostic), unchanged or lightly generalized:**
- `classifier.py`, `training_index.py`, `embeddings.py`, `preprocessing.py`,
  `label_registry.py`, `gmail_client.py`, `gmail_parser.py`, `models.py` — pure runtime, no
  storage assumptions. No changes beyond what's already noted (`add_many` already landed).
- `pubsub_loop.py` — the notification/history/ack loop is shared by both backends. It acks only
  after successful event processing. On `state`, that includes a durable cursor write; on
  `legacy`, the cursor write remains intentionally non-durable, so this improves ordering
  without changing legacy's fresh-watch restart semantics.
- The classify-and-apply logic in `inbox_check.py` / `history_processor.py`. Their *only*
  backend dependency is where a skip/label write goes and what set of `known_ids` they consult.

**The seam — a `StorageBackend` protocol.** Define a narrow interface the runtime talks to, so
`classify_and_label.py` and the handlers never name a concrete store:

```text
StorageBackend:
    load_index() -> TrainingIndex            # legacy: load_all+build; state: join
    known_ids() -> set[str]
    skip_vote_ids() -> set[str]
    upsert_label(message, label_id, vec) / upsert_skip(message, vec)
    get/set_last_processed_history_id() # legacy: process-local only; never persisted
    # state-only extras (put_embedding, pending_new, meta, fingerprint) live behind capability
    #   checks or on the state subclass; legacy no-ops / raises NotImplementedError for them.
    put_embedding(id, vec)              # state-only direct cache write during bootstrap/rebuild
                                        # (state_store.py names it upsert_embedding); legacy no-ops
```

**Cursor durability is backend-specific.** Only `state` persists the history cursor (in
`state.db`). The legacy adapter may keep a process-local cursor for the running loop, but
`set_last_processed_history_id` never writes to disk, so legacy continues to fresh-`watch()`
every boot — preserving today's behavior byte-for-byte. See "Switching backends" for the full
rationale.

**Backend-specific files:**
- `storage_legacy.py` — thin adapter wrapping today's `MessageStore` + `EmbeddingCache` over
  the three files. This is a *move/wrap* of existing code, not a rewrite; behavior identical.
- `state_store.py` + `bootstrap.py` — the new backend (this plan). Opens **only**
  `state.db`/`state.rebuild.db`.

**Backend selection (explicit, defaults to legacy):** `classify_and_label.py` picks the backend
from a single `--storage {legacy,state}` flag (env `CLASSY_STORAGE`), defaulting to `legacy`.
The selector is the *only* branch; downstream code sees just a `StorageBackend`. An unflagged
run is exactly today's behavior. The progressive-bootstrap / maturity / read-only-boundary logic
is reached **only** on the `state` path; the `legacy` path keeps its current startup verbatim.

## Deployment: two backends, one deployed at a time

Deploy chooses the backend and writes it into the service configuration; it never uploads,
deletes, or rewrites the other backend's database contents.

- `gcp-deploy-legacy` — today's deploy: ships code (+ the three DBs, as now) and sets the
  service to start with `--storage legacy`. Unchanged from current `gcp-deploy` (keep
  `gcp-deploy` as an alias for it so existing muscle memory/scripts are safe).
- `gcp-deploy-state` — ships **code + credentials only** (`--exclude='data'`) and sets the
  service to start with `--storage state`. Its preflight guards require credentials, but not
  `training.db`, `inbox_sample.db`, or `embeddings.db`. Does not upload or delete any
  `data/*.db`.
- The backend is persisted in the systemd unit / runner (an added flag or `CLASSY_STORAGE=` in
  the service's environment), rewritten on each deploy, so a plain `gcp-restart` keeps whatever
  backend was last deployed.
- **Switching back is a redeploy, not a cleanup:** `gcp-deploy-legacy` after running `state`
  finds the legacy DB contents exactly as they were left (the `state` backend never opened
  them) and starts against them; `state.db` remains on disk, ignored. And vice-versa. Neither
  deploy removes or rewrites the other's database contents — per coexistence invariant #3.

## Files

- New `src/gmail_classifier/state_store.py` — the `state.db` wrapper (a **new file**, not an
  extension of `embedding_cache.py`, which belongs to the legacy backend and must stay
  untouched): `embeddings` + `labels` + `pending_new` + `meta` tables; `upsert_label`,
  `upsert_embedding`, `get_labels()`, `get_known_ids()`, `iter_index()` (join), `get/set_meta()`,
  `get/set_last_processed_history_id()`, `pending_new` helpers. Opens only
  `state.db`/`state.rebuild.db`. One SQLite connection. May *reuse* `EmbeddingCache`'s
  vector-blob (de)serialization by importing the helper, but writes to its own file. Treat SQLite
  sidecars (`state.db-wal`, `state.db-shm`, and rebuild equivalents) as part of the state
  backend's file set.
- New `src/gmail_classifier/storage_backend.py` — the `StorageBackend` protocol (the seam
  above) plus `storage_legacy.py` adapting the existing `MessageStore`+`EmbeddingCache`. The
  `state_store`/`bootstrap` pair is the other implementation.
- New `src/gmail_classifier/bootstrap.py` — `bootstrap_index(client, embedder, store, ...)`,
  testable with fakes (no heavy imports), one-at-a-time fetch/embed/persist, resumable
  (skips ids already in `embeddings`). Folds in `fetcher.py`'s list/diff logic and the
  round-robin/front-loaded-skip ordering.
- `scripts/classify_and_label.py` — add a `--storage {legacy,state}` selector (env
  `CLASSY_STORAGE`, default `legacy`) that constructs the chosen `StorageBackend`; the rest of
  `main` talks only to that interface. **Keep the current `MessageStore.load_all` +
  `build_training_data` startup as the `legacy` path, verbatim** — it is not replaced, only
  moved behind the selector. The `state` path is new: **(a)** open `state.db`; **(b)** validate
  schema/config/ML fingerprint; **(c)** two-phase rebuild if needed; **(d)** if empty, call
  `watch()` first and bootstrap; **(e)** else load index from the join and process Gmail history
  from the persisted cursor. Legacy keeps `--training-db`/`--skip-db`; state uses `--state-db`.
- `pubsub.py` / `pubsub_loop.py` — change pull/ack ownership so messages are acknowledged only
  after history events are processed and `last_processed_history_id` is persisted. `pull()` returns
  notifications plus ack ids; a separate `ack()` acknowledges them after successful processing.
  Preserve the existing "backlog across outage" behavior.
- `label_change_handler.py` — persist through the `StorageBackend` seam instead of naming
  `MessageStore`: `backend.upsert_label(message, label_id, vec)` /
  `backend.upsert_skip(message, vec)`. On the legacy backend the adapter does today's
  `save_message`; on state it does the label+vector upsert and discards the body. `index.add`/
  `add_many` untouched (`4cf66a8`).
- `inbox_check.py` / `history_processor.py` — consult `backend.known_ids()` (not just
  `skip_ids`) so labeled-but-still-INBOX messages aren't reprocessed. On the state backend the
  skip-pool write becomes `upsert_skip(message, vec)` only after maturity, and `pending_new`
  before maturity; on legacy these map to the current empty-label `save_message`. The maturity /
  `pending_new` behavior is state-only and is a no-op on legacy.
- `training_index.py` — no change needed: `add_many(...)` (`4cf66a8`) already appends a batch
  with a single `np.vstack`, so bootstrap round-robin batches use it directly.
- `scripts/gcp-deploy.sh` — parameterize by backend (driven by the `gcp-deploy-legacy` /
  `gcp-deploy-state` targets). Guards are backend-specific: legacy still requires the three DBs;
  state requires only credentials. Legacy path is today's behavior (ships the three DBs), but the
  sync list must stay explicit (`training.db`, `inbox_sample.db`, optional `embeddings.db`), not
  `data/*.db`, so it cannot overwrite `state.db`. State path ships **code + credentials only**
  (`--exclude='data'`) and writes `--storage state` into the service config. Both rely on the
  `--exclude='data'` + untar-over behavior, which never deletes files absent from the archive —
  so **neither deploy removes or rewrites the other backend's database contents**. Optional
  `--seed-state` to upload a prebuilt `state.db` and skip first boot, but not required and never
  implicit.
- `Makefile` — add `reset-state` (local: stop, `rm -f data/state.db data/state.db-wal
  data/state.db-shm data/state.rebuild.db data/state.rebuild.db-wal data/state.rebuild.db-shm`),
  `gcp-reset-state` (same removal under `$INSTALL_DIR/data`, then start), and `gcp-state-status`
  (active backend, schema/fingerprint/bootstrap status/index size/per-label counts/skip count/
  pending count/history cursor). See "Deployment: two backends, one deployed at a time" for the
  backend-selection targets (`gcp-deploy-legacy` / `gcp-deploy-state`).
- `README.md` — **needs a pass to document both backends.** Keep the current legacy deploy
  instructions (`make embed`, DB upload) but scope them to `gcp-deploy-legacy`; make clear that
  unqualified local commands, the macOS service, and `gcp-deploy` remain legacy by default. Add a
  `state` backend section: `gcp-deploy-state` ships code + credentials only and the VM bootstraps
  from Gmail; document first-boot warm-up time, auto-rebuild-on-ML-change, `make
  gcp-reset-state`, and that switching backends is a redeploy that leaves the other's data intact.
  `make fetch-training`/`fetch-inbox` remain required for legacy, optional (local-only) for state.

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
5. **Local dev unaffected by default.** `make fetch-training`/`fetch-inbox` + the DB files can
   remain for local runs; unqualified local targets and the macOS service continue to use
   `legacy` unless `--storage state` / `CLASSY_STORAGE=state` is explicitly set. (Or converge
   local onto the same bootstrap later — out of scope here.)
6. **`state.db` is private derived data.** It has no bodies/subjects/senders, but it still
   contains message ids, label ids/names, and embeddings. Use `chmod 700 data/`,
   `chmod 600 data/state.db`, never upload `state.db` from the VM by default, and keep
   message ids out of normal logs.
7. **Skip pool is never refreshed (accepted).** `__skip__` vectors are seeded once at bootstrap
   and persist indefinitely. As sampled inbox mail is later archived or deleted in Gmail, those
   rows linger — semantically harmless (an old "inbox-like" example is still a valid negative)
   but unbounded-in-staleness. Accepted for now; a periodic resample of the skip pool is a
   possible later refinement, not required for correctness.

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

### Unit — backend isolation (coexistence invariant)
- **State backend opens only its own files:** run the full `state` startup + bootstrap against
  a fake/temp `data/` and assert `training.db`/`inbox_sample.db`/`embeddings.db` are never
  opened (e.g. sentinel files with a mtime that must not change, or an open() spy) — no read,
  no write, no delete.
- **Presence-independence:** a `state` boot with all three legacy files present produces the
  same index and same calls as a `state` boot on an empty `data/` — the legacy files are
  ignored entirely (no migration).
- **Legacy backend unchanged:** the `legacy` path never opens `state.db`; its startup calls and
  index match the pre-plan baseline (regression guard on the moved-behind-selector code).
- **Selector default:** no `--storage` flag / no `CLASSY_STORAGE` → `legacy`.
- **Storage seam preserves legacy needs:** the legacy adapter receives a full `Message` and saves
  the body exactly as today's `MessageStore` path does; the state adapter persists only id/label/
  vector and stores no body/subject/sender.
- **State deploy guard scope:** `gcp-deploy-state` succeeds with credentials present and no
  legacy DB files; `gcp-deploy-legacy` still fails fast when its required DBs are absent.
- **`reset-state` scope:** deletes only `state.db`/`state.db-wal`/`state.db-shm` and
  `state.rebuild.db*`; legacy files untouched.

### Unit — switching backends (cross-restart safety)
- **Legacy cursor stays non-durable:** the legacy adapter may advance a cursor in memory inside
  one running loop, but `set_last_processed_history_id(x)` followed by a fresh adapter's
  `get_last_processed_history_id()` returns `None` (or the fresh-watch sentinel) — legacy persists
  no cursor, so it keeps watching forward from start-of-service rather than replaying, exactly as
  today.
- **Legacy after state — labeled mail treated as truth:** for an id `state` labeled+archived,
  legacy's inbox check either never lists it (not in INBOX) or, if it appears in INBOX with a
  user label, **skips it** — never re-classifies or re-archives it.
- **Legacy after state — unlabeled inbox mail is re-classified:** an unlabeled inbox message that
  `state` left alone **is** run through legacy's classifier (confirms the accepted behavior
  difference: legacy re-enables backlog labeling, `state` boots read-only).
- **Legacy after state — stale `labelsRemoved` is a no-op:** a `labelsRemoved` for an id legacy
  never stored does not crash and does not corrupt the stores (delete-from-training no-ops).
- **State after legacy — cursor replay ingests legacy labels:** `state` resumes from its
  persisted cursor across a window in which labels were applied; those labels are folded into the
  index as training truth (asserts the accepted "Gmail is truth" behavior, and that an expired
  cursor instead triggers full reconcile).
- **No concurrent workers:** switching backends stops the existing service and rewrites the
  selector before restart; tests/VM smoke check verify there is only one `gmail-classifier`
  process and one backend setting in the systemd unit.

### Unit — startup dispatch (schema/config/fingerprint)
- **Match** → load index from the join, `client.get_message` **never called** (no fetch).
- **ML mismatch** → build `state.rebuild.db`, validate it, then atomically swap; crash during
  rebuild leaves old `state.db` untouched. The rebuilt store **carries the prior
  `last_processed_history_id` forward** (assert the cursor is preserved, not re-pinned to a
  fresh watch id).
- **Empty store** → call `watch()` first, persist boundary/cursor, bootstrap runs (fingerprint
  written on completion).
- **Excluded-label mismatch** → now-excluded rows are removed or a two-phase rebuild is
  triggered; newly-included labels are bootstrapped. Assert this takes the cheap reconcile
  path (no full re-embed) since `excluded_labels_hash` is not in the fingerprint — an
  exclusion-only change **does not** call `get_message`/`embed` for unchanged ids.

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
- Bootstrap grows the index via `add_many` per batch (single `np.vstack`), not one append per
  message — already covered by `test_training_index_batch.py` (`4cf66a8`).

### Memory / Behavior / Deploy (as before)
- Memory: instrumented bootstrap on the VM shows peak materially below the old ~606 MB (no
  +447 MB corpus load); steady-state ~220 MB unchanged. (Idle RSS already held flat by the
  shipped idle-trim fix `be24c59`.) Note: the `[mem]`/`log_mem` startup probes were removed in
  `0a8ef3d`, and the surviving per-message `log_prefix` RSS does **not** sample the startup
  transient — so verifying the "no +447 MB" claim requires temporarily re-adding a peak probe
  (or reading RSS once right after bootstrap completes).
- Behavior: a correction still shifts a prediction (existing live-adaptation guarantee);
  full suite green.
- Deploy (state): on a freshly created VM, `gcp-create → gcp-deploy-state → gcp-start`
  produces a working classifier with **no DB upload**; a redeploy (code change, ML/config
  unchanged) **preserves `data/state.db`** across the `--exclude='data'` untar-over; restart
  is fast (reads derived store, no full fetch).
- Deploy (switch-back, on a real VM): after running the `state` backend, `gcp-deploy-legacy`
  finds the three legacy DB contents exactly as left (unmodified by the `state` run) and starts
  against them; `state.db` remains on disk, ignored. Then `gcp-deploy-state` again starts against
  the preserved `state.db` with no re-bootstrap. Confirms deploy selects, never destroys.
- Deploy sync lists are explicit: legacy deploy cannot upload/overwrite `state.db`, and state
  deploy cannot upload/overwrite `training.db`, `inbox_sample.db`, or `embeddings.db` except via
  an explicit legacy deploy.
- Status: `make gcp-state-status` reports active backend, schema version, ML fingerprint,
  bootstrap status, index size, per-label counts, skip count, pending count, and last processed
  history id.

## Relationship to prior plans

- **Supersedes** the lightweight-classifier direction (`optimized-purring-globe.md` Phases
  1–3 and `classy_gcp_memory_classification_plan.md`) *for the GCP goal*: once the transient
  is gone and steady-state is ~220 MB, there's no memory case left for replacing the
  semantic classifier. Quality stays at the semantic baseline (no degradation).
- **Builds on** the Phase 0 findings and the startup fixes already shipped (`cf4ee18`,
  `1616593`, `fd0b6d6`, `80845fa`).
