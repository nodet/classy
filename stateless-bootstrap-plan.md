# Plan: Stateless GCP deployment — bootstrap from Gmail, persist only derived state

Date: 2026-06-28
Status: proposed (supersedes the lightweight-classifier direction for the GCP goal)

## Goal

Treat **Gmail as the single source of truth**. Make a fresh GCP deploy require only
*code + credentials* — no locally-built `training.db` / `inbox_sample.db` / `embeddings.db`
to upload. Trade a slow first boot for statelessness and trivial deployment:

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
in `training.db` only to be re-embeddable. A stateless design never persists bodies.

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

### Persisted state (derived, no bodies) — one file, `data/state.db`

A single SQLite file replaces all three of today's DBs. The runtime
`TrainingIndex` is just the join `embeddings ⋈ labels on message_id`. **No
message bodies/subjects/senders are stored anywhere on the VM.**

| Table | Columns | Role |
|---|---|---|
| `embeddings` | `message_id, vector` | the vector cache (exists today) |
| `labels` | `message_id, label` | `label` = user label, or `__skip__` for the skip pool |
| `meta` | `key, value` | one row: **ML fingerprint** (embedding model name + text-representation version) |

(Name `state.db` chosen over keeping `embeddings.db` since it now holds more than
embeddings; one file = one connection, atomic, trivial to reset.)

### Startup logic (`scripts/classify_and_label.py:main`)

1. Open the derived store. If it already has a usable index (`embeddings + id→label`),
   load it and go straight to the loop — **fast restart, no Gmail fetch**.
2. If empty (fresh VM) → **bootstrap from Gmail**:
   - `list_user_labels()` minus excluded (XLC/XLE/XLCap).
   - For each label: `list_message_ids(label_id, max_results=--max-per-label)`.
   - For the skip pool: list recent INBOX ids, **minus any id that already carries a
     user label** (see "Labeled wins over skip" below).
   - For each id **not already embedded**: `get_message` → parse → `build_text_representation`
     → `embedder.embed` → `cache.put(id, vec)` + record `id→label` (or skip). Discard the
     raw message. **One at a time** — bounded memory, resumable.
   - Build `TrainingIndex` from the cache + label map.

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
3. Either path → the pubsub loop. The **warm** path enters it unchanged. The **cold**
   (bootstrap) path enters a progressive variant — see "Progressive bootstrap" below —
   so the service is live and safe from the first second rather than after a 20-min wait.

### Resumability (matters during the slow first boot)

Because each vector + label row is committed as computed, a crash at minute 15 of a 20-min
bootstrap **resumes** (step 2 skips already-embedded ids) rather than restarting. This is
why per-message caching (commit `fd0b6d6`) was worth doing.

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
inbox check is **removed from the cold path** (the warm-restart fast path, which has a cache,
keeps today's behavior).

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
  how mature the model becomes. Permanent, per-message, decided by the pinned `historyId`.
- **Maturity gate**: even genuinely-new mail is not labeled until the index is broad enough —
  **≥~20 examples per user label AND the skip pool loaded**. Confidence is
  `winning_score / total_score` with `__skip__` neighbors in the denominator
  (`classifier.py:73,137`); without the skip mass loaded, early confidence is spuriously high
  and the service **over-labels**. Since the live path applies *and archives* at ≥0.80
  (`inbox_check.py:83,93`), an early mistake is a semi-irreversible action on the mailbox — so
  the gate is conservative by design.

Consequence (accepted): new mail arriving during early warmup, before the maturity gate opens,
stays **unlabeled in the inbox** — not lost, just not acted on yet, and a natural skip/recheck
candidate later. The correct conservative default for a brand-new service.

#### Single-threaded interleave (not a background thread)

Do the bootstrap *in* the pubsub loop, not a side thread. `TrainingIndex.add` reassigns
`self.embeddings` via `np.vstack` and mutates a list + dict (`training_index.py:23-35`); a
concurrent `classify` reading `self.embeddings` mid-`vstack` races on a half-built array, and
two threads would share one FastEmbed model and compete for the e2-micro's single core. Instead,
process one round-robin batch *between* `run_iteration` calls: a batch (one message per label +
a few skip), then service any pending notification, repeat until the corpus is exhausted. This
is single-threaded (no lock, no index race, one embedder caller) and naturally throttled (live
mail preempts bootstrap between batches). Bootstrap finishes somewhat later in wall-clock — the
tradeoff we already accept, since the goal is early responsiveness, not fast completion.

#### First-boot summary

`watch()` → pin `historyId` → load ~50 skip → round-robin labels + inbox, committing each
vector (resumable) → once **≥20/label AND skip pool present**, begin labeling **new** mail only
→ existing mail stays untouched forever. Single-threaded interleave in the pubsub loop.

### Live adaptation (preserve today's behavior)

`label_change_handler.process_label_changes` currently writes bodies to
`training_store`/`skip_store` *and* updates the in-memory index
(`label_change_handler.py:109,117,139,144`). Under the new model the in-memory `index.add`
stays; the persistence target changes from "save body to MessageStore" to "upsert
`id→label` + `cache.put(id, vec)` in the derived store." Same learn-on-correction behavior,
no bodies persisted. The `index.add(...)` calls are untouched.

The live update points map 1:1 from today's body-writes to label+vector upserts:
- new inbox msg → skip (`inbox_check`/`history_processor` save empty-label body)
  ⇒ `labels.upsert(id, '__skip__')` + `embeddings.put(id, vec)`
- label applied/corrected (`label_change_handler.py:109,117`) ⇒
  `labels.upsert(id, name)` + `embeddings.put(id, vec)` + (unchanged) `index.add`
- label removed back to inbox (`label_change_handler.py:139,144`) ⇒
  `labels.upsert(id, '__skip__')`

## State lifecycle (the persistence guarantees)

The whole point is that **state survives everything except an ML change or an
explicit reset**:

| Event | Behavior |
|---|---|
| **Service restart** | `state.db` present → load index, straight to loop. No Gmail fetch. State intact. |
| **Code deploy, ML unchanged** | `gcp-deploy.sh` builds the tarball with `--exclude='data'` and untars *over* `INSTALL_DIR`; `tar x` never deletes files absent from the archive, so `data/state.db` is untouched. Deploy + restart → fast, state preserved. |
| **ML changed** (embedding model or `build_text_representation`) | Startup compares the `meta` fingerprint to the current code's. Mismatch ⇒ cached vectors are stale ⇒ **auto-rebuild from Gmail** and rewrite the fingerprint. No manual action, no silent wrong vectors. |
| **Explicit reset** | `make gcp-reset-state` / `make reset-state` — stop, `rm -f .../data/state.db`, start → next boot bootstraps fresh. The escape hatch "just in case." |

**Fingerprint:** a string like `"all-MiniLM-L6-v2|textrepr-v1"`. The model name
comes from `Embedder`'s `model_name`; the `textrepr-vN` is a manual constant
bumped whenever `build_text_representation`/`preprocess_email_body` changes in a
way that alters embeddings. Stored in `meta` on bootstrap, checked on every
startup.

**Design defaults chosen** (flag if you want the alternative):
- *One file, two tables* (not separate vector/label files) — single atomic reset.
- *Auto-rebuild on fingerprint mismatch* (not "refuse to start + tell the user to
  reset") — avoids surprise downtime requiring manual intervention; the explicit
  `reset-state` still exists for force.

## Files

- New `src/gmail_classifier/state_store.py` (or extend `embedding_cache.py`) — the
  `state.db` wrapper: `embeddings` + `labels` + `meta` tables; `upsert_label`,
  `get_labels()`, `iter_index()` (join), `get/set_fingerprint()`. One SQLite connection.
- New `src/gmail_classifier/bootstrap.py` — `bootstrap_index(client, embedder, store, ...)`,
  testable with fakes (no heavy imports), one-at-a-time fetch/embed/persist, resumable
  (skips ids already in `embeddings`). Folds in `fetcher.py`'s list/diff logic.
- `scripts/classify_and_label.py` — replace the `MessageStore.load_all` + `build_training_data`
  startup block with: **(a)** open `state.db`; **(b)** if fingerprint mismatches current ML,
  wipe + bootstrap; **(c)** if empty, bootstrap; **(d)** else load index from the join. Drop
  `--training-db`/`--skip-db` defaults for a single `--state-db` path (keep old flags as
  overrides for local use).
- `label_change_handler.py` — retarget persistence from `MessageStore.save_message` to
  `store.upsert_label` + `cache.put`; keep `index.add` untouched.
- `inbox_check.py` / `history_processor.py` — skip-pool write becomes `upsert_label(id, '__skip__')`.
- `scripts/gcp-deploy.sh` — stop shipping `data/*.db`; ship code + credentials only.
  Verify the `--exclude='data'` + untar-over behavior preserves `state.db` across deploys
  (it does today). Optional `--seed-state` to upload a prebuilt `state.db` and skip first
  boot, but not required.
- `Makefile` — add `reset-state` (local: stop, `rm -f data/state.db`) and `gcp-reset-state`
  (VM: stop, `rm -f $INSTALL_DIR/data/state.db`, start).
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

## Verification

- Unit: `bootstrap.py` with a fake client/embedder — builds the right index, **skips
  already-embedded ids on a second run** (resumability), excludes XL*, one-at-a-time embed.
- Memory: instrumented bootstrap on the VM shows peak materially below the old ~606 MB (no
  +447 MB corpus load); steady-state ~220 MB unchanged.
- Behavior: a correction still shifts a prediction (existing live-adaptation guarantee);
  full suite green.
- Deploy: on a freshly created VM, `gcp-create → gcp-deploy (code+creds) → gcp-start`
  produces a working classifier with **no DB upload**; restart is fast (reads derived store).

## Relationship to prior plans

- **Supersedes** the lightweight-classifier direction (`optimized-purring-globe.md` Phases
  1–3 and `classy_gcp_memory_classification_plan.md`) *for the GCP goal*: once the transient
  is gone and steady-state is ~220 MB, there's no memory case left for replacing the
  semantic classifier. Quality stays at the semantic baseline (no degradation).
- **Builds on** the Phase 0 findings and the startup fixes already shipped (`cf4ee18`,
  `1616593`, `fd0b6d6`, `80845fa`).
