# Test Plan: Pub/Sub Loop + Coverage Gaps

Status: proposed (2026-06-28). Motivated by a near-regression in the pub/sub
reconnect fix (commit `0c4ec97`) that no test would have caught.

**Progress:**

- Tier 1 DONE (2026-06-28, commit `ad1b199`) — extracted
  `src/gmail_classifier/pubsub_loop.py` (`run_iteration`, `next_backoff`,
  `is_network_error`, `LoopState`, `LoopDeps`); `_run_pubsub_mode` is now a
  thin wrapper; 14 tests in `tests/test_pubsub_loop.py` (all 8 planned cases).
  The backlog guard was confirmed by bug-injection (advancing history_id on
  recovery makes it fail).
- Tier 2 DONE (2026-06-28) — extracted `process_inbox` into
  `src/gmail_classifier/inbox_check.py`; `_check_inbox` is now a thin wrapper
  (owns MessageStore lifecycle + printing). 7 tests in
  `tests/test_inbox_check.py` (label/apply, already-labeled skip, NO_LABEL
  skip-store save, dry-run, missing-label warning, skip_ids filtering,
  supplied-inbox_ids). Removed now-dead imports from the script. Full suite:
  187 passing.
- Tier 3 DONE (2026-06-28) — no refactor needed. Discovered
  `label_registry.py` was already well-tested (11 tests), so 3a collapsed to 1
  edge-case test; added 3 `auth.get_credentials` branch tests in new
  `tests/test_auth.py`. 4 new tests total (not 9). Full suite: 191 passing.

## Background / problem

`scripts/classify_and_label.py` holds the two long-running event loops
(`_run_pubsub_mode`, `_run_poll_mode`) and the inbox classification step
(`_check_inbox`). None of them have tests. The script can't be imported in a
test process because module-level imports pull in heavy ML deps
(`onnxruntime`, `Embedder`, numpy stacks). `tests/test_sigterm.py` documents
this and works around it with a `subprocess` + inlined-code pattern.

The most fragile behavior — the **offline backlog guarantee** (on connection
recovery, `history_id` is NOT advanced to the fresh `watch()` result, so
`get_history()` still reaches back before the outage and processes mail that
piled up) — is emergent from the loop's control flow and is currently
unprotected. See [[pubsub-loop-untested]].

## Strategy decision: extract, don't subprocess

Two options for making the loop testable:

- **(A) Subprocess + fake injected modules** (like `test_sigterm`). Works
  without touching `classify_and_label.py`, but is slow, awkward to assert on
  (parse stdout), and can't inspect `history_id`/`backoff` state directly.
- **(B) Refactor the loop body into an importable, dependency-injected unit.**
  Preferred. Keeps heavy ML deps out of the import path and lets tests assert
  on returned state directly.

**Chosen: (B).** The refactor is itself the highest-value change — it's what
makes the backlog guarantee assertable.

### Refactor sketch (prerequisite for the loop tests)

Goal: isolate the per-iteration logic so it imports cleanly and takes its
collaborators as arguments.

1. Create `src/gmail_classifier/pubsub_loop.py` (new module, no heavy imports —
   only stdlib + typing; `process_history_events` / `process_label_changes`
   are already light and live in their own modules).
2. Move the per-iteration body of `_run_pubsub_mode` into a pure-ish function:

   ```python
   def run_iteration(state, client, subscriber, deps) -> LoopState:
       """One iteration: handle backoff/recreate, pull, recover, process.
       Returns the next LoopState (history_id, expiration, backoff, ...).
       Raises nothing for network errors — encodes them in returned state."""
   ```

   where `LoopState` is a small dataclass `(history_id, expiration, backoff,
   subscriber)` and `deps` bundles the processing callables + config
   (topic, timeouts, caps) so they can be faked.
3. `scripts/classify_and_label.py` keeps `_run_pubsub_mode` as a thin `while
   True:` wrapper that owns the real `client`/`embedder`/`subscriber`
   construction and calls `run_iteration` repeatedly. The heavy-dep wiring
   stays in the script; the logic moves to the importable module.
4. Factor the duplicated network-error backoff handling (currently in both the
   `except (OSError, ConnectionError)` and `except Exception` 503 branches,
   `classify_and_label.py:333-351`) into one helper
   `next_backoff(current) -> int` and test it directly.

Keep each step behavior-preserving; run the existing suite (esp.
`test_pubsub.py`, `test_history_processor.py`) between steps.

## Tier 1 — Pub/Sub loop tests (the priority)

New file: `tests/test_pubsub_loop.py`. All use fakes — no network, no ML.
Fakes needed:

- `FakeSubscriber`: `pull(timeout)` returns a scripted sequence (lists or
  raises `OSError`/`ServiceUnavailable` per call); records `close()` calls and
  construction count.
- `FakeClient`: records `watch()` / `get_history(history_id)` calls; `watch()`
  returns a configurable fresh history id; `get_history` returns scripted
  events or raises `HistoryExpiredError`.
- Stub processing deps (no-op `process_history_events` /
  `process_label_changes` returning scripted results).

Cases:

1. **Backlog preserved across outage (the core regression guard).**
   Start `history_id=100`. Drive: healthy → `pull` raises `OSError` (enter
   backoff) → next `pull` returns `[]` (recovered, empty) → next `pull`
   returns a notification. Assert `get_history` was ultimately called with
   `100` (or the last pre-outage id), NOT the fresh id returned by the
   recovery `watch()`. This is the test that would have caught the near-miss.

2. **Recovery on empty pull.** With `backoff>0`, a `pull()` returning `[]`
   resets `backoff` to 0 and emits "Connection restored" exactly once.
   (Pre-fix behavior would deepen backoff instead — assert the new behavior.)

3. **Old subscriber closed on retry.** After entering backoff and looping,
   assert the previous subscriber's `close()` was called exactly once per
   recreation and a new subscriber was constructed.

4. **Backoff progression + cap.** Repeated failures yield 5 → 10 → 20 → 40 →
   60 → 60 (capped at 60, per `next_backoff`). Drives the extracted helper.

5. **Network error classification.** `OSError`/`ConnectionError` and a gRPC
   "ServiceUnavailable"/"503" both enter backoff; a non-network `Exception`
   propagates (is re-raised), matching `classify_and_label.py:350-351`.

6. **History expired → inbox fallback + re-watch.** `get_history` raises
   `HistoryExpiredError` → fallback inbox check invoked, `watch()` called to
   get a fresh id, loop continues without crashing.

7. **Watch renewal near expiry.** When `expiration - now < 1h` and healthy,
   `watch()` is called and `history_id` is preserved (renewal must not skip
   the backlog). Inject a fake clock so no real time passes.

8. **Self-labeled echo suppression.** A message id the classifier just labeled
   is added to `self_labeled` and ignored when it echoes back via history.
   (Covers the `ignore_ids` path — currently untested end-to-end.)

## Tier 2 — `_check_inbox` extraction + tests

`_check_inbox` (`classify_and_label.py:354-413`) is the poll-mode classify
step and is also reachable from pubsub mode (initial check + history-expired
fallback). Extract its body to an importable function taking injected
`client`/`embedder`/`index`/`registry` and test:

1. New unlabeled inbox message → classified, `apply_label(archive=True)`
   called, id added to `skip_ids` and `self_labeled`.
2. Message already carrying a user label → skipped (no classify/apply).
3. Low-confidence / SKIP action → saved to skip store, not labeled.
4. `--dry-run` → no `apply_label`, no store writes.
5. Label predicted but missing in Gmail → WARNING path, no apply, no crash.
6. Already-in-`skip_ids` messages filtered before any API fetch.

## Tier 3 — fill module gaps (no refactor needed)

Scoped down after review: the goal is to pin genuine logic, not to chase
coverage. Several candidates were rejected as glue (testing them only asserts a
mock was called).

**Correction (2026-06-28):** the original sketch claimed `label_registry.py`
had no tests. That was wrong — `tests/test_label_registry.py` already had 11
tests covering `max_label_width` exclusion, all three `ensure_known` branches,
and the `is_excluded` unknown-id guard. So 3a collapsed to a single missing
edge case. Real net for Tier 3: **4 new tests, not 9.**

### 3a. `label_registry.py` — DONE (1 new test; the rest already existed)

Already covered by the pre-existing file. Only gap added:

1. `max_label_width` is 0 when everything is excluded (the `default=0` guard —
   must not crash on an empty non-excluded set).
   (`test_max_label_width_zero_when_all_excluded`)

### 3b. `auth.py::get_credentials` — DONE (3 branches, new `tests/test_auth.py`)

Branch *selection* is ours and regressable. Uses `tmp_path` for the credentials
dir; patches `auth.Credentials.from_authorized_user_file` and `auth.Request`.

1. Valid stored token present → returned as-is; `refresh()` NOT called; token
   file NOT rewritten. (`test_valid_token_reused_without_refresh_or_rewrite`)
2. Expired token WITH a refresh_token → `creds.refresh(Request())` called and
   the refreshed token written back to `tmp_path/token.json`.
   (`test_expired_token_is_refreshed_and_resaved`)
3. No token file AND no `client_secret.json` → raises `FileNotFoundError`.
   (`test_missing_token_and_secret_raises`)

### Rejected (documented so we don't reconsider on a whim)

- **`auth.get_gmail_service`** — one-line `build("gmail","v1",...)` wrapper;
  a test only asserts the mock was called. Glue, skip.
- **`get_credentials` interactive browser flow** (`run_local_server`) — can't /
  shouldn't unit-test.
- **`_send_crash_alert` (function body)** — glue: build service, format
  traceback, call `send_message`. Skip.
- **`__main__` crash-alert masking contract** (a crash-alert failure must not
  mask the original exception, lines 377-382) — genuinely valuable but lives in
  `__main__`, so it needs a subprocess test like `test_sigterm`. Low frequency;
  leaning skip. Revisit only if the crash-alert path changes.
- **`dry_run.py`** — read-only script; low value. Skip.

## Out of scope

- Real network / real Gmail / real Pub/Sub calls.
- The ML model itself (embeddings/classifier already have dedicated tests).
- `__main__` signal wiring beyond what `test_sigterm.py` already covers.

## Sequencing

1. Tier-1 refactor (extract `pubsub_loop.run_iteration` + `next_backoff`),
   keeping the existing suite green.
2. Add `tests/test_pubsub_loop.py` (cases 1–8). **Land case 1 first** — it is
   the regression guard that justifies this whole plan.
3. Tier-2 extraction + `_check_inbox` tests.
4. Tier-3 gap-fill — `label_registry` (~6) then `auth.get_credentials` (3); no
   refactor needed (both already importable).

## Verification

After each tier: `uv run --with pytest python -m pytest -q` (full suite).
Target: the loop refactor adds no behavioral change detectable by the existing
tests, and case 1 fails if `history_id` is advanced on recovery (i.e. it
reproduces the near-regression).
