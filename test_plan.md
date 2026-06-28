# Test Plan: Pub/Sub Loop + Coverage Gaps

Status: proposed (2026-06-28). Motivated by a near-regression in the pub/sub
reconnect fix (commit `0c4ec97`) that no test would have caught.

**Progress:** Tier 1 DONE (2026-06-28) — extracted
`src/gmail_classifier/pubsub_loop.py` (`run_iteration`, `next_backoff`,
`is_network_error`, `LoopState`, `LoopDeps`); `_run_pubsub_mode` is now a thin
wrapper; 14 tests in `tests/test_pubsub_loop.py` (all 8 planned cases). The
backlog guard was confirmed by bug-injection (advancing history_id on recovery
makes it fail). Full suite: 180 passing. Tiers 2-3 still pending.

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

- **`auth.py` (0 tests).** With `tmp_path` + mocked `Credentials`/`build`:
  loads existing valid token; refreshes an expired token with refresh_token;
  raises `FileNotFoundError` when neither token nor client secret exists.
  Don't exercise the interactive browser flow.
- **`label_registry.py` (no dedicated file; only used incidentally).**
  `ensure_known()` returns True without refresh when id known; refreshes once
  and returns True when newly present; returns False when still unknown.
  `max_label_width` excludes excluded labels; `is_excluded`/`get_id`/
  `get_name` basics.
- **`_send_crash_alert` (`classify_and_label.py:420`).** Extract to importable
  helper; assert it formats the traceback and calls `client.send_message`, and
  that a failure inside it is swallowed (doesn't mask the original exception —
  the `__main__` guard's `except Exception: pass`).
- **`dry_run.py` core.** If feasible after Tier 2 extraction, reuse the shared
  classify step so dry-run is covered transitively; otherwise skip (low value,
  read-only script).

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
4. Tier-3 gap-fill (auth, label_registry, crash alert).

## Verification

After each tier: `uv run --with pytest python -m pytest -q` (full suite).
Target: the loop refactor adds no behavioral change detectable by the existing
tests, and case 1 fails if `history_id` is advanced on recovery (i.e. it
reproduces the near-regression).
