#!/usr/bin/env python3
"""Classify unlabeled inbox messages and apply labels via Gmail API.

Uses training data + inbox snapshot as skip examples to classify
new messages that aren't in the skip pool.

Modes:
  poll (default): check inbox every N seconds
  pubsub: wait for Gmail push notifications via Pub/Sub
"""
import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Set

from gmail_classifier.auth import get_credentials, get_gmail_service
from gmail_classifier.classifier import Action
from gmail_classifier.config import excluded_labels
from gmail_classifier.embeddings import Embedder
from gmail_classifier.gmail_client import GmailClient
from gmail_classifier.history_processor import process_history_events
from gmail_classifier.label_change_handler import process_label_changes
from gmail_classifier.label_registry import LabelRegistry
from gmail_classifier.memory import trim_memory
from gmail_classifier.storage_legacy import LegacyBackend

PUBSUB_TOPIC = "projects/classy-498012/topics/gmail-notifications"
PUBSUB_SUBSCRIPTION = "projects/classy-498012/subscriptions/gmail-notifications-sub"


MAX_LINE = 130


def now():
    """Timestamp prefix for log lines, with live RSS so we can watch
    memory track per-message processing over time (see memory.log_prefix)."""
    from gmail_classifier.memory import log_prefix
    return log_prefix()


def truncate(line):
    """Truncate a line to MAX_LINE chars, adding [...] if truncated."""
    if len(line) <= MAX_LINE:
        return line
    return line[:MAX_LINE - 6] + " [...]"


def deployed_version():
    """Code version stamped at deploy time (see scripts/gcp-deploy.sh).

    The VM has no .git, so gcp-deploy writes `git describe` into
    `.deployed_version` at the repo root. Returns "unknown" if absent (e.g. a
    local run straight from a checkout, where git itself is the source of truth).
    """
    stamp = Path(__file__).resolve().parent.parent / ".deployed_version"
    try:
        return stamp.read_text().strip() or "unknown"
    except OSError:
        return "unknown"


def _validate_storage_mode(args):
    """Reject storage/mode combinations that would violate the state backend's
    read-only boundary.

    Poll mode *is* the labeling inbox path: ``_run_poll_mode`` calls
    ``_check_inbox`` every interval, which lists the current INBOX and hands it
    to ``process_inbox`` -- classify + apply labels with ``archive=True`` +
    record skips. That is correct for legacy (it treats the current inbox as
    work) but unsafe for the state backend, which must only advance via history
    replay from its durable cursor and never label/archive pre-boundary backlog.
    ``--mode`` defaults to poll while ``--storage``/``$CLASSY_STORAGE`` can be
    state, so guard here rather than trusting the pubsub-only sweep gate. This is
    permanent: the state backend advances only via history replay (and, on an
    aged cursor, the read-only resync), never a labeling poll -- so poll mode has
    no safe meaning for it.

    Also validate the storage value itself: argparse only enforces ``choices``
    for values typed on the command line, not for the default -- and the default
    comes straight from ``$CLASSY_STORAGE``. So a typo like ``CLASSY_STORAGE=stat``
    would slip through, and ``_build_backend`` would treat any non-"legacy" value
    as the state backend. Reject unknown values here, before that dispatch.
    """
    if args.storage not in ("legacy", "state"):
        print(
            f"--storage must be 'legacy' or 'state' (got {args.storage!r}). "
            "Check the --storage argument or the $CLASSY_STORAGE environment "
            "variable for a typo."
        )
        sys.exit(1)

    if args.storage == "state" and args.mode != "pubsub":
        print(
            f"--storage state requires --mode pubsub (got --mode {args.mode}). "
            "Poll mode sweeps and labels the current inbox, which would archive "
            "pre-boundary backlog under the state backend; it advances only via "
            "history replay. Re-run with --mode pubsub."
        )
        sys.exit(1)


@dataclass
class BootstrapPlan:
    """A deferred progressive cold bootstrap (Phase 5).

    The BOOTSTRAP path no longer blocks: ``_build_backend`` pins the read-only
    boundary (so the durable cursor exists and Pub/Sub can resume) and returns
    this plan instead of fetching the whole corpus. The caller loads the empty
    index, then hands the plan to ``_run_pubsub_mode``, which builds a
    ``ProgressiveBootstrap`` over the *live* index and interleaves batches with
    notification servicing. ``None`` everywhere else (warm/reconcile/rebuild
    finish synchronously, and legacy has no bootstrap)."""
    excluded: Set[str]
    max_per_label: int
    gmail_account_id: Optional[str]


def _build_backend(args, excluded, client, embedder):
    """Construct the selected StorageBackend and, for a cold state boot, a
    deferred :class:`BootstrapPlan`. Returns ``(backend, plan)`` where ``plan``
    is ``None`` unless a progressive bootstrap must run. The only place that
    branches on the backend choice; callers downstream see just a StorageBackend.

    Legacy is today's three-DB adapter. State opens the single state.db and
    dispatches on its persisted meta:

    - WARM -> load the join directly.
    - BOOTSTRAP -> pin the boundary now, defer the fetch to a progressive
      driver that runs interleaved with the live loop (Phase 5).
    - REBUILD -> ML fingerprint changed; re-embed into state.rebuild.db and
      atomically swap it in (carrying the cursor forward).
    - RECONCILE -> excluded-label set changed; cheap membership fix in place.
    - INCOMPATIBLE -> schema/account mismatch; a hard stop (fail closed).

    ``client``/``embedder`` are the already-authenticated Gmail client and the
    embedder, so the state path can validate the store against the *actual*
    mailbox account id and current ML fingerprint.
    """
    if args.storage == "legacy":
        return LegacyBackend(args.training_db, args.skip_db, excluded), None

    # Explicit: anything that isn't a known backend is a bug or an unvalidated
    # value slipping past _validate_storage_mode -- never silently fall through
    # to the state backend (which archives differently and must not run by
    # accident on a mistyped $CLASSY_STORAGE).
    if args.storage != "state":
        raise ValueError(f"unknown storage backend: {args.storage!r}")

    from gmail_classifier.storage_state import (
        STATE_SCHEMA_VERSION,
        StartupDecision,
        StateBackend,
        compute_excluded_hash,
        compute_ml_fingerprint,
        decide_startup,
    )

    # The live mailbox's account id, so a state.db copied from another account
    # is rejected instead of warm-started with the wrong label ids/cursor.
    gmail_account_id = client.get_profile_email()

    backend = StateBackend(args.state_db, excluded)
    decision = decide_startup(
        backend.store,
        schema_version=STATE_SCHEMA_VERSION,
        ml_fingerprint=compute_ml_fingerprint(embedder.model_name, embedder.dimension),
        excluded_hash=compute_excluded_hash(excluded),
        gmail_account_id=gmail_account_id,
    )

    if decision is StartupDecision.WARM:
        return backend, None

    if decision is StartupDecision.INCOMPATIBLE:
        # Schema or account mismatch: the file is unusable as-is and there is no
        # safe automatic recovery (a wrong-account DB must not be silently
        # rebuilt over). Fail closed; the operator resets explicitly.
        backend.close()
        print(
            "State backend is INCOMPATIBLE with this mailbox/config (schema or "
            "account mismatch). Refusing to load or overwrite it; run "
            "'make reset-state' to bootstrap fresh."
        )
        sys.exit(1)

    from gmail_classifier import bootstrap as _bootstrap

    if decision is StartupDecision.BOOTSTRAP:
        # Empty/interrupted store: progressive cold bootstrap (Phase 5). Pin the
        # read-only boundary NOW -- watch() first, before any message is read, so
        # pre-existing mail stays read-only and the durable cursor exists for the
        # loop to resume from. The actual fetch is deferred to a
        # ProgressiveBootstrap that the pubsub loop pumps between notifications,
        # so the service is live from the first second rather than blocking on a
        # ~10-20 min cold fetch.
        print("State backend: bootstrapping from Gmail (progressive)...")
        _bootstrap.ensure_boundary(
            client, backend.store, PUBSUB_TOPIC,
            log=lambda m: print(f"  {m}", flush=True))
        plan = BootstrapPlan(
            excluded=excluded, max_per_label=args.max_per_label,
            gmail_account_id=gmail_account_id)
        return backend, plan

    if decision is StartupDecision.RECONCILE:
        # Excluded-label config changed: cheap membership fix, no re-embed of
        # retained ids.
        print("State backend: reconciling excluded-label change...")
        _bootstrap.reconcile_exclusions(
            client, embedder, backend.store,
            excluded=excluded, max_per_label=args.max_per_label,
            log=lambda m: print(f"  {m}", flush=True),
        )
        return backend, None

    if decision is StartupDecision.REBUILD:
        # ML fingerprint changed: vectors are stale, label map + cursor are not.
        # Build state.rebuild.db, then atomically swap it in.
        print("State backend: ML changed, rebuilding embeddings from Gmail...")
        rebuilt = _rebuild_and_swap(args, excluded, client, embedder, backend)
        # A single deploy can change BOTH the ML fingerprint and the excluded
        # set. decide_startup checks ML before exclusions, so it returned REBUILD
        # and the rebuild carried the OLD excluded_labels_hash forward. Fold the
        # membership change in now -- otherwise a long-running daemon keeps using
        # now-excluded labels (and omits newly-included ones) until some later
        # restart happens to hit RECONCILE. reconcile_exclusions rewrites the hash
        # so the next boot sees WARM.
        current_hash = compute_excluded_hash(excluded)
        if rebuilt.store.get_meta("excluded_labels_hash") != current_hash:
            print("State backend: exclusions also changed, reconciling...")
            _bootstrap.reconcile_exclusions(
                client, embedder, rebuilt.store,
                excluded=excluded, max_per_label=args.max_per_label,
                log=lambda m: print(f"  {m}", flush=True),
            )
        return rebuilt, None

    # Defensive: decide_startup returns only the enum members handled above.
    backend.close()
    raise ValueError(f"unhandled startup decision: {decision!r}")


def _rebuild_db_path(state_db: str) -> str:
    """``data/state.db`` -> ``data/state.rebuild.db`` (same dir/filesystem so the
    swap is an atomic rename, not a cross-device copy)."""
    if state_db.endswith(".db"):
        return state_db[:-3] + ".rebuild.db"
    return state_db + ".rebuild"


def _rebuild_and_swap(args, excluded, client, embedder, backend):
    """Re-embed into state.rebuild.db, close both, atomically swap, reopen.

    The old state.db is never removed before the validated replacement is in
    place; a crash mid-rebuild leaves the old store untouched (the rebuild store
    lacks the completed fingerprint, so it would not validate as WARM anyway).
    """
    from gmail_classifier import bootstrap as _bootstrap
    from gmail_classifier.storage_state import StateBackend

    rebuild_path = _rebuild_db_path(args.state_db)
    # Start the rebuild from a clean file so a stale prior attempt can't leak in.
    for path in (rebuild_path, rebuild_path + "-wal", rebuild_path + "-shm",
                 rebuild_path + "-journal"):
        if os.path.exists(path):
            os.remove(path)

    rebuild_backend = StateBackend(rebuild_path, excluded)
    _bootstrap.rebuild_index(
        client, embedder, backend.store, rebuild_backend.store,
        log=lambda m: print(f"  {m}", flush=True),
    )
    # Close BOTH connections before renaming so no -wal/-shm is orphaned.
    backend.close()
    rebuild_backend.close()
    _bootstrap.atomic_swap_state_db(args.state_db, rebuild_path)
    return StateBackend(args.state_db, excluded)


def main():
    parser = argparse.ArgumentParser(
        description="Classify inbox messages and apply labels"
    )
    parser.add_argument(
        "--training-db", default="data/training.db",
        help="Path to training message store (default: data/training.db)",
    )
    parser.add_argument(
        "--skip-db", default="data/inbox_sample.db",
        help="Path to inbox/skip message store (default: data/inbox_sample.db)",
    )
    parser.add_argument(
        "--state-db", default="data/state.db",
        help="Path to the state backend's single DB (default: data/state.db)",
    )
    parser.add_argument(
        "--credentials", default="credentials",
        help="Credentials directory (default: credentials)",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of neighbors for KNN (default: 5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without modifying Gmail",
    )
    parser.add_argument(
        "--max-messages", type=int, default=50,
        help="Max inbox messages to process per run (default: 50)",
    )
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Seconds between checks in poll mode (default: 300)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run once and exit (no loop)",
    )
    parser.add_argument(
        "--mode", choices=["poll", "pubsub"], default="poll",
        help="Notification mode: poll (default) or pubsub",
    )
    parser.add_argument(
        "--storage", choices=["legacy", "state"],
        default=os.environ.get("CLASSY_STORAGE", "legacy"),
        help="Storage backend (default: legacy, or $CLASSY_STORAGE)",
    )
    parser.add_argument(
        "--max-per-label", type=int, default=200,
        help="State backend: max messages to bootstrap per label and for the "
             "skip pool (default: 200). Bounds first-boot fetch size.",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="State backend: print a status report of the local state.db "
             "(schema, fingerprint, bootstrap status, index/label/skip/pending "
             "counts, history cursor) and exit. Reads only state.db -- no Gmail "
             "call, no model load -- so it is safe to run alongside the service.",
    )
    parser.add_argument(
        "--test-alert", action="store_true",
        help="Send a test crash-alert email and exit. Exercises the full "
             "production alert path (auth + Gmail send) without starting the "
             "service. Run to verify crash notifications will work.",
    )
    args = parser.parse_args()

    # --report is a read-only status dump; it never authenticates or loads the
    # model, so handle it before the storage/mode validation (which is about the
    # *running* service) and before any Gmail connection.
    if args.report:
        from gmail_classifier.state_status import print_report
        print_report(args.state_db, excluded_config=list(excluded_labels()))
        return

    if args.test_alert:
        raise RuntimeError("Test alert: crash notification is working")

    _validate_storage_mode(args)

    print(f"gmail-classifier version: {deployed_version()}", flush=True)

    excluded = set(excluded_labels())
    if excluded:
        print(f"  Excluded labels: {', '.join(sorted(excluded))}")

    # Connect to Gmail FIRST. The state backend needs the authenticated
    # mailbox's account id to validate that a state.db belongs to this mailbox
    # (a copied/stale DB from another account must not warm-start), and that
    # check has to run before the backend is selected -- so auth precedes the
    # backend build rather than following the index load.
    print("Authenticating...")
    credentials_dir = Path(args.credentials)
    creds = get_credentials(credentials_dir)
    service = get_gmail_service(credentials_dir)
    client = GmailClient(service)
    _credentials = creds  # saved for Pub/Sub client

    embedder = Embedder()

    # Build the storage backend behind the seam. The runtime below talks only
    # to this interface, never to a concrete store. Only the selector branches
    # on which backend to build; everything downstream sees a StorageBackend.
    # ``plan`` is a deferred progressive bootstrap on a cold state boot, else
    # None.
    backend, plan = _build_backend(args, excluded, client, embedder)

    # Build the runtime index (config exclusion + labeled-wins-over-skip dedup +
    # cache-backed embedding). The assembly logic lives in the backend/training
    # module so it is unit-testable apart from this I/O shell.
    print("Embedding training data...")
    loaded = backend.load_index(embedder)
    index, skip_ids, stats = loaded.index, loaded.skip_ids, loaded.stats

    # A cold progressive boot legitimately starts with an EMPTY index -- the
    # ProgressiveBootstrap fills it in the loop. Only treat an empty index as
    # fatal when there is no bootstrap plan to populate it (a warm store that
    # somehow has nothing is a real error).
    if stats.n_train == 0 and plan is None:
        print("No training messages found.")
        sys.exit(1)

    if plan is not None:
        print("  starting with an empty index; bootstrapping progressively")
    else:
        print(f"  {stats.n_train} training messages")
        note = f" ({stats.n_dropped} also labeled, kept as labeled)" if stats.n_dropped else ""
        print(f"  {stats.n_skip} skip examples{note}")
    trim_memory()
    print(f"  {index.embeddings.shape[0]} embeddings, {index.embeddings.shape[1]} dimensions")

    # Build label registry (refreshes automatically on new labels)
    registry = LabelRegistry(client, excluded=excluded)

    try:
        if args.mode == "pubsub":
            _run_pubsub_mode(args, client, _credentials, embedder, index,
                             registry, skip_ids, backend, plan=plan)
        else:
            _run_poll_mode(args, client, embedder, index,
                           registry, skip_ids, backend)
    finally:
        backend.close()


def _run_poll_mode(args, client, embedder, index,
                   registry, skip_ids, backend):
    """Poll inbox every N seconds."""
    print(f"\nReady (poll mode, every {args.interval}s). Ctrl+C to stop.\n")

    while True:
        _check_inbox(args, client, embedder, index, registry, skip_ids, backend)

        if args.once:
            break
        time.sleep(args.interval)


def _classify_new_ids(new_ids, args, client, embedder, index, registry,
                      skip_ids, self_labeled, backend):
    """Fetch, classify, and label/skip a list of new inbox message ids.

    Reuses ``process_history_events`` by synthesizing ``messagesAdded`` events
    for the ids, so the drain path and the live path share one classifier.
    Returns the result dicts (for the caller to fold into its trim decision)."""
    from gmail_classifier.pending_new import events_for_ids

    # These ids are to be classified as new. The drain path passes ids that were
    # parked during warmup, and parking added them to skip_ids -- so drop them
    # here, or collect_new_inbox_ids (inside process_history_events) would filter
    # them straight back out and the drain would classify nothing while still
    # clearing the pending rows. process_history_events re-adds each id to
    # skip_ids after classifying, so a duplicate event in the same session is
    # still ignored. In the live path these ids are already absent from skip_ids,
    # so the discard is a no-op.
    for mid in new_ids:
        skip_ids.discard(mid)

    results = process_history_events(
        events=events_for_ids(new_ids),
        client=client,
        embedder=embedder,
        train_embeddings=index.embeddings,
        train_labels=index.labels,
        label_name_to_id=registry.name_to_id,
        user_label_ids=registry.user_label_ids,
        excluded_labels=registry._excluded,
        skip_ids=skip_ids,
        k=args.k,
        dry_run=args.dry_run,
        registry=registry,
    )
    w = registry.max_label_width
    for r in results:
        sender = r["sender"]
        subject = r["subject"]
        if r["action"] in (Action.LABEL, Action.LABEL_WITH_REVIEW):
            print(truncate(f"{now()} {r['label']:{w}s}  {r['confidence']:6.1%}  {sender} — {subject}"))
            if r.get("applied"):
                self_labeled.add(r["message_id"])
        else:
            print(truncate(f"{now()} {'':{w}s}  {r['confidence']:6.1%}  {sender} — {subject}"))
            if not args.dry_run:
                msg = r["message"]
                backend.upsert_skip(msg, r.get("embedding"))
    return results


def _process_events(events, args, client, embedder, index, registry,
                    skip_ids, self_labeled, backend, controller=None,
                    history_id=None):
    """Handle a batch of history events: label changes, classification, output.

    ``controller`` is the progressive-bootstrap maturity controller, or ``None``
    on warm/legacy (where new mail is always classified immediately). While a
    controller exists and the index is not yet mature, genuinely-new inbox mail
    is **parked** in ``pending_new`` (no label, no archive) rather than
    classified -- guarding a still-warming model from over-labeling a fresh
    mailbox. Label-change events are always processed, so user corrections keep
    growing the index even before maturity."""
    if not events:
        return

    from gmail_classifier.history_processor import collect_new_inbox_ids
    from gmail_classifier.pending_new import park_immature_mail

    # Process label changes (persist via the backend + update in-memory index).
    # This runs regardless of maturity -- corrections are always learned.
    movements = process_label_changes(
        events=events,
        client=client,
        backend=backend,
        label_id_to_name=registry.id_to_name,
        user_label_ids=registry.user_label_ids,
        excluded_labels=set(),
        index=index,
        embedder=embedder,
        registry=registry,
        ignore_ids=self_labeled,
    )

    for src, dst, count in movements:
        print(f"{now()} {count} {'email' if count == 1 else 'emails'} moved from {src} to {dst}")

    results = []
    if controller is not None and not controller.is_mature():
        # Pre-maturity: park new inbox mail; do NOT classify or archive it. It
        # is drained through the normal classifier once the gate opens.
        parked = park_immature_mail(events, skip_ids, backend,
                                    history_id=history_id or "")
        if parked:
            print(f"{now()} parked {len(parked)} new message(s) until the "
                  f"index matures")
    else:
        new_ids = collect_new_inbox_ids(events, skip_ids)
        if new_ids:
            results = _classify_new_ids(
                new_ids, args, client, embedder, index, registry,
                skip_ids, self_labeled, backend)

    # Only reclaim when the batch did real work. Hand back the heap a heavy
    # message (big HTML parse + embed) just grew, so RSS falls back to idle
    # instead of ratcheting to the worst-case peak.
    if movements or results:
        trim_memory()


class _MaturityController:
    """Owns the progressive bootstrap driver and the maturity transition.

    The loop consults :meth:`is_mature` to decide whether new mail is labelled
    or parked, and calls :meth:`maybe_drain` after each step so that the moment
    the gate opens, the mail parked while the model was warming is classified
    through the normal path and cleared -- exactly once."""

    def __init__(self, driver, drain_fn, log):
        self._driver = driver
        self._drain_fn = drain_fn  # () -> classify+clear pending_new
        self._log = log
        self._drained = False

    @property
    def done(self):
        return self._driver.done

    def run_batch(self):
        return self._driver.run_batch()

    def is_mature(self):
        return self._driver.is_mature()

    def maybe_drain(self):
        """Drain pending_new once, the first time the gate is open. Idempotent:
        a second call after draining is a no-op, and a crash mid-drain leaves
        the survivors for the next call (drain_pending removes rows only after
        they are processed)."""
        if self._drained or not self._driver.is_mature():
            return
        self._log("Index matured; draining parked mail through the classifier")
        self._drain_fn()
        self._drained = True


def _make_drain(args, client, embedder, index, registry, skip_ids,
                self_labeled, backend):
    """A closure that classifies and clears every parked ``pending_new`` row
    through the normal classifier. Reused by the bootstrap controller (drained
    when the gate opens) and by the warm-startup drain below."""
    from gmail_classifier.pending_new import drain_pending

    def _drain():
        drain_pending(backend, process_ids=lambda ids: _classify_new_ids(
            ids, args, client, embedder, index, registry,
            skip_ids, self_labeled, backend))

    return _drain


def _build_controller(plan, args, client, embedder, index, registry, skip_ids,
                      self_labeled, backend, log):
    """Construct the progressive-bootstrap maturity controller, or ``None`` when
    there is no cold bootstrap to run (warm/reconcile/rebuild/legacy)."""
    if plan is None:
        return None

    from gmail_classifier.progressive import ProgressiveBootstrap

    driver = ProgressiveBootstrap(
        client=client, embedder=embedder, store=backend.store, index=index,
        excluded=plan.excluded, max_per_label=plan.max_per_label,
        gmail_account_id=plan.gmail_account_id,
        log=lambda m: print(f"  {m}", flush=True),
    )
    drain = _make_drain(args, client, embedder, index, registry, skip_ids,
                        self_labeled, backend)
    return _MaturityController(driver, drain, log)


def _run_pubsub_mode(args, client, credentials, embedder, index,
                     registry, skip_ids, backend, plan=None):
    """Wait for Pub/Sub notifications and process via history API.

    When ``plan`` is set (a cold state boot), the index starts empty and is
    filled by a ``ProgressiveBootstrap`` interleaved with notification
    servicing: one bounded batch per loop step until the corpus is embedded.
    New mail arriving before the index matures is parked; it is drained the
    moment the maturity gate opens."""
    from gmail_classifier.pubsub import PubSubSubscriber
    from gmail_classifier.pubsub_loop import (
        LoopState, LoopDeps, run_bootstrap_iteration, run_iteration,
    )

    # Register for notifications
    print("Registering Gmail watch...")
    watch_history_id, expiration = client.watch(PUBSUB_TOPIC)
    print(f"  Watch active, historyId={watch_history_id}")

    # Resume from the backend's durable cursor if it has one (state backend, warm
    # restart) so history since the last processed id is replayed rather than
    # skipped by adopting the fresh watch boundary. Legacy has no durable cursor
    # (returns None), so it starts from the fresh watch id exactly as before.
    is_legacy = args.storage == "legacy"
    resume_id = backend.get_last_processed_history_id()

    # The state read-only resync recovery (Phase 6): the shared recovery for both
    # a long-outage warm restart (classify_gap -> RESYNC just below) and a mid-run
    # history expiry (wired as the loop's `resync` dep). It reconciles labels from
    # Gmail as truth, re-pins a fresh boundary, rebuilds the live index/skip_ids
    # in place, and returns the fresh (history_id, expiration). It NEVER labels or
    # archives the accumulated backlog. None on legacy (no durable cursor).
    def _resync():
        from gmail_classifier import bootstrap as _bootstrap
        return _bootstrap.read_only_resync(
            client, embedder, backend.store, PUBSUB_TOPIC,
            excluded=registry._excluded, max_per_label=args.max_per_label,
            index=index, skip_ids=skip_ids,
            log=lambda m: print(f"  {m}", flush=True))

    if is_legacy:
        # Legacy adopts the fresh watch id as its starting cursor, as today.
        history_id = watch_history_id
    else:
        # State: decide from the durable cursor + last_processed_at how to
        # recover. A missing/expired/over-window/invalid-timestamp cursor no
        # longer fails closed (Phase 6) -- it takes the read-only resync instead.
        from gmail_classifier.storage_state import GapDecision, classify_gap
        if classify_gap(backend.store, backend.store.now_ms()) is GapDecision.REPLAY:
            # Genuine short outage: replay-and-classify from the persisted cursor.
            history_id = resume_id
            print(f"  Resuming from persisted historyId={resume_id}")
        else:
            # Long outage, or missing/invalid cursor/timestamp: read-only resync
            # + re-pin. Never labels/archives the backlog; only mail after the
            # fresh boundary is classifiable.
            print("State backend: history gap too large; running read-only resync...")
            history_id, expiration = _resync()

    # Track messages labeled by the classifier itself (to ignore echoed events)
    self_labeled = set()

    # The labeling inbox sweep is LEGACY-ONLY. It lists the current INBOX and
    # classifies whatever is unlabeled -- catch-up-after-downtime behavior that
    # is correct for legacy (which always fresh-watches and treats the current
    # inbox as work). It is unsafe for the state backend: inbox listing carries
    # no per-message historyId, so it cannot enforce the read-only boundary, and
    # known_ids is incomplete when the skip pool is only sampled -- so a sweep
    # could label/archive pre-boundary backlog. State catches up via history
    # replay from its durable cursor instead; when the cursor is expired or the
    # gap exceeds WARM_RECOVERY_WINDOW it takes the read-only resync + re-pin
    # (reconcile labels, never label/archive the backlog), never an inbox sweep.
    if is_legacy:
        # Do an initial inbox check to catch anything missed.
        print("Initial inbox check...")
        _check_inbox(args, client, embedder, index, registry, skip_ids, backend,
                     self_labeled)

    # Drain any pending_new rows stranded by a crash that happened after
    # bootstrap wrote bootstrap_status="complete" but before all parked mail was
    # drained. On the next boot that store reads WARM (plan is None), so no
    # bootstrap controller is built and the controller's maybe_drain never runs
    # -- the rows would sit forever. A warm store is by definition mature, so
    # classify + clear them here at startup. A cold boot (plan set) instead lets
    # the controller drain once its gate opens; legacy's get_pending() is a
    # no-op, so this is empty there. Runs before the once-return so a --once
    # restart still recovers stranded rows.
    if plan is None:
        _make_drain(args, client, embedder, index, registry, skip_ids,
                    self_labeled, backend)()

    if args.once:
        return

    def _make_subscriber():
        return PubSubSubscriber(
            subscription_path=PUBSUB_SUBSCRIPTION, credentials=credentials
        )

    print(f"\nReady (pubsub mode). Waiting for notifications...\n")

    def _log(message, lead_newline=False):
        prefix = "\n" if lead_newline else ""
        print(f"{prefix}{now()} {message}")

    def _check_inbox_fallback():
        # The loop calls this on HistoryExpiredError only on legacy (state
        # supplies the `resync` dep instead, so its expiry branch never reaches
        # here). On legacy it's the inbox poll -- catch-up-after-downtime, as
        # today. On state a labeling inbox sweep would archive pre-boundary
        # backlog, which is exactly what the read-only resync avoids.
        _check_inbox(args, client, embedder, index, registry, skip_ids, backend,
                     self_labeled)

    # Progressive bootstrap controller (cold state boot only). None on warm/
    # legacy, where _process_events classifies new mail immediately.
    controller = _build_controller(
        plan, args, client, embedder, index, registry, skip_ids, self_labeled,
        backend, _log)

    def _process(events):
        # The loop advances its own history_id AFTER process_events; pass the
        # current cursor so parked rows record the historyId they were first
        # seen at.
        _process_events(events, args, client, embedder, index, registry,
                        skip_ids, self_labeled, backend,
                        controller=controller, history_id=state.history_id)

    # State-only recovery hooks (None/no-op on legacy). The resync is the history-
    # expiry recovery (reconcile + re-pin, never label the backlog); the heartbeat
    # keeps a live-but-idle cursor from aging into the recovery window. Both are
    # gated off while the progressive bootstrap runs: the boundary is fresh and
    # the store not yet complete, so neither applies.
    def _heartbeat(now_ms):
        if is_legacy or (controller is not None and not controller.done):
            return
        from gmail_classifier import bootstrap as _bootstrap
        _bootstrap.heartbeat_cursor(client, backend.store, now_ms, log=_log)

    deps = LoopDeps(
        make_subscriber=_make_subscriber,
        watch=lambda: client.watch(PUBSUB_TOPIC),
        get_history=client.get_history,
        check_inbox=_check_inbox_fallback,
        process_events=_process,
        persist_cursor=backend.set_last_processed_history_id,
        log=_log,
        is_bootstrapping=lambda: controller is not None and not controller.done,
        resync=None if is_legacy else _resync,
        heartbeat=_heartbeat,
    )

    state = LoopState(
        history_id=history_id,
        expiration=expiration,
        backoff=0,
        subscriber=_make_subscriber(),
    )
    try:
        # Phase 5: while the progressive bootstrap is running, each loop step
        # embeds one bounded batch and THEN services notifications, so a
        # notification that arrived mid-bootstrap is handled between batches --
        # not after the whole corpus. Once the gate opens, drain parked mail.
        while controller is not None and not controller.done:
            state = run_bootstrap_iteration(state, deps, controller)
            controller.maybe_drain()
        # A gate that opened only on the final batch still needs its drain.
        if controller is not None:
            controller.maybe_drain()

        while True:
            state = run_iteration(state, deps)
    finally:
        # Close the gRPC channel deterministically on shutdown (SIGTERM ->
        # SystemExit). Otherwise its __del__ finalizer fires during interpreter
        # teardown and races a threading lock, printing a harmless but noisy
        # traceback.
        try:
            state.subscriber.close()
        except Exception:
            pass


def _check_inbox(args, client, embedder, index, registry, skip_ids, backend,
                 self_labeled=None):
    """Check inbox and classify new messages (poll mode)."""
    from gmail_classifier.inbox_check import process_inbox

    # Peek at whether there's anything new before doing any work, so the idle
    # case stays cheap.
    inbox_ids = client.list_message_ids(label_id="INBOX", max_results=args.max_messages)
    if not any(mid not in skip_ids for mid in inbox_ids):
        return

    results = process_inbox(
        client=client,
        embedder=embedder,
        index=index,
        registry=registry,
        skip_ids=skip_ids,
        backend=backend,
        k=args.k,
        max_messages=args.max_messages,
        dry_run=args.dry_run,
        self_labeled=self_labeled,
        inbox_ids=inbox_ids,
    )

    w = registry.max_label_width
    for r in results:
        sender = r["sender"]
        if r.get("warning"):
            print(f"{now()} WARNING: label '{r['label']}' not found in Gmail, skipping")
        elif r["action"] in (Action.LABEL, Action.LABEL_WITH_REVIEW):
            print(truncate(f"{now()} {r['label']:{w}s}  {r['confidence']:6.1%}  {sender} — {r['subject']}"))
        else:
            print(truncate(f"{now()} {'':{w}s}  {r['confidence']:6.1%}  {sender} — {r['subject']}"))

    # Heavy parse+embed work just ran; return the heap to the OS (see
    # _process_events).
    trim_memory()


def _sigterm_handler(signum, frame):
    # One-shot: restore default disposition so a second SIGTERM arriving during
    # interpreter teardown (threading._shutdown joining slow grpc threads)
    # terminates the process normally instead of raising SystemExit into
    # shutdown code, which prints an "Exception ignored" traceback.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    raise SystemExit(0)


def _send_crash_alert(exc):
    """Attempt to email ourselves a crash notification."""
    import traceback
    from gmail_classifier.auth import get_gmail_service
    from gmail_classifier.gmail_client import GmailClient

    service = get_gmail_service(Path("credentials"))
    client = GmailClient(service)
    to = client.get_profile_email()
    body = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    client.send_message(
        to=to,
        subject="gmail-classifier crashed",
        body=body,
    )


def _run_with_shutdown_handling(main_fn):
    """Run ``main_fn`` and translate shutdown/exit into the right process code.

    A clean stop (Ctrl-C, or SIGTERM via ``_sigterm_handler`` raising
    ``SystemExit(0)``) prints "Stopped." and exits 0. But a **failed** exit --
    ``_validate_storage_mode``'s ``sys.exit(1)`` or the state cursor's
    ``raise SystemExit(...)`` -- must propagate its original non-zero code, not
    be flattened to success. Otherwise systemd/cron/CI would read a rejected
    config or fail-closed startup as a clean run and never restart or alert.
    """
    try:
        main_fn()
    except KeyboardInterrupt:
        print(f"\n{datetime.now().strftime('%H:%M:%S')} Stopped.")
        sys.exit(0)
    except SystemExit as e:
        # Only a zero/None code is a clean stop worth the "Stopped." banner;
        # any non-zero code is a real failure and must reach the caller intact.
        if e.code in (0, None):
            print(f"\n{datetime.now().strftime('%H:%M:%S')} Stopped.")
        raise
    except Exception as e:
        try:
            _send_crash_alert(e)
        except Exception:
            pass  # don't mask the original error
        raise


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm_handler)
    try:
        _run_with_shutdown_handling(main)
    except RuntimeError as e:
        if "Test alert" in str(e):
            print("Test alert sent.")
        else:
            raise
