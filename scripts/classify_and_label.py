#!/usr/bin/env python3
"""Classify unlabeled inbox messages and apply labels via Gmail API.

Listens for Gmail push notifications via Pub/Sub, classifies new messages
using KNN on embeddings, and applies labels. The state backend (state.db)
bootstraps from Gmail on first boot and maintains a durable history cursor.
"""
import argparse
import os
import signal
import sys
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
    """Construct the state backend and, for a cold boot, a deferred
    :class:`BootstrapPlan`. Returns ``(backend, plan)`` where ``plan``
    is ``None`` unless a progressive bootstrap must run.

    Dispatches on the persisted meta in state.db:

    - WARM -> load the join directly.
    - BOOTSTRAP -> pin the boundary now, defer the fetch to a progressive
      driver that runs interleaved with the live loop.
    - REBUILD -> ML fingerprint changed; re-embed into state.rebuild.db and
      atomically swap it in (carrying the cursor forward).
    - RECONCILE -> excluded-label set changed; cheap membership fix in place.
    - INCOMPATIBLE -> schema/account mismatch; a hard stop (fail closed).
    """
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
        "--state-db", default="data/state.db",
        help="Path to state.db (default: data/state.db)",
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
        "--once", action="store_true",
        help="Run once and exit (no loop)",
    )
    parser.add_argument(
        "--max-per-label", type=int, default=200,
        help="Max messages to bootstrap per label and for the "
             "skip pool (default: 200). Bounds first-boot fetch size.",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print a status report of the local state.db "
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

    if args.report:
        from gmail_classifier.state_status import print_report
        print_report(args.state_db, excluded_config=list(excluded_labels()))
        return

    if args.test_alert:
        raise RuntimeError("Test alert: crash notification is working")

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
        _run_pubsub_mode(args, client, _credentials, embedder, index,
                         registry, skip_ids, backend, plan=plan)
    finally:
        backend.close()


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

    resume_id = backend.get_last_processed_history_id()

    def _resync():
        from gmail_classifier import bootstrap as _bootstrap
        return _bootstrap.read_only_resync(
            client, embedder, backend.store, PUBSUB_TOPIC,
            excluded=registry._excluded, max_per_label=args.max_per_label,
            index=index, skip_ids=skip_ids,
            log=lambda m: print(f"  {m}", flush=True))

    from gmail_classifier.storage_state import GapDecision, classify_gap
    if classify_gap(backend.store, backend.store.now_ms()) is GapDecision.REPLAY:
        history_id = resume_id
        print(f"  Resuming from persisted historyId={resume_id}")
    else:
        print("State backend: history gap too large; running read-only resync...")
        history_id, expiration = _resync()

    # Track messages labeled by the classifier itself (to ignore echoed events)
    self_labeled = set()

    # Drain any pending_new rows stranded by a crash that happened after
    # bootstrap completed but before all parked mail was drained.
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

    # Progressive bootstrap controller (cold boot only). None on warm start,
    # where _process_events classifies new mail immediately.
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

    def _heartbeat(now_ms):
        if controller is not None and not controller.done:
            return
        from gmail_classifier import bootstrap as _bootstrap
        _bootstrap.heartbeat_cursor(client, backend.store, now_ms, log=_log)

    deps = LoopDeps(
        make_subscriber=_make_subscriber,
        watch=lambda: client.watch(PUBSUB_TOPIC),
        get_history=client.get_history,
        check_inbox=lambda: None,
        process_events=_process,
        persist_cursor=backend.set_last_processed_history_id,
        log=_log,
        is_bootstrapping=lambda: controller is not None and not controller.done,
        resync=_resync,
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
