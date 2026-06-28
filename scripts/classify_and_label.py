#!/usr/bin/env python3
"""Classify unlabeled inbox messages and apply labels via Gmail API.

Uses training data + inbox snapshot as skip examples to classify
new messages that aren't in the skip pool.

Modes:
  poll (default): check inbox every N seconds
  pubsub: wait for Gmail push notifications via Pub/Sub
"""
import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from gmail_classifier.auth import get_credentials, get_gmail_service
from gmail_classifier.classifier import Action, SKIP_LABEL
from gmail_classifier.config import excluded_labels
from gmail_classifier.embedding_cache import EmbeddingCache
from gmail_classifier.embeddings import Embedder
from gmail_classifier.gmail_client import GmailClient
from gmail_classifier.history_processor import process_history_events
from gmail_classifier.label_change_handler import process_label_changes
from gmail_classifier.label_registry import LabelRegistry
from gmail_classifier.memory import log_mem, trim_memory
from gmail_classifier.storage import MessageStore
from gmail_classifier.training import build_training_data
from gmail_classifier.training_index import TrainingIndex

PUBSUB_TOPIC = "projects/classy-498012/topics/gmail-notifications"
PUBSUB_SUBSCRIPTION = "projects/classy-498012/subscriptions/gmail-notifications-sub"


MAX_LINE = 130


def now():
    """Timestamp prefix for log lines, with live RSS so we can watch
    memory track per-message processing over time."""
    from gmail_classifier.memory import rss_mb
    rss = rss_mb()
    mem = f"{rss:5.0f}MB" if rss is not None else "  n/a"
    return f"{datetime.now().strftime('%H:%M:%S')} {mem}"


def truncate(line):
    """Truncate a line to MAX_LINE chars, adding [...] if truncated."""
    if len(line) <= MAX_LINE:
        return line
    return line[:MAX_LINE - 6] + " [...]"


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
    args = parser.parse_args()

    log_mem("startup: before training DB load")

    # Load training data
    print("Loading training data...")
    train_store = MessageStore(args.training_db)
    train_messages = train_store.load_all()
    train_store.close()
    log_mem("startup: after training DB load")

    if not train_messages:
        print("No training messages found.")
        sys.exit(1)

    # Exclude labels
    excluded = set(excluded_labels())
    if excluded:
        train_messages = [m for m in train_messages if m.labels and m.labels[0] not in excluded]
        print(f"  Excluded labels: {', '.join(sorted(excluded))}")

    print(f"  {len(train_messages)} training messages")

    # Load skip examples
    skip_messages = []
    skip_ids = set()
    skip_path = Path(args.skip_db)
    if skip_path.exists():
        skip_store = MessageStore(args.skip_db)
        skip_messages = skip_store.load_all()
        skip_store.close()
        skip_ids = {m.id for m in skip_messages}
        for m in skip_messages:
            m.labels = [SKIP_LABEL]
        print(f"  {len(skip_messages)} skip examples")
    log_mem("startup: after skip DB load")

    # Build training index (with embedding cache for fast startup)
    all_train_messages = train_messages + skip_messages
    cache_path = Path(args.training_db).parent / "embeddings.db"
    cache = EmbeddingCache(str(cache_path))
    print("Embedding training data...")
    embedder = Embedder()
    log_mem("startup: after Embedder() load")
    train_embeddings, train_labels, train_ids = build_training_data(
        all_train_messages, embedder=embedder, cache=cache,
    )
    cache.close()
    log_mem("startup: after build_training_data")
    del all_train_messages, train_messages, skip_messages
    trim_memory()
    log_mem("startup: after del + malloc_trim")
    print(f"  {train_embeddings.shape[0]} embeddings, {train_embeddings.shape[1]} dimensions")

    # Connect to Gmail
    print("Authenticating...")
    credentials_dir = Path(args.credentials)
    creds = get_credentials(credentials_dir)
    service = get_gmail_service(credentials_dir)
    client = GmailClient(service)
    _credentials = creds  # saved for Pub/Sub client

    # Build label registry (refreshes automatically on new labels)
    registry = LabelRegistry(client, excluded=excluded)

    index = TrainingIndex(train_embeddings, train_labels, train_ids)

    if args.mode == "pubsub":
        _run_pubsub_mode(args, client, _credentials, embedder, index,
                         registry, skip_ids)
    else:
        _run_poll_mode(args, client, embedder, index,
                       registry, skip_ids)


def _run_poll_mode(args, client, embedder, index,
                   registry, skip_ids):
    """Poll inbox every N seconds."""
    print(f"\nReady (poll mode, every {args.interval}s). Ctrl+C to stop.\n")

    while True:
        _check_inbox(args, client, embedder, index, registry, skip_ids)

        if args.once:
            break
        time.sleep(args.interval)


def _process_events(events, args, client, embedder, index, registry,
                    skip_ids, self_labeled, dots):
    """Handle a batch of history events: label changes, classification, output.

    ``dots`` is a single-element list used as a mutable flag tracking whether
    an idle "." heartbeat was the last thing printed (so we can emit a newline
    before real output).
    """
    if not events:
        print(".", end="", flush=True)
        dots[0] = True
        return

    # Process label changes (update training/skip DBs + in-memory index)
    training_store = MessageStore(args.training_db)
    skip_store = MessageStore(args.skip_db)

    movements = process_label_changes(
        events=events,
        client=client,
        training_store=training_store,
        skip_store=skip_store,
        label_id_to_name=registry.id_to_name,
        user_label_ids=registry.user_label_ids,
        excluded_labels=set(),
        index=index,
        embedder=embedder,
        registry=registry,
        ignore_ids=self_labeled,
    )

    # Process new inbox messages
    results = process_history_events(
        events=events,
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

    # Print results (only if there's something to report)
    if movements or results:
        if dots[0]:
            print()  # newline after dots
            dots[0] = False
    for src, dst, count in movements:
        print(f"{now()} {count} {'email' if count == 1 else 'emails'} moved from {src} to {dst}")
    for r in results:
        sender = r["sender"]
        subject = r["subject"]
        w = registry.max_label_width
        if r["action"] in (Action.LABEL, Action.LABEL_WITH_REVIEW):
            print(truncate(f"{now()} {r['label']:{w}s}  {r['confidence']:6.1%}  {sender} — {subject}"))
            if r.get("applied"):
                self_labeled.add(r["message_id"])
        else:
            print(truncate(f"{now()} {'':{w}s}  {r['confidence']:6.1%}  {sender} — {subject}"))
            if not args.dry_run:
                msg = r["message"]
                msg.labels = []
                skip_store.save_message(msg)

    training_store.close()
    skip_store.close()

    # Hand back the heap a heavy message (big HTML parse + embed) just grew,
    # so RSS falls back to idle instead of ratcheting to the worst-case peak.
    trim_memory()
    log_mem("after events batch")


def _run_pubsub_mode(args, client, credentials, embedder, index,
                     registry, skip_ids):
    """Wait for Pub/Sub notifications and process via history API."""
    from gmail_classifier.pubsub import PubSubSubscriber
    from gmail_classifier.pubsub_loop import LoopState, LoopDeps, run_iteration

    # Register for notifications
    print("Registering Gmail watch...")
    history_id, expiration = client.watch(PUBSUB_TOPIC)
    print(f"  Watch active, historyId={history_id}")

    # Track messages labeled by the classifier itself (to ignore echoed events)
    self_labeled = set()

    # Do an initial inbox check to catch anything missed
    print("Initial inbox check...")
    _check_inbox(args, client, embedder, index, registry, skip_ids, self_labeled)

    if args.once:
        return

    def _make_subscriber():
        return PubSubSubscriber(
            subscription_path=PUBSUB_SUBSCRIPTION, credentials=credentials
        )

    print(f"\nReady (pubsub mode). Waiting for notifications...\n")
    log_mem("steady-state: pubsub loop ready")

    dots = [False]  # mutable heartbeat flag shared with _process_events

    def _log(message, lead_newline=False):
        prefix = "\n" if lead_newline else ""
        print(f"{prefix}{now()} {message}")

    deps = LoopDeps(
        make_subscriber=_make_subscriber,
        watch=lambda: client.watch(PUBSUB_TOPIC),
        get_history=client.get_history,
        check_inbox=lambda: _check_inbox(
            args, client, embedder, index, registry, skip_ids, self_labeled),
        process_events=lambda events: _process_events(
            events, args, client, embedder, index, registry,
            skip_ids, self_labeled, dots),
        log=_log,
    )

    state = LoopState(
        history_id=history_id,
        expiration=expiration,
        backoff=0,
        subscriber=_make_subscriber(),
    )
    try:
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


def _check_inbox(args, client, embedder, index, registry, skip_ids,
                 self_labeled=None):
    """Check inbox and classify new messages (poll mode)."""
    from gmail_classifier.inbox_check import process_inbox

    # Peek at whether there's anything new before opening the store, so the
    # idle case stays a cheap "." heartbeat with no DB handle.
    inbox_ids = client.list_message_ids(label_id="INBOX", max_results=args.max_messages)
    if not any(mid not in skip_ids for mid in inbox_ids):
        print(".", end="", flush=True)
        return

    print()  # newline after any dots
    skip_store = MessageStore(args.skip_db)
    try:
        results = process_inbox(
            client=client,
            embedder=embedder,
            index=index,
            registry=registry,
            skip_ids=skip_ids,
            skip_store=skip_store,
            k=args.k,
            max_messages=args.max_messages,
            dry_run=args.dry_run,
            self_labeled=self_labeled,
            inbox_ids=inbox_ids,
        )
    finally:
        skip_store.close()

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
    log_mem("after inbox batch")


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
    body = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    client.send_message(
        to="me",
        subject="gmail-classifier crashed",
        body=body,
    )


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm_handler)
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        print(f"\n{datetime.now().strftime('%H:%M:%S')} Stopped.")
        sys.exit(0)
    except Exception as e:
        try:
            _send_crash_alert(e)
        except Exception:
            pass  # don't mask the original error
        raise
