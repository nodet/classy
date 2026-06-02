#!/usr/bin/env python3
"""Classify unlabeled inbox messages and apply labels via Gmail API.

Uses training data + inbox snapshot as skip examples to classify
new messages that aren't in the skip pool.

Modes:
  poll (default): check inbox every N seconds
  pubsub: wait for Gmail push notifications via Pub/Sub
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from gmail_classifier.auth import get_credentials, get_gmail_service
from gmail_classifier.classifier import classify, Action, SKIP_LABEL
from gmail_classifier.embeddings import Embedder
from gmail_classifier.gmail_client import GmailClient
from gmail_classifier.gmail_parser import parse_gmail_message
from gmail_classifier.history_processor import process_history_events
from gmail_classifier.label_change_handler import process_label_changes
from gmail_classifier.label_registry import LabelRegistry
from gmail_classifier.models import HistoryExpiredError
from gmail_classifier.preprocessing import preprocess_email_body, build_text_representation
from gmail_classifier.storage import MessageStore
from gmail_classifier.training import build_training_data
from gmail_classifier.training_index import TrainingIndex

PUBSUB_TOPIC = "projects/classy-498012/topics/gmail-notifications"
PUBSUB_SUBSCRIPTION = "projects/classy-498012/subscriptions/gmail-notifications-sub"


def now():
    return datetime.now().strftime("%H:%M:%S")


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
        "--exclude-labels", nargs="*", default=[],
        help="Labels to exclude from predictions",
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

    # Load training data
    print("Loading training data...")
    train_store = MessageStore(args.training_db)
    train_messages = train_store.load_all()
    train_store.close()

    if not train_messages:
        print("No training messages found.")
        sys.exit(1)

    # Exclude labels
    excluded = set(args.exclude_labels)
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

    # Build training index
    all_train_messages = train_messages + skip_messages
    print("Embedding training data...")
    embedder = Embedder()
    train_embeddings, train_labels, train_ids = build_training_data(all_train_messages, embedder=embedder)
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


def _run_pubsub_mode(args, client, credentials, embedder, index,
                     registry, skip_ids):
    """Wait for Pub/Sub notifications and process via history API."""
    from gmail_classifier.pubsub import PubSubSubscriber

    # Register for notifications
    print("Registering Gmail watch...")
    history_id, expiration = client.watch(PUBSUB_TOPIC)
    print(f"  Watch active, historyId={history_id}")

    # Do an initial inbox check to catch anything missed
    print("Initial inbox check...")
    _check_inbox(args, client, embedder, index, registry, skip_ids)

    if args.once:
        return

    # Track messages labeled by the classifier itself (to ignore echoed events)
    self_labeled = set()

    subscriber = PubSubSubscriber(
        subscription_path=PUBSUB_SUBSCRIPTION, credentials=credentials
    )
    print(f"\nReady (pubsub mode). Waiting for notifications...\n")

    while True:
        # Renew watch if expiring within 1 hour
        now_ms = int(time.time() * 1000)
        if expiration - now_ms < 3600_000:
            history_id_new, expiration = client.watch(PUBSUB_TOPIC)
            print(f"{now()} Watch renewed")

        # Pull notifications (blocks up to 60s)
        notifications = subscriber.pull(timeout=60)

        if not notifications:
            continue

        # Use the most recent historyId from notifications
        max_history = max(n.history_id for n in notifications)

        try:
            events = client.get_history(history_id)
        except HistoryExpiredError:
            print(f"{now()} History expired, falling back to inbox poll")
            _check_inbox(args, client, embedder, index, registry, skip_ids)
            # Re-watch to get fresh historyId
            history_id, expiration = client.watch(PUBSUB_TOPIC)
            continue

        if events:
            print()  # newline after any dots
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
            for src, dst, count in movements:
                print(f"{now()} {count} {'email' if count == 1 else 'emails'} moved from {src} to {dst}")

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

            # Print and persist results
            for r in results:
                sender = r["sender"]
                subject = r["subject"]
                if r["action"] in (Action.LABEL, Action.LABEL_WITH_REVIEW):
                    action_str = "LABEL" if r["action"] == Action.LABEL else "REVIEW"
                    print(f"{now()} [{action_str}] {r['label']:20s} {r['confidence']:5.1%}  {sender} — {subject}")
                    if r.get("applied"):
                        self_labeled.add(r["message_id"])
                else:
                    print(f"{now()} [SKIP]  {r['confidence']:5.1%}  {sender} — {subject}")
                    if not args.dry_run:
                        msg = r["message"]
                        msg.labels = []
                        skip_store.save_message(msg)

            training_store.close()
            skip_store.close()
        else:
            print(".", end="", flush=True)

        # Advance history pointer
        history_id = max_history


def _check_inbox(args, client, embedder, index, registry, skip_ids):
    """Check inbox and classify new messages (poll mode)."""
    inbox_ids = client.list_message_ids(label_id="INBOX", max_results=args.max_messages)
    new_ids = [mid for mid in inbox_ids if mid not in skip_ids]

    if not new_ids:
        print(".", end="", flush=True)
        return

    print()  # newline after any dots
    skip_store = MessageStore(args.skip_db)

    for mid in new_ids:
        raw = client.get_message(mid)

        # Check if it already has a user label
        msg_label_ids = raw.get("labelIds", [])
        if any(lid in registry.user_label_ids for lid in msg_label_ids):
            continue

        # Parse and classify
        msg = parse_gmail_message(raw)
        body = preprocess_email_body(msg.body_html)
        text = build_text_representation(
            from_name=msg.from_name,
            from_address=msg.from_address,
            subject=msg.subject,
            body=body,
            list_id=msg.list_id,
        )
        query_embedding = embedder.embed(text)
        result = classify(query_embedding, index.embeddings, index.labels, k=args.k)

        sender = msg.from_name or msg.from_address

        if result.action == Action.LABEL or result.action == Action.LABEL_WITH_REVIEW:
            label_id = registry.get_id(result.label)
            if not label_id:
                print(f"{now()} WARNING: label '{result.label}' not found in Gmail, skipping")
                continue

            action_str = "LABEL" if result.action == Action.LABEL else "REVIEW"
            print(f"{now()} [{action_str}] {result.label:20s} {result.confidence:5.1%}  {sender} — {msg.subject}")

            if not args.dry_run:
                client.apply_label(mid, label_id, archive=True)
        else:
            print(f"{now()} [SKIP]  {result.confidence:5.1%}  {sender} — {msg.subject}")
            if not args.dry_run:
                msg.labels = []
                skip_store.save_message(msg)

        # Remember this message so we don't re-process it
        skip_ids.add(mid)

    skip_store.close()


if __name__ == "__main__":
    main()
