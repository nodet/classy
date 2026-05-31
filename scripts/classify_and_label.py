#!/usr/bin/env python3
"""Classify unlabeled inbox messages and apply labels via Gmail API.

Uses training data + inbox snapshot as skip examples to classify
new messages that aren't in the skip pool. Runs in a loop every 5 minutes.
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from gmail_classifier.auth import get_gmail_service
from gmail_classifier.classifier import classify, Action, SKIP_LABEL
from gmail_classifier.embeddings import Embedder
from gmail_classifier.gmail_client import GmailClient
from gmail_classifier.gmail_parser import parse_gmail_message
from gmail_classifier.preprocessing import preprocess_email_body, build_text_representation
from gmail_classifier.storage import MessageStore
from gmail_classifier.training import build_training_data


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
        help="Seconds between checks (default: 300)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run once and exit (no loop)",
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
    train_embeddings, train_labels = build_training_data(all_train_messages, embedder=embedder)
    print(f"  {train_embeddings.shape[0]} embeddings, {train_embeddings.shape[1]} dimensions")

    # Connect to Gmail
    print("Authenticating...")
    credentials_dir = Path(args.credentials)
    service = get_gmail_service(credentials_dir)
    client = GmailClient(service)

    # Get label name→id mapping
    user_labels = client.list_user_labels()
    label_name_to_id = {name: lid for lid, name in user_labels}
    user_label_ids = {lid for lid, name in user_labels}

    print(f"\nReady. Checking every {args.interval}s (Ctrl+C to stop).\n")

    while True:
        _check_inbox(args, client, embedder, train_embeddings, train_labels,
                     label_name_to_id, user_label_ids, skip_ids)

        if args.once:
            break
        time.sleep(args.interval)


def _check_inbox(args, client, embedder, train_embeddings, train_labels,
                 label_name_to_id, user_label_ids, skip_ids):
    """Check inbox and classify new messages."""
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
        if any(lid in user_label_ids for lid in msg_label_ids):
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
        result = classify(query_embedding, train_embeddings, train_labels, k=args.k)

        sender = msg.from_name or msg.from_address

        if result.action == Action.LABEL or result.action == Action.LABEL_WITH_REVIEW:
            label_id = label_name_to_id.get(result.label)
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
