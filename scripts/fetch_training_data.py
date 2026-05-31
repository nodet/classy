#!/usr/bin/env python3
"""Fetch labeled messages from Gmail and store them locally.

Usage:
    python scripts/fetch_training_data.py [--db data/training.db] [--credentials credentials/]

First run will open a browser for OAuth consent.
"""

import argparse
import sys
from pathlib import Path

from gmail_classifier.auth import get_gmail_service
from gmail_classifier.fetcher import fetch_messages_for_label
from gmail_classifier.gmail_client import GmailClient
from gmail_classifier.storage import MessageStore


def main():
    parser = argparse.ArgumentParser(description="Fetch labeled Gmail messages")
    parser.add_argument("--db", default="data/training.db", help="SQLite database path")
    parser.add_argument("--credentials", default="credentials", help="Credentials directory")
    parser.add_argument("--labels", nargs="*", help="Only fetch these label names (default: all user labels)")
    args = parser.parse_args()

    credentials_dir = Path(args.credentials)
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print("Authenticating...")
    service = get_gmail_service(credentials_dir)
    client = GmailClient(service)

    print("Fetching labels...")
    all_labels = client.list_user_labels()
    print(f"  Found {len(all_labels)} user labels")

    if args.labels:
        labels_to_fetch = [(lid, name) for lid, name in all_labels if name in args.labels]
        skipped = set(args.labels) - {name for _, name in labels_to_fetch}
        if skipped:
            print(f"  Warning: labels not found: {skipped}")
    else:
        labels_to_fetch = all_labels

    store = MessageStore(str(db_path))

    for label_id, label_name in labels_to_fetch:
        print(f"  Fetching '{label_name}'...")
        fetch_messages_for_label(client, store, label_id, label_name)
        count = len(store.load_by_label(label_name))
        print(f"    {count} messages stored")

    total = len(store.load_all())
    print(f"\nDone. {total} total messages in {db_path}")
    store.close()


if __name__ == "__main__":
    main()
