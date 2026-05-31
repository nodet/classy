#!/usr/bin/env python3
"""Fetch recent unlabeled inbox messages for dry-run classification.

Usage:
    python scripts/fetch_inbox.py [--db data/inbox_sample.db] [--count 100]
"""

import argparse
from pathlib import Path

from gmail_classifier.auth import get_gmail_service
from gmail_classifier.gmail_client import GmailClient
from gmail_classifier.gmail_parser import parse_gmail_message
from gmail_classifier.storage import MessageStore


def main():
    parser = argparse.ArgumentParser(description="Fetch recent inbox messages")
    parser.add_argument("--db", default="data/inbox_sample.db", help="SQLite database path")
    parser.add_argument("--credentials", default="credentials", help="Credentials directory")
    parser.add_argument("--count", type=int, default=500, help="Number of messages to fetch")
    args = parser.parse_args()

    credentials_dir = Path(args.credentials)
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print("Authenticating...")
    service = get_gmail_service(credentials_dir)
    client = GmailClient(service)
    store = MessageStore(str(db_path))

    print(f"Fetching up to {args.count} inbox messages...")
    # List messages in INBOX (Gmail system label)
    message_ids = client.list_message_ids(label_id="INBOX")
    message_ids = message_ids[:args.count]

    new_ids = [mid for mid in message_ids if not store.has_message(mid)]
    print(f"  {len(message_ids)} in inbox, {len(new_ids)} new to fetch")

    if new_ids:
        raw_messages = client.get_messages(new_ids)
        for raw in raw_messages:
            msg = parse_gmail_message(raw)
            store.save_message(msg)

    total = len(store.load_all())
    print(f"\nDone. {total} messages in {db_path}")
    store.close()


if __name__ == "__main__":
    main()
