#!/usr/bin/env python3
"""Dry-run classification: classify inbox messages without modifying Gmail.

Shows what the classifier would do for each unlabeled inbox message.
"""
import argparse
import sys

import numpy as np

from gmail_classifier.classifier import classify, Action, SKIP_LABEL
from gmail_classifier.embeddings import Embedder
from gmail_classifier.preprocessing import preprocess_email_body, build_text_representation
from gmail_classifier.storage import MessageStore
from gmail_classifier.training import build_training_data


def main():
    parser = argparse.ArgumentParser(
        description="Dry-run classification on inbox messages"
    )
    parser.add_argument(
        "--training-db", default="data/training.db",
        help="Path to training message store (default: data/training.db)",
    )
    parser.add_argument(
        "--inbox-db", default="data/inbox_sample.db",
        help="Path to inbox message store (default: data/inbox_sample.db)",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of neighbors for KNN (default: 5)",
    )
    parser.add_argument(
        "--exclude-labels", nargs="*", default=[],
        help="Labels to exclude from predictions (e.g. --exclude-labels XLC XLE XLCap)",
    )
    parser.add_argument(
        "--skip-db", default="",
        help="Path to inbox/skip message store (messages used as negative examples)",
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

    # Exclude labels if requested
    excluded = set(args.exclude_labels)
    if excluded:
        train_messages = [m for m in train_messages if m.labels and m.labels[0] not in excluded]
        print(f"  Excluded labels: {', '.join(sorted(excluded))}")

    print(f"  {len(train_messages)} training messages")

    # Load skip (negative) examples if provided
    skip_messages = []
    skip_db = args.skip_db or args.inbox_db
    if skip_db:
        print(f"Loading skip examples from {skip_db}...")
        skip_store = MessageStore(skip_db)
        skip_messages = skip_store.load_all()
        skip_store.close()
        # Tag them with __skip__ label
        for m in skip_messages:
            m.labels = [SKIP_LABEL]
        print(f"  {len(skip_messages)} skip examples")

    # Combine training + skip for building index
    all_train_messages = train_messages + skip_messages

    # Build training index
    print("Embedding training data...")
    embedder = Embedder()
    train_embeddings, train_labels = build_training_data(all_train_messages, embedder=embedder)
    print(f"  {train_embeddings.shape[0]} embeddings, {train_embeddings.shape[1]} dimensions")

    # Load inbox messages (same DB as skip — we classify them to see what would happen)
    print("Loading inbox messages...")
    inbox_store = MessageStore(args.inbox_db)
    inbox_messages = inbox_store.load_all()
    inbox_store.close()

    if not inbox_messages:
        print("No inbox messages found.")
        sys.exit(1)

    print(f"  {len(inbox_messages)} inbox messages")
    print()

    # Build index of inbox message IDs to their position in training set
    # (skip messages are appended after train_messages)
    n_train = len(train_messages)
    inbox_id_to_train_idx = {}
    for i, msg in enumerate(skip_messages):
        inbox_id_to_train_idx[msg.id] = n_train + i

    # Classify each inbox message
    sure = []
    review = []
    low = []

    for msg in inbox_messages:
        body = preprocess_email_body(msg.body_html)
        text = build_text_representation(
            from_name=msg.from_name,
            from_address=msg.from_address,
            subject=msg.subject,
            body=body,
            list_id=msg.list_id,
        )
        query_embedding = embedder.embed(text)

        # Exclude this message from training set (LOO) if it's in skip examples
        if msg.id in inbox_id_to_train_idx:
            idx = inbox_id_to_train_idx[msg.id]
            mask = np.ones(len(train_embeddings), dtype=bool)
            mask[idx] = False
            result = classify(query_embedding, train_embeddings[mask],
                            [l for i, l in enumerate(train_labels) if mask[i]], k=args.k)
        else:
            result = classify(query_embedding, train_embeddings, train_labels, k=args.k)

        sender = msg.from_name or msg.from_address
        top_sim = result.neighbors[0][0] if result.neighbors else 0.0
        line = f"  {result.label or '(none)':20s} {result.confidence:5.1%} sim={top_sim:.3f}  {sender} — {msg.subject}"

        if result.action == Action.LABEL:
            sure.append(line)
        elif result.action == Action.LABEL_WITH_REVIEW:
            review.append(line)
        else:
            low.append(line)

    # Print grouped results (least interesting first)
    print(f"LOW ({len(low)}):")
    for line in low:
        print(line)

    print(f"\nREVIEW ({len(review)}):")
    for line in review:
        print(line)

    print(f"\nSURE ({len(sure)}):")
    for line in sure:
        print(line)

    print(f"\nTotal: {len(sure)} sure, {len(review)} review, {len(low)} low")


if __name__ == "__main__":
    main()
