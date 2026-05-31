#!/usr/bin/env python3
"""Dry-run classification: classify inbox messages without modifying Gmail.

Shows what the classifier would do for each unlabeled inbox message.
"""
import argparse
import sys

from gmail_classifier.classifier import classify, Action
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

    # Build training index
    print("Embedding training data...")
    embedder = Embedder()
    train_embeddings, train_labels = build_training_data(train_messages, embedder=embedder)
    print(f"  {train_embeddings.shape[0]} embeddings, {train_embeddings.shape[1]} dimensions")

    # Load inbox messages
    print("Loading inbox messages...")
    inbox_store = MessageStore(args.inbox_db)
    inbox_messages = inbox_store.load_all()
    inbox_store.close()

    if not inbox_messages:
        print("No inbox messages found.")
        sys.exit(1)

    print(f"  {len(inbox_messages)} inbox messages")
    print()

    # Classify each inbox message
    action_counts = {Action.LABEL: 0, Action.LABEL_WITH_REVIEW: 0, Action.NO_LABEL: 0}

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
        result = classify(query_embedding, train_embeddings, train_labels, k=args.k)

        action_counts[result.action] += 1

        # Format verdict
        if result.action == Action.LABEL:
            verdict = "WOULD LABEL"
        elif result.action == Action.LABEL_WITH_REVIEW:
            verdict = "WOULD LABEL (review)"
        else:
            verdict = "SKIP (low confidence)"

        # Print result
        sender = msg.from_name or msg.from_address
        print(f"{'─' * 70}")
        print(f"From: {sender}")
        print(f"Subject: {msg.subject}")
        print(f"  → {result.label or '(none)'} (confidence: {result.confidence:.1%}) — {verdict}")

        # Show neighbors
        if result.neighbors:
            print(f"  Neighbors:")
            for sim, lbl in result.neighbors[:5]:
                print(f"    {sim:.3f}  [{lbl}]")
        print()

    # Summary
    print(f"{'═' * 70}")
    print(f"Summary ({len(inbox_messages)} messages):")
    print(f"  WOULD LABEL:          {action_counts[Action.LABEL]:4d}")
    print(f"  WOULD LABEL (review): {action_counts[Action.LABEL_WITH_REVIEW]:4d}")
    print(f"  SKIP (low confidence):{action_counts[Action.NO_LABEL]:4d}")


if __name__ == "__main__":
    main()
