#!/usr/bin/env python3
"""Train on stored messages and evaluate with leave-one-out cross-validation."""
import argparse
import sys
from collections import Counter

from gmail_classifier.config import excluded_labels
from gmail_classifier.evaluate import run_evaluation
from gmail_classifier.evaluation import per_label_precision
from gmail_classifier.storage import MessageStore


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate classification quality using leave-one-out cross-validation"
    )
    parser.add_argument(
        "--db", default="data/training.db",
        help="Path to the SQLite message store (default: data/training.db)",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of neighbors for KNN (default: 5)",
    )
    parser.add_argument(
        "--skip-db", default="data/inbox_sample.db",
        help="Path to inbox/skip messages used as negative examples (default: data/inbox_sample.db)",
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Don't use skip examples (evaluate without negative examples)",
    )
    args = parser.parse_args()

    # Load messages
    store = MessageStore(args.db)
    messages = store.load_all()
    store.close()

    if not messages:
        print("No messages found in the database.")
        sys.exit(1)

    # Exclude labels configured in config.toml
    excluded = set(excluded_labels())
    if excluded:
        messages = [m for m in messages if m.labels and m.labels[0] not in excluded]
        print(f"Excluded labels: {', '.join(sorted(excluded))}")
        print()

    # Show dataset summary
    label_counts = Counter(m.labels[0] for m in messages if m.labels)
    print(f"Dataset: {len(messages)} messages, {len(label_counts)} labels")
    print()
    print("Label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:30s} {count:5d}")
    print()

    # Load skip examples
    skip_messages = None
    if not args.no_skip:
        from pathlib import Path
        skip_path = Path(args.skip_db)
        if skip_path.exists():
            from gmail_classifier.classifier import SKIP_LABEL
            skip_store = MessageStore(args.skip_db)
            skip_messages = skip_store.load_all()
            skip_store.close()
            for m in skip_messages:
                m.labels = [SKIP_LABEL]
            print(f"Skip examples: {len(skip_messages)} messages from {args.skip_db}")
        else:
            print(f"Skip DB not found: {args.skip_db} (running without skip examples)")
    print()

    # Run evaluation
    print(f"Running leave-one-out cross-validation (k={args.k})...")
    print("(This embeds all messages and classifies each one — may take a while)")
    print()

    table, results = run_evaluation(messages, k=args.k, skip_messages=skip_messages)

    # Print metrics table
    print(f"{'Threshold':>10s} {'Precision':>10s} {'Coverage':>10s} {'Labeled':>8s}")
    print("-" * 42)
    for threshold, precision, coverage in table:
        n_labeled = sum(1 for r in results if r.confidence >= threshold and r.predicted_label)
        print(f"{threshold:>10.2f} {precision:>10.1%} {coverage:>10.1%} {n_labeled:>8d}")
    print()

    # Overall accuracy (all predictions made)
    predicted = [r for r in results if r.predicted_label]
    if predicted:
        correct = sum(1 for r in predicted if r.predicted_label == r.true_label)
        print(f"Overall accuracy (all predictions): {correct}/{len(predicted)} = {correct/len(predicted):.1%}")

    # Per-label precision at 0.80 threshold
    print(f"\nPer-label precision (at >=0.80 confidence):")
    print(f"  {'Label':30s} {'Precision':>10s} {'Correct':>8s} {'Total':>8s}")
    print(f"  {'-'*60}")
    plp = per_label_precision(results, 0.80)
    for label, (prec, correct, total) in sorted(plp.items(), key=lambda x: x[1][0]):
        print(f"  {label:30s} {prec:>10.1%} {correct:>8d} {total:>8d}")

    # Show some errors at the 0.80 threshold
    errors_80 = [
        r for r in results
        if r.confidence >= 0.80 and r.predicted_label and r.predicted_label != r.true_label
    ]
    if errors_80:
        print(f"\nErrors at >=0.80 confidence ({len(errors_80)} total):")
        for r in errors_80[:10]:
            print(f"  true={r.true_label:20s} pred={r.predicted_label:20s} conf={r.confidence:.3f}")


if __name__ == "__main__":
    main()
