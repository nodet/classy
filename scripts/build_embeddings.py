#!/usr/bin/env python3
"""Pre-compute embeddings for all training and skip messages.

Writes to data/embeddings.db. This cache is used by classify_and_label.py
at startup to avoid recomputing embeddings on the VM.
"""
import argparse
import sys
from pathlib import Path

from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.embedding_cache import EmbeddingCache
from gmail_classifier.embeddings import Embedder
from gmail_classifier.storage import MessageStore
from gmail_classifier.training import build_training_data


def main():
    parser = argparse.ArgumentParser(description="Pre-compute embedding cache")
    parser.add_argument(
        "--training-db", default="data/training.db",
        help="Path to training message store",
    )
    parser.add_argument(
        "--skip-db", default="data/inbox_sample.db",
        help="Path to inbox/skip message store",
    )
    parser.add_argument(
        "--cache-db", default="data/embeddings.db",
        help="Path to embedding cache output",
    )
    parser.add_argument(
        "--exclude-labels", nargs="*", default=[],
        help="Labels to exclude from training",
    )
    args = parser.parse_args()

    # Load training messages
    print("Loading training data...")
    train_store = MessageStore(args.training_db)
    train_messages = train_store.load_all()
    train_store.close()

    if not train_messages:
        print("No training messages found.")
        sys.exit(1)

    excluded = set(args.exclude_labels)
    if excluded:
        train_messages = [m for m in train_messages if m.labels and m.labels[0] not in excluded]
        print(f"  Excluded labels: {', '.join(sorted(excluded))}")
    print(f"  {len(train_messages)} training messages")

    # Load skip examples
    skip_messages = []
    skip_path = Path(args.skip_db)
    if skip_path.exists():
        skip_store = MessageStore(args.skip_db)
        skip_messages = skip_store.load_all()
        skip_store.close()
        for m in skip_messages:
            m.labels = [SKIP_LABEL]
        print(f"  {len(skip_messages)} skip examples")

    all_messages = train_messages + skip_messages

    # Build with cache (computes only missing embeddings)
    cache = EmbeddingCache(args.cache_db)
    print("Computing embeddings (cached entries will be skipped)...")
    embedder = Embedder()
    embeddings, labels, ids = build_training_data(all_messages, embedder=embedder, cache=cache)
    cache.close()

    print(f"Done. {len(ids)} embeddings in {args.cache_db}")
    cache_size = Path(args.cache_db).stat().st_size
    print(f"  Cache size: {cache_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
