"""Core evaluation logic: build training data, run LOO, compute metrics."""
from typing import List, Tuple

import numpy as np

from gmail_classifier.cross_validation import PredictionResult, leave_one_out
from gmail_classifier.evaluation import compute_metrics_table
from gmail_classifier.models import Message
from gmail_classifier.training import build_training_data

DEFAULT_THRESHOLDS = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]


def run_evaluation(
    messages: List[Message],
    k: int = 5,
    thresholds: List[float] | None = None,
    skip_messages: List[Message] | None = None,
) -> Tuple[List[Tuple[float, float, float]], List[PredictionResult]]:
    """Run full evaluation pipeline: embed, LOO cross-validation, metrics.

    If skip_messages are provided, they are included as __skip__ examples
    in every LOO iteration (never left out).

    Returns (metrics_table, prediction_results) where metrics_table is
    a list of (threshold, precision, coverage) tuples.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    from gmail_classifier.embeddings import Embedder
    embedder = Embedder()

    embeddings, labels = build_training_data(messages, embedder=embedder)

    extra_embeddings = None
    extra_labels = None
    if skip_messages:
        extra_embeddings, extra_labels = build_training_data(skip_messages, embedder=embedder)

    results = leave_one_out(
        embeddings, labels, k=k,
        extra_embeddings=extra_embeddings, extra_labels=extra_labels,
    )
    table = compute_metrics_table(results, thresholds)
    return table, results
