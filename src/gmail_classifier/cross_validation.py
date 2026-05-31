from dataclasses import dataclass
from typing import List

import numpy as np

from gmail_classifier.classifier import (
    Action,
    SKIP_LABEL,
    aggregate_scores,
    compute_confidence,
    decide_action,
    find_neighbors,
    MIN_EXAMPLES_PER_LABEL,
)


@dataclass
class PredictionResult:
    true_label: str
    predicted_label: str
    confidence: float
    action: Action


def leave_one_out(
    embeddings: np.ndarray,
    labels: List[str],
    k: int = 5,
    extra_embeddings: np.ndarray | None = None,
    extra_labels: List[str] | None = None,
) -> List[PredictionResult]:
    """Run leave-one-out cross-validation.

    For each example i, classify it using all other examples as training data.
    If extra_embeddings/extra_labels are provided, they are always included
    in the training set (never left out) — used for __skip__ examples.

    Returns a list of PredictionResult, one per example.
    """
    n = len(embeddings)
    has_extra = extra_embeddings is not None and len(extra_embeddings) > 0
    results = []

    for i in range(n):
        # Build training set excluding example i
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        train_embs = embeddings[mask]
        train_labels = [labels[j] for j in range(n) if j != i]

        # Append extra (skip) examples if provided
        if has_extra:
            train_embs = np.vstack([train_embs, extra_embeddings])
            train_labels = train_labels + extra_labels

        # Determine eligible labels (>= MIN_EXAMPLES_PER_LABEL in remaining set)
        from collections import Counter
        counts = Counter(train_labels)
        eligible = {lbl for lbl, cnt in counts.items() if cnt >= MIN_EXAMPLES_PER_LABEL}

        # Find neighbors
        neighbors = find_neighbors(embeddings[i], train_embs, train_labels, k=k)

        # Filter to eligible labels
        eligible_neighbors = [(sim, lbl) for sim, lbl in neighbors if lbl in eligible]

        if not eligible_neighbors:
            results.append(PredictionResult(
                true_label=labels[i],
                predicted_label="",
                confidence=0.0,
                action=Action.NO_LABEL,
            ))
            continue

        scores = aggregate_scores(eligible_neighbors)
        predicted_label, confidence = compute_confidence(scores)

        # If __skip__ wins, treat as no label
        if predicted_label == SKIP_LABEL:
            results.append(PredictionResult(
                true_label=labels[i],
                predicted_label="",
                confidence=0.0,
                action=Action.NO_LABEL,
            ))
            continue

        action = decide_action(confidence)

        results.append(PredictionResult(
            true_label=labels[i],
            predicted_label=predicted_label,
            confidence=confidence,
            action=action,
        ))

    return results
