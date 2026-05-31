from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np


class Action(Enum):
    LABEL = "label"
    LABEL_WITH_REVIEW = "label_with_review"
    NO_LABEL = "no_label"


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    action: Action
    neighbors: List[Tuple[float, str]]


HIGH_CONFIDENCE_THRESHOLD = 0.95
MEDIUM_CONFIDENCE_THRESHOLD = 0.80
MIN_EXAMPLES_PER_LABEL = 5


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def find_neighbors(
    query: np.ndarray,
    training_embeddings: np.ndarray,
    labels: List[str],
    k: int = 5,
) -> List[Tuple[float, str]]:
    """Find the K most similar training examples to the query.

    Returns a list of (similarity, label) tuples sorted by similarity descending.
    """
    # Compute similarities against all training examples at once
    norm_query = np.linalg.norm(query)
    if norm_query == 0:
        return []
    norms = np.linalg.norm(training_embeddings, axis=1)
    # Avoid division by zero
    valid = norms > 0
    sims = np.zeros(len(training_embeddings))
    sims[valid] = training_embeddings[valid] @ query / (norms[valid] * norm_query)

    # Get top-k indices
    actual_k = min(k, len(training_embeddings))
    top_indices = np.argsort(sims)[::-1][:actual_k]

    return [(float(sims[i]), labels[i]) for i in top_indices]


def aggregate_scores(neighbors: List[Tuple[float, str]]) -> dict:
    """Sum similarity scores per label from neighbor list."""
    scores: dict = {}
    for sim, label in neighbors:
        scores[label] = scores.get(label, 0.0) + sim
    return scores


def compute_confidence(scores: dict) -> Tuple[str, float]:
    """Compute confidence as ratio of winning score to total score.

    Returns (winning_label, confidence) where confidence is between 0 and 1.
    """
    if not scores:
        return ("", 0.0)
    total = sum(scores.values())
    if total == 0:
        return ("", 0.0)
    winning_label = max(scores, key=scores.get)
    confidence = scores[winning_label] / total
    return (winning_label, confidence)


def decide_action(confidence: float) -> Action:
    """Decide what action to take based on confidence level."""
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return Action.LABEL
    elif confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return Action.LABEL_WITH_REVIEW
    else:
        return Action.NO_LABEL


def _eligible_labels(labels: List[str]) -> set:
    """Return labels that have at least MIN_EXAMPLES_PER_LABEL training examples."""
    from collections import Counter
    counts = Counter(labels)
    return {label for label, count in counts.items() if count >= MIN_EXAMPLES_PER_LABEL}


def classify(
    query: np.ndarray,
    training_embeddings: np.ndarray,
    labels: List[str],
    k: int = 5,
) -> ClassificationResult:
    """Full classification pipeline: find neighbors, aggregate, decide."""
    if len(training_embeddings) == 0:
        return ClassificationResult(
            label="", confidence=0.0, action=Action.NO_LABEL, neighbors=[]
        )

    eligible = _eligible_labels(labels)

    neighbors = find_neighbors(query, training_embeddings, labels, k=k)
    # Filter neighbors to only eligible labels
    eligible_neighbors = [(sim, lbl) for sim, lbl in neighbors if lbl in eligible]

    if not eligible_neighbors:
        return ClassificationResult(
            label="", confidence=0.0, action=Action.NO_LABEL, neighbors=neighbors
        )

    scores = aggregate_scores(eligible_neighbors)
    label, confidence = compute_confidence(scores)
    action = decide_action(confidence)

    return ClassificationResult(
        label=label, confidence=confidence, action=action, neighbors=neighbors
    )
