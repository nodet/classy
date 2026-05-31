from enum import Enum
from typing import List, Tuple

import numpy as np


class Action(Enum):
    LABEL = "label"
    LABEL_WITH_REVIEW = "label_with_review"
    NO_LABEL = "no_label"


HIGH_CONFIDENCE_THRESHOLD = 0.95
MEDIUM_CONFIDENCE_THRESHOLD = 0.80


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
