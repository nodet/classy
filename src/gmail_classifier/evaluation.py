from typing import List, Tuple

from gmail_classifier.cross_validation import PredictionResult


def precision_at_threshold(
    results: List[PredictionResult], threshold: float
) -> float:
    """Compute precision for predictions at or above the confidence threshold.

    Precision = correct predictions / total predictions made.
    Returns 1.0 if no predictions are made at this threshold.
    """
    predictions = [
        r for r in results if r.confidence >= threshold and r.predicted_label != ""
    ]
    if not predictions:
        return 1.0
    correct = sum(1 for r in predictions if r.predicted_label == r.true_label)
    return correct / len(predictions)


def coverage_at_threshold(
    results: List[PredictionResult], threshold: float
) -> float:
    """Compute coverage: fraction of examples that receive a prediction at this threshold.

    Coverage = predictions made / total examples.
    """
    if not results:
        return 0.0
    predictions = sum(
        1 for r in results if r.confidence >= threshold and r.predicted_label != ""
    )
    return predictions / len(results)


def compute_metrics_table(
    results: List[PredictionResult],
    thresholds: List[float],
) -> List[Tuple[float, float, float]]:
    """Compute precision and coverage at multiple thresholds.

    Returns list of (threshold, precision, coverage) tuples.
    """
    return [
        (t, precision_at_threshold(results, t), coverage_at_threshold(results, t))
        for t in thresholds
    ]


def per_label_precision(
    results: List[PredictionResult], threshold: float
) -> dict:
    """Compute precision per true label at a given confidence threshold.

    Returns dict of {label: (precision, n_correct, n_total)} for each label
    that has at least one prediction at or above the threshold.
    """
    from collections import defaultdict

    # Group by true label: count how many were predicted (at threshold) and how many correctly
    label_correct = defaultdict(int)
    label_total = defaultdict(int)

    for r in results:
        if r.confidence >= threshold and r.predicted_label != "":
            label_total[r.true_label] += 1
            if r.predicted_label == r.true_label:
                label_correct[r.true_label] += 1

    return {
        label: (
            label_correct[label] / label_total[label],
            label_correct[label],
            label_total[label],
        )
        for label in label_total
    }
