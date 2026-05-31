import pytest

from gmail_classifier.classifier import Action
from gmail_classifier.cross_validation import PredictionResult
from gmail_classifier.evaluation import (
    precision_at_threshold,
    coverage_at_threshold,
    compute_metrics_table,
)


def _result(true, pred, confidence):
    """Helper to create a PredictionResult."""
    action = Action.LABEL if confidence >= 0.95 else (
        Action.LABEL_WITH_REVIEW if confidence >= 0.80 else Action.NO_LABEL
    )
    return PredictionResult(
        true_label=true,
        predicted_label=pred,
        confidence=confidence,
        action=action,
    )


@pytest.fixture
def sample_results():
    """10 results with varying confidence and correctness."""
    return [
        _result("A", "A", 0.99),  # correct, high confidence
        _result("A", "A", 0.97),  # correct, high confidence
        _result("B", "B", 0.96),  # correct, high confidence
        _result("A", "B", 0.93),  # wrong, medium confidence
        _result("B", "B", 0.88),  # correct, medium confidence
        _result("A", "A", 0.85),  # correct, medium confidence
        _result("B", "A", 0.82),  # wrong, medium confidence
        _result("A", "A", 0.70),  # correct, low confidence
        _result("B", "B", 0.60),  # correct, low confidence
        _result("A", "", 0.0),    # no prediction
    ]


def test_precision_at_high_threshold(sample_results):
    """At 0.95, only the 3 high-confidence correct predictions are labeled."""
    precision = precision_at_threshold(sample_results, 0.95)
    # 3 predictions above 0.95, all correct
    assert precision == pytest.approx(1.0)


def test_precision_at_medium_threshold(sample_results):
    """At 0.80, predictions above 0.80 are included (some wrong)."""
    precision = precision_at_threshold(sample_results, 0.80)
    # 7 predictions above 0.80: 5 correct, 2 wrong
    assert precision == pytest.approx(5 / 7)


def test_precision_at_zero_threshold(sample_results):
    """At 0.0, all predictions with confidence > 0 are included."""
    precision = precision_at_threshold(sample_results, 0.0)
    # 9 predictions with confidence > 0: 7 correct, 2 wrong
    assert precision == pytest.approx(7 / 9)


def test_precision_no_predictions_above_threshold(sample_results):
    """When no predictions are above threshold, precision is undefined (return 1.0)."""
    precision = precision_at_threshold(sample_results, 1.01)
    assert precision == 1.0


def test_coverage_at_high_threshold(sample_results):
    """At 0.95, only 3 out of 10 examples are covered."""
    coverage = coverage_at_threshold(sample_results, 0.95)
    assert coverage == pytest.approx(3 / 10)


def test_coverage_at_medium_threshold(sample_results):
    """At 0.80, 7 out of 10 examples are covered."""
    coverage = coverage_at_threshold(sample_results, 0.80)
    assert coverage == pytest.approx(7 / 10)


def test_coverage_at_zero_threshold(sample_results):
    """At 0.0, 9 out of 10 are covered (one has confidence=0)."""
    coverage = coverage_at_threshold(sample_results, 0.0)
    assert coverage == pytest.approx(9 / 10)


def test_compute_metrics_table(sample_results):
    """compute_metrics_table returns rows with (threshold, precision, coverage)."""
    thresholds = [0.95, 0.80, 0.50]
    table = compute_metrics_table(sample_results, thresholds)
    assert len(table) == 3
    # Each entry is (threshold, precision, coverage)
    assert table[0][0] == 0.95
    assert table[0][1] == pytest.approx(1.0)
    assert table[0][2] == pytest.approx(3 / 10)
    assert table[1][0] == 0.80
    assert table[1][1] == pytest.approx(5 / 7)
    assert table[1][2] == pytest.approx(7 / 10)
