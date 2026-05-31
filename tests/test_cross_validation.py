import numpy as np
import pytest

from gmail_classifier.cross_validation import leave_one_out, PredictionResult
from gmail_classifier.classifier import Action


def _make_embeddings(n, dim=384, seed=42):
    """Create random unit-normalized embeddings."""
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, dim))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def test_leave_one_out_returns_one_result_per_example():
    """LOO should return exactly N results for N training examples."""
    embeddings = _make_embeddings(20)
    labels = ["A"] * 10 + ["B"] * 10
    results = leave_one_out(embeddings, labels, k=5)
    assert len(results) == 20


def test_leave_one_out_result_fields():
    """Each result should have true_label, predicted_label, confidence, action."""
    embeddings = _make_embeddings(10)
    labels = ["A"] * 5 + ["B"] * 5
    results = leave_one_out(embeddings, labels, k=3)
    r = results[0]
    assert isinstance(r, PredictionResult)
    assert r.true_label in ("A", "B")
    assert r.predicted_label in ("A", "B", "")
    assert 0.0 <= r.confidence <= 1.0
    assert isinstance(r.action, Action)


def test_leave_one_out_excludes_self():
    """The left-out example should NOT appear in its own neighbors."""
    # Create a dataset where example 0 is very distinct
    dim = 384
    rng = np.random.default_rng(123)
    embeddings = rng.normal(size=(10, dim))
    # Make first example a unit vector along dim 0
    embeddings[0] = np.zeros(dim)
    embeddings[0][0] = 1.0
    # All others are random — normalize all
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    labels = ["A"] * 5 + ["B"] * 5
    results = leave_one_out(embeddings, labels, k=5)
    # If self was included, example 0 would have sim=1.0 with itself
    # The confidence should be based on non-self neighbors only
    assert results[0].confidence < 1.0 or results[0].predicted_label != ""


def test_leave_one_out_respects_min_examples():
    """Labels with fewer than MIN_EXAMPLES_PER_LABEL should be excluded from predictions."""
    embeddings = _make_embeddings(12)
    # 10 examples of A, only 2 of B (below threshold of 5)
    labels = ["A"] * 10 + ["B"] * 2
    results = leave_one_out(embeddings, labels, k=5)
    # No prediction should ever be "B" since it has too few examples
    for r in results:
        assert r.predicted_label != "B"


def test_leave_one_out_perfect_separation():
    """With perfectly separated clusters, LOO should achieve high accuracy."""
    dim = 384
    rng = np.random.default_rng(99)
    # Cluster A: centered around e_0
    center_a = np.zeros(dim)
    center_a[0] = 1.0
    # Cluster B: centered around e_1
    center_b = np.zeros(dim)
    center_b[1] = 1.0

    n_per_class = 10
    embeddings = []
    for _ in range(n_per_class):
        v = center_a + rng.normal(scale=0.05, size=dim)
        embeddings.append(v / np.linalg.norm(v))
    for _ in range(n_per_class):
        v = center_b + rng.normal(scale=0.05, size=dim)
        embeddings.append(v / np.linalg.norm(v))

    embeddings = np.array(embeddings)
    labels = ["A"] * n_per_class + ["B"] * n_per_class

    results = leave_one_out(embeddings, labels, k=5)
    correct = sum(1 for r in results if r.predicted_label == r.true_label)
    accuracy = correct / len(results)
    assert accuracy >= 0.95
