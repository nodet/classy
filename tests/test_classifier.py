import numpy as np
import pytest

from gmail_classifier.classifier import (
    Action,
    SKIP_LABEL,
    aggregate_scores,
    classify,
    compute_confidence,
    cosine_similarity,
    decide_action,
    find_neighbors,
    MIN_EXAMPLES_PER_LABEL,
)


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_known_value():
    # 45 degree angle -> cos(45) = sqrt(2)/2 ≈ 0.7071
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 1.0]) / np.sqrt(2)
    assert cosine_similarity(a, b) == pytest.approx(np.sqrt(2) / 2, abs=1e-7)


def test_knn_finds_k_nearest():
    # 5 training vectors in 3D, query is closest to indices 0, 1, 2
    training = np.array([
        [1.0, 0.0, 0.0],  # idx 0: identical to query
        [0.9, 0.1, 0.0],  # idx 1: very close
        [0.8, 0.2, 0.0],  # idx 2: close
        [0.0, 1.0, 0.0],  # idx 3: orthogonal
        [0.0, 0.0, 1.0],  # idx 4: orthogonal
    ])
    # Normalize
    training = training / np.linalg.norm(training, axis=1, keepdims=True)
    labels = ["A", "A", "B", "C", "C"]
    query = np.array([1.0, 0.0, 0.0])

    neighbors = find_neighbors(query, training, labels, k=3)
    assert len(neighbors) == 3
    # Should be sorted by similarity descending
    assert neighbors[0][0] == pytest.approx(1.0, abs=1e-5)
    assert neighbors[0][1] == "A"
    assert neighbors[1][1] == "A"
    assert neighbors[2][1] == "B"


def test_knn_returns_fewer_if_training_set_smaller_than_k():
    training = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    labels = ["X", "Y"]
    query = np.array([1.0, 0.0])

    neighbors = find_neighbors(query, training, labels, k=5)
    assert len(neighbors) == 2


def test_score_aggregation_single_label():
    neighbors = [(0.9, "Tech"), (0.8, "Tech"), (0.7, "Tech"), (0.6, "Tech"), (0.5, "Tech")]
    scores = aggregate_scores(neighbors)
    assert scores == {"Tech": pytest.approx(3.5)}


def test_score_aggregation_multiple_labels():
    neighbors = [
        (0.9, "Tech"), (0.8, "Tech"), (0.7, "Tech"),
        (0.6, "Travel"), (0.5, "Travel"),
    ]
    scores = aggregate_scores(neighbors)
    assert scores == {"Tech": pytest.approx(2.4), "Travel": pytest.approx(1.1)}


def test_confidence_all_same_label():
    scores = {"Tech": 3.5}
    label, confidence = compute_confidence(scores)
    assert label == "Tech"
    assert confidence == pytest.approx(1.0)


def test_confidence_split_labels():
    scores = {"Tech": 2.4, "Travel": 1.1}
    label, confidence = compute_confidence(scores)
    assert label == "Tech"
    assert confidence == pytest.approx(2.4 / 3.5)


def test_confidence_even_split():
    scores = {"A": 1.0, "B": 1.0}
    label, confidence = compute_confidence(scores)
    assert confidence == pytest.approx(0.5)


def test_confidence_three_labels():
    scores = {"Tech": 2.0, "Travel": 0.8, "News": 0.2}
    label, confidence = compute_confidence(scores)
    assert label == "Tech"
    assert confidence == pytest.approx(2.0 / 3.0)


def test_decision_high_confidence_labels():
    action = decide_action(0.96)
    assert action == Action.LABEL


def test_decision_medium_confidence_labels_with_marker():
    action = decide_action(0.85)
    assert action == Action.LABEL_WITH_REVIEW


def test_decision_low_confidence_no_label():
    action = decide_action(0.70)
    assert action == Action.NO_LABEL


def test_label_with_few_examples_excluded():
    # "Tech" has 20 examples, "Rare" has 3 (below MIN_EXAMPLES_PER_LABEL)
    # Query is closest to "Rare" vectors but Rare should be excluded
    dim = 10
    rng = np.random.default_rng(42)

    # Create "Tech" cluster near [1,0,0,...] and "Rare" cluster near [0,1,0,...]
    tech_vecs = rng.normal(0, 0.1, (20, dim))
    tech_vecs[:, 0] += 1.0
    tech_vecs = tech_vecs / np.linalg.norm(tech_vecs, axis=1, keepdims=True)

    rare_vecs = rng.normal(0, 0.1, (3, dim))
    rare_vecs[:, 1] += 1.0
    rare_vecs = rare_vecs / np.linalg.norm(rare_vecs, axis=1, keepdims=True)

    training = np.vstack([tech_vecs, rare_vecs])
    labels = ["Tech"] * 20 + ["Rare"] * 3

    # Query is very close to "Rare" cluster
    query = np.zeros(dim)
    query[1] = 1.0

    result = classify(query, training, labels, k=5)
    # "Rare" should be excluded due to min examples threshold
    assert result.label != "Rare"


def test_label_at_threshold_included():
    dim = 10
    rng = np.random.default_rng(42)

    # "NewLabel" has exactly MIN_EXAMPLES_PER_LABEL examples
    new_vecs = rng.normal(0, 0.1, (MIN_EXAMPLES_PER_LABEL, dim))
    new_vecs[:, 0] += 1.0
    new_vecs = new_vecs / np.linalg.norm(new_vecs, axis=1, keepdims=True)

    other_vecs = rng.normal(0, 0.1, (20, dim))
    other_vecs[:, 1] += 1.0
    other_vecs = other_vecs / np.linalg.norm(other_vecs, axis=1, keepdims=True)

    training = np.vstack([new_vecs, other_vecs])
    labels = ["NewLabel"] * MIN_EXAMPLES_PER_LABEL + ["Other"] * 20

    # Query is close to "NewLabel" cluster
    query = np.zeros(dim)
    query[0] = 1.0

    result = classify(query, training, labels, k=5)
    assert result.label == "NewLabel"


def test_classify_clear_winner():
    dim = 10
    rng = np.random.default_rng(123)

    # "Tech" cluster tightly packed around [1,0,0,...], 10 examples
    tech_vecs = rng.normal(0, 0.05, (10, dim))
    tech_vecs[:, 0] += 1.0
    tech_vecs = tech_vecs / np.linalg.norm(tech_vecs, axis=1, keepdims=True)

    # "Travel" cluster tightly packed around [0,1,0,...], 10 examples
    travel_vecs = rng.normal(0, 0.05, (10, dim))
    travel_vecs[:, 1] += 1.0
    travel_vecs = travel_vecs / np.linalg.norm(travel_vecs, axis=1, keepdims=True)

    training = np.vstack([tech_vecs, travel_vecs])
    labels = ["Tech"] * 10 + ["Travel"] * 10

    # Query very close to Tech
    query = np.zeros(dim)
    query[0] = 1.0

    result = classify(query, training, labels, k=5)
    assert result.label == "Tech"
    assert result.confidence > 0.9
    assert result.action == Action.LABEL
    assert len(result.neighbors) == 5


def test_classify_ambiguous():
    dim = 10
    rng = np.random.default_rng(99)

    # Two clusters with slight noise, both at similar distance to query
    tech_vecs = rng.normal(0, 0.05, (5, dim))
    tech_vecs[:, 0] += 1.0
    tech_vecs[:, 2] += 0.5
    tech_vecs = tech_vecs / np.linalg.norm(tech_vecs, axis=1, keepdims=True)

    travel_vecs = rng.normal(0, 0.05, (5, dim))
    travel_vecs[:, 1] += 1.0
    travel_vecs[:, 2] += 0.5
    travel_vecs = travel_vecs / np.linalg.norm(travel_vecs, axis=1, keepdims=True)

    training = np.vstack([tech_vecs, travel_vecs])
    labels = ["Tech"] * 5 + ["Travel"] * 5

    # Query along the shared dimension — roughly equidistant to both clusters
    query = np.zeros(dim)
    query[2] = 1.0

    result = classify(query, training, labels, k=10)
    # Both labels should get similar scores -> low confidence
    assert result.confidence < 0.80
    assert result.action == Action.NO_LABEL


def test_classify_empty_training_set():
    query = np.array([1.0, 0.0, 0.0])
    training = np.empty((0, 3))
    labels = []

    result = classify(query, training, labels, k=5)
    assert result.label == ""
    assert result.action == Action.NO_LABEL


def test_classify_skip_wins():
    """When __skip__ gets the highest aggregate score, result is NO_LABEL."""
    dim = 10
    rng = np.random.default_rng(42)

    # Skip cluster near query
    skip_vecs = rng.normal(0, 0.05, (10, dim))
    skip_vecs[:, 0] += 1.0
    skip_vecs = skip_vecs / np.linalg.norm(skip_vecs, axis=1, keepdims=True)

    # Real label cluster far from query
    real_vecs = rng.normal(0, 0.05, (10, dim))
    real_vecs[:, 1] += 1.0
    real_vecs = real_vecs / np.linalg.norm(real_vecs, axis=1, keepdims=True)

    training = np.vstack([skip_vecs, real_vecs])
    labels = [SKIP_LABEL] * 10 + ["Tech"] * 10

    # Query close to skip cluster
    query = np.zeros(dim)
    query[0] = 1.0

    result = classify(query, training, labels, k=5)
    assert result.label == ""
    assert result.action == Action.NO_LABEL


def test_classify_skip_dilutes_confidence():
    """When __skip__ is among neighbors but doesn't win, it dilutes confidence."""
    dim = 10
    rng = np.random.default_rng(77)

    # Real label cluster
    tech_vecs = rng.normal(0, 0.05, (10, dim))
    tech_vecs[:, 0] += 1.0
    tech_vecs[:, 2] += 0.3
    tech_vecs = tech_vecs / np.linalg.norm(tech_vecs, axis=1, keepdims=True)

    # Skip cluster nearby
    skip_vecs = rng.normal(0, 0.05, (10, dim))
    skip_vecs[:, 0] += 0.8
    skip_vecs[:, 2] += 0.6
    skip_vecs = skip_vecs / np.linalg.norm(skip_vecs, axis=1, keepdims=True)

    training = np.vstack([tech_vecs, skip_vecs])
    labels = ["Tech"] * 10 + [SKIP_LABEL] * 10

    # Query between the two clusters
    query = np.zeros(dim)
    query[0] = 1.0
    query[2] = 0.5
    query = query / np.linalg.norm(query)

    result = classify(query, training, labels, k=5)
    # Tech might win but confidence should be reduced by skip presence
    if result.label == "Tech":
        assert result.confidence < 0.95
