import numpy as np
import pytest

from gmail_classifier.classifier import cosine_similarity, find_neighbors


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
