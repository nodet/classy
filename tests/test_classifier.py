import numpy as np

from gmail_classifier.classifier import cosine_similarity


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


import pytest
