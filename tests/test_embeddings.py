import numpy as np
import pytest

from gmail_classifier.embeddings import Embedder


@pytest.fixture(scope="module")
def embedder():
    """Load the model once for all tests in this module."""
    return Embedder()


@pytest.mark.slow
def test_embed_single_text(embedder):
    vec = embedder.embed("Hello world")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    # Should be unit-normalized
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.slow
def test_embed_batch(embedder):
    texts = ["Hello world", "Goodbye world", "Another sentence"]
    vecs = embedder.embed_batch(texts)
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (3, 384)
    # Each vector should be unit-normalized
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


@pytest.mark.slow
def test_similar_texts_have_higher_similarity(embedder):
    v1 = embedder.embed("Python programming tutorial")
    v2 = embedder.embed("Learn Python coding")
    v3 = embedder.embed("Best pizza recipe in Italy")
    # v1 and v2 should be more similar than v1 and v3
    sim_related = np.dot(v1, v2)
    sim_unrelated = np.dot(v1, v3)
    assert sim_related > sim_unrelated
