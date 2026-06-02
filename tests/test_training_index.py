"""Tests for mutable TrainingIndex."""
import numpy as np

from gmail_classifier.training_index import TrainingIndex


def test_create_from_arrays():
    embeddings = np.random.randn(5, 384).astype(np.float32)
    labels = ["Tech", "Tech", "Travel", "__skip__", "__skip__"]
    ids = ["m1", "m2", "m3", "m4", "m5"]

    index = TrainingIndex(embeddings, labels, ids)
    assert len(index) == 5
    assert index.labels == labels
    assert index.embeddings.shape == (5, 384)


def test_add_appends_to_index():
    embeddings = np.random.randn(3, 384).astype(np.float32)
    labels = ["Tech", "Tech", "Travel"]
    ids = ["m1", "m2", "m3"]

    index = TrainingIndex(embeddings, labels, ids)
    new_emb = np.random.randn(384).astype(np.float32)
    index.add("m4", new_emb, "News")

    assert len(index) == 4
    assert index.labels[3] == "News"
    assert np.array_equal(index.embeddings[3], new_emb)


def test_add_replaces_existing_id():
    embeddings = np.random.randn(3, 384).astype(np.float32)
    labels = ["Tech", "Travel", "News"]
    ids = ["m1", "m2", "m3"]

    index = TrainingIndex(embeddings, labels, ids)
    new_emb = np.random.randn(384).astype(np.float32)
    index.add("m2", new_emb, "Gurobi")

    assert len(index) == 3
    # m2 should now have label Gurobi
    idx = index._id_to_idx["m2"]
    assert index.labels[idx] == "Gurobi"
    assert np.array_equal(index.embeddings[idx], new_emb)


def test_remove_deletes_from_index():
    embeddings = np.random.randn(4, 384).astype(np.float32)
    labels = ["Tech", "Travel", "News", "__skip__"]
    ids = ["m1", "m2", "m3", "m4"]

    index = TrainingIndex(embeddings, labels, ids)
    index.remove("m2")

    assert len(index) == 3
    assert "m2" not in index._id_to_idx
    # Remaining IDs should all be accessible
    assert "m1" in index._id_to_idx
    assert "m3" in index._id_to_idx
    assert "m4" in index._id_to_idx


def test_remove_nonexistent_is_noop():
    embeddings = np.random.randn(3, 384).astype(np.float32)
    labels = ["Tech", "Travel", "News"]
    ids = ["m1", "m2", "m3"]

    index = TrainingIndex(embeddings, labels, ids)
    index.remove("m99")
    assert len(index) == 3


def test_contains():
    embeddings = np.random.randn(3, 384).astype(np.float32)
    labels = ["Tech", "Travel", "News"]
    ids = ["m1", "m2", "m3"]

    index = TrainingIndex(embeddings, labels, ids)
    assert "m1" in index
    assert "m99" not in index
