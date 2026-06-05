import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from gmail_classifier.embedding_cache import EmbeddingCache
from gmail_classifier.models import Message
from gmail_classifier.training import build_training_data, prepare_texts


def _make_message(id, subject, from_addr, body_html="", label="Tech", list_id=""):
    return Message(
        id=id,
        subject=subject,
        from_address=from_addr,
        from_name="",
        body_html=body_html,
        labels=[label],
        list_id=list_id,
        date="",
    )


def test_prepare_texts_builds_representations():
    """prepare_texts returns one text string per message using build_text_representation."""
    messages = [
        _make_message("1", "Hello", "a@b.com", "<p>Body one</p>"),
        _make_message("2", "World", "c@d.com", "<p>Body two</p>"),
    ]
    texts, labels, ids = prepare_texts(messages)
    assert len(texts) == 2
    assert len(labels) == 2
    assert ids == ["1", "2"]
    # Text should contain subject and sender
    assert "Hello" in texts[0]
    assert "a@b.com" in texts[0]
    assert "Body one" in texts[0]
    assert labels[0] == "Tech"


def test_prepare_texts_uses_first_label():
    """When a message has multiple labels, use the first one."""
    msg = Message(
        id="1", subject="Sub", from_address="x@y.com",
        from_name="", body_html="", labels=["Travel", "Tech"],
        list_id="", date="",
    )
    texts, labels, ids = prepare_texts([msg])
    assert labels[0] == "Travel"


def test_prepare_texts_skips_messages_without_labels():
    """Messages with no labels are excluded from training data."""
    messages = [
        _make_message("1", "Good", "a@b.com", label="Tech"),
        Message(id="2", subject="No label", from_address="x@y.com",
                from_name="", body_html="", labels=[], list_id="", date=""),
    ]
    texts, labels, ids = prepare_texts(messages)
    assert len(texts) == 1
    assert labels[0] == "Tech"
    assert ids == ["1"]


@pytest.mark.slow
def test_build_training_data_returns_embeddings_and_labels():
    """build_training_data returns (embeddings array, labels list) from messages."""
    messages = [
        _make_message("1", "Python tutorial", "dev@py.org", "<p>Learn Python</p>", "Tech"),
        _make_message("2", "Flight to Paris", "air@fly.com", "<p>Your booking</p>", "Travel"),
        _make_message("3", "New framework", "news@dev.io", "<p>Check this out</p>", "Tech"),
    ]
    embeddings, labels, ids = build_training_data(messages)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 384)
    assert labels == ["Tech", "Travel", "Tech"]
    assert ids == ["1", "2", "3"]
    # Embeddings should be normalized
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_build_training_data_uses_cache(tmp_path):
    """build_training_data returns cached embeddings without calling embedder."""
    messages = [
        _make_message("1", "Hello", "a@b.com", label="Tech"),
        _make_message("2", "World", "c@d.com", label="Travel"),
    ]

    # Pre-populate cache with known vectors
    cache = EmbeddingCache(str(tmp_path / "embeddings.db"))
    vec1 = np.random.randn(384).astype(np.float32)
    vec2 = np.random.randn(384).astype(np.float32)
    cache.put("1", vec1)
    cache.put("2", vec2)

    # Use a mock embedder that should NOT be called
    mock_embedder = MagicMock()

    embeddings, labels, ids = build_training_data(messages, embedder=mock_embedder, cache=cache)

    mock_embedder.embed_batch.assert_not_called()
    assert embeddings.shape == (2, 384)
    np.testing.assert_allclose(embeddings[0], vec1, atol=1e-7)
    np.testing.assert_allclose(embeddings[1], vec2, atol=1e-7)
    assert labels == ["Tech", "Travel"]


def test_build_training_data_cache_partial_miss(tmp_path):
    """build_training_data embeds only uncached messages and stores them."""
    messages = [
        _make_message("1", "Hello", "a@b.com", label="Tech"),
        _make_message("2", "World", "c@d.com", label="Travel"),
        _make_message("3", "Bye", "e@f.com", label="Finance"),
    ]

    cache = EmbeddingCache(str(tmp_path / "embeddings.db"))
    vec1 = np.random.randn(384).astype(np.float32)
    cache.put("1", vec1)

    # Mock embedder returns known vectors for the 2 misses
    mock_embedder = MagicMock()
    miss_vecs = np.random.randn(2, 384).astype(np.float32)
    mock_embedder.embed_batch.return_value = miss_vecs

    embeddings, labels, ids = build_training_data(messages, embedder=mock_embedder, cache=cache)

    # Only uncached messages should be embedded
    mock_embedder.embed_batch.assert_called_once()
    call_texts = mock_embedder.embed_batch.call_args[0][0]
    assert len(call_texts) == 2

    # Verify output
    assert embeddings.shape == (3, 384)
    np.testing.assert_allclose(embeddings[0], vec1, atol=1e-7)
    np.testing.assert_allclose(embeddings[1], miss_vecs[0], atol=1e-7)
    np.testing.assert_allclose(embeddings[2], miss_vecs[1], atol=1e-7)

    # Verify new embeddings were stored in cache
    assert cache.get("2") is not None
    assert cache.get("3") is not None
