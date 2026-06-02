import numpy as np
import pytest
from unittest.mock import patch, MagicMock

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
