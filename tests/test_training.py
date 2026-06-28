import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from gmail_classifier.embedding_cache import EmbeddingCache
from gmail_classifier.models import Message
from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.training import (
    assemble_training_index,
    build_training_data,
    exclude_labeled_from_skip,
    prepare_texts,
)


class _FakeEmbedder:
    """Deterministic embedder: a distinct unit-ish vector per id, no model load."""

    def __init__(self, dim=4):
        self.dim = dim

    def embed(self, text):
        return np.ones(self.dim, dtype=np.float32)


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


def test_build_training_data_skips_text_prep_for_cache_hits(tmp_path):
    """Cache hits must not have their body parsed — that BeautifulSoup work
    is the startup cost we defer to misses only."""
    messages = [
        _make_message("1", "Hello", "a@b.com", body_html="<p>hi</p>", label="Tech"),
        _make_message("2", "World", "c@d.com", body_html="<p>yo</p>", label="Travel"),
    ]

    cache = EmbeddingCache(str(tmp_path / "embeddings.db"))
    cache.put("1", np.random.randn(384).astype(np.float32))  # hit
    # "2" is a miss

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = np.random.randn(384).astype(np.float32)

    with patch("gmail_classifier.training._message_text", wraps=lambda m: "x") as mt:
        build_training_data(messages, embedder=mock_embedder, cache=cache)

    prepped_ids = {call.args[0].id for call in mt.call_args_list}
    assert prepped_ids == {"2"}  # only the miss was text-prepped


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

    # Mock embedder returns a known vector per miss, one call each
    mock_embedder = MagicMock()
    miss_vecs = np.random.randn(2, 384).astype(np.float32)
    mock_embedder.embed.side_effect = list(miss_vecs)

    embeddings, labels, ids = build_training_data(messages, embedder=mock_embedder, cache=cache)

    # Only the 2 uncached messages should be embedded, one at a time
    assert mock_embedder.embed.call_count == 2
    mock_embedder.embed_batch.assert_not_called()

    # Verify output, in original order
    assert embeddings.shape == (3, 384)
    np.testing.assert_allclose(embeddings[0], vec1, atol=1e-7)
    np.testing.assert_allclose(embeddings[1], miss_vecs[0], atol=1e-7)
    np.testing.assert_allclose(embeddings[2], miss_vecs[1], atol=1e-7)

    # Verify new embeddings were stored in cache
    assert cache.get("2") is not None
    assert cache.get("3") is not None


def test_build_training_data_duplicate_id_not_counted_as_miss(tmp_path):
    """A message in both the training and skip stores appears twice in the
    input. If it's cached, it must not be embedded -- the miss count is driven
    by positions whose id is absent, not by len(ids) - len(cached), which a
    duplicate would inflate."""
    messages = [
        _make_message("1", "Hello", "a@b.com", label="Tech"),
        _make_message("1", "Hello", "a@b.com", label="__skip__"),  # same id, dup
        _make_message("2", "World", "c@d.com", label="Travel"),
    ]

    cache = EmbeddingCache(str(tmp_path / "embeddings.db"))
    cache.put("1", np.random.randn(384).astype(np.float32))
    cache.put("2", np.random.randn(384).astype(np.float32))

    mock_embedder = MagicMock()
    embeddings, labels, ids = build_training_data(
        messages, embedder=mock_embedder, cache=cache)

    # Both unique ids are cached, so nothing is embedded despite the duplicate.
    mock_embedder.embed.assert_not_called()
    mock_embedder.embed_batch.assert_not_called()
    # The duplicate row is preserved in output (one per input message).
    assert ids == ["1", "1", "2"]
    assert embeddings.shape == (3, 384)


def test_exclude_labeled_from_skip_drops_labeled_ids():
    """A skip message whose id also appears in training is dropped: labeled
    wins over skip."""
    train = [
        _make_message("1", "A", "a@b.com", label="Tech"),
        _make_message("2", "B", "c@d.com", label="Travel"),
    ]
    skip = [
        _make_message("1", "A", "a@b.com", label="Tech"),   # also labeled -> drop
        _make_message("9", "Z", "z@z.com", label="Tech"),   # inbox-only -> keep
    ]
    result = exclude_labeled_from_skip(skip, train)
    assert [m.id for m in result] == ["9"]


def test_exclude_labeled_from_skip_no_overlap_is_identity():
    train = [_make_message("1", "A", "a@b.com", label="Tech")]
    skip = [_make_message("9", "Z", "z@z.com", label="Tech")]
    result = exclude_labeled_from_skip(skip, train)
    assert [m.id for m in result] == ["9"]
    # Inputs are not mutated.
    assert len(skip) == 1


def test_exclude_labeled_from_skip_prevents_orphaned_index_row():
    """After dedup, the TrainingIndex has one row per id, so a later
    correction reaches the (single) row -- no orphaned, uncorrectable vote."""
    from gmail_classifier.training_index import TrainingIndex

    train = [_make_message("1", "A", "a@b.com", label="Tech")]
    skip = [_make_message("1", "A", "a@b.com", label="__skip__")]  # same id

    skip = exclude_labeled_from_skip(skip, train)
    all_msgs = train + skip  # skip is now empty

    embeddings = np.random.randn(len(all_msgs), 4).astype(np.float32)
    labels = [m.labels[0] for m in all_msgs]
    ids = [m.id for m in all_msgs]
    index = TrainingIndex(embeddings, labels, ids)

    assert len(index) == 1
    assert index.labels == ["Tech"]
    # A correction on id "1" updates the one row rather than an orphan.
    index.add("1", np.random.randn(4).astype(np.float32), "Travel")
    assert index.labels == ["Travel"]
    assert len(index) == 1


def _assemble(train, skip, excluded, tmp_path):
    cache = EmbeddingCache(str(tmp_path / "embeddings.db"))
    try:
        return assemble_training_index(
            train, skip, excluded=excluded,
            embedder=_FakeEmbedder(), cache=cache,
        )
    finally:
        cache.close()


def test_assemble_excludes_configured_labels(tmp_path):
    """Training messages whose first label is excluded are dropped from the index."""
    train = [
        _make_message("1", "A", "a@b.com", label="Tech"),
        _make_message("2", "B", "c@d.com", label="XLC"),  # excluded
    ]
    index, skip_ids, stats = _assemble(train, [], {"XLC"}, tmp_path)
    assert stats.n_train == 1
    assert index.labels == ["Tech"]
    assert "2" not in index


def test_assemble_skip_ids_retains_every_sampled_id(tmp_path):
    """skip_ids keeps ALL sampled inbox ids -- including ones also labeled --
    so the live loop won't re-classify an already-seen message, even though the
    labeled one is dropped from the training votes."""
    train = [_make_message("1", "A", "a@b.com", label="Tech")]
    skip = [
        _make_message("1", "A", "a@b.com", label="Tech"),  # also labeled
        _make_message("9", "Z", "z@z.com", label="Tech"),  # inbox-only
    ]
    index, skip_ids, stats = _assemble(train, skip, set(), tmp_path)
    assert skip_ids == {"1", "9"}


def test_assemble_drops_labeled_from_skip_votes(tmp_path):
    """The overlap id is kept as a labeled example (not a __skip__ vote), and
    n_dropped reports how many were dropped."""
    train = [_make_message("1", "A", "a@b.com", label="Tech")]
    skip = [
        _make_message("1", "A", "a@b.com", label="Tech"),  # dropped from skip
        _make_message("9", "Z", "z@z.com", label="Tech"),  # kept as skip
    ]
    index, skip_ids, stats = _assemble(train, skip, set(), tmp_path)
    assert stats.n_train == 1
    assert stats.n_skip == 1
    assert stats.n_dropped == 1
    # id "1" carries its real label; only "9" became a __skip__ vote.
    assert index.labels == ["Tech", SKIP_LABEL]


def test_assemble_no_overlap_reports_zero_dropped(tmp_path):
    train = [_make_message("1", "A", "a@b.com", label="Tech")]
    skip = [_make_message("9", "Z", "z@z.com", label="Tech")]
    index, skip_ids, stats = _assemble(train, skip, set(), tmp_path)
    assert stats.n_dropped == 0


def test_assemble_empty_skip(tmp_path):
    """No skip store: index is the training set, skip_ids is empty."""
    train = [
        _make_message("1", "A", "a@b.com", label="Tech"),
        _make_message("2", "B", "c@d.com", label="Travel"),
    ]
    index, skip_ids, stats = _assemble(train, [], set(), tmp_path)
    assert stats.n_train == 2
    assert stats.n_skip == 0
    assert skip_ids == set()
    assert index.labels == ["Tech", "Travel"]


def test_assemble_index_has_one_row_per_id(tmp_path):
    """End-to-end: after dedup, an overlapping id has a single, correctable row
    -- no orphaned vote from a message living in both stores."""
    train = [_make_message("1", "A", "a@b.com", label="Tech")]
    skip = [_make_message("1", "A", "a@b.com", label="Tech")]  # same id
    index, skip_ids, stats = _assemble(train, skip, set(), tmp_path)
    assert len(index) == 1
    index.add("1", np.ones(4, dtype=np.float32), "Travel")
    assert index.labels == ["Travel"]
    assert len(index) == 1
