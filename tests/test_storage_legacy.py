"""Tests for the LegacyBackend adapter (three-DB layout behind the seam)."""
from unittest.mock import MagicMock

import numpy as np

from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.models import Message
from gmail_classifier.storage import MessageStore
from gmail_classifier.storage_legacy import LegacyBackend


def _paths(tmp_path):
    return str(tmp_path / "training.db"), str(tmp_path / "inbox_sample.db")


class _FakeEmbedder:
    """Deterministic embedder: distinct unit vectors keyed by id, cached-free."""

    def __init__(self, dim=8):
        self._dim = dim
        self.embed_calls = []

    def embed(self, text):
        self.embed_calls.append(text)
        return np.ones(self._dim, dtype=np.float32)


def test_load_index_builds_from_training_and_skip(tmp_path):
    training_db, skip_db = _paths(tmp_path)

    train = MessageStore(training_db)
    train.save_message(Message(id="t1", subject="a", from_address="a@x.com", labels=["Tech"]))
    train.save_message(Message(id="t2", subject="b", from_address="b@x.com", labels=["Tech"]))
    train.close()

    skip = MessageStore(skip_db)
    skip.save_message(Message(id="s1", subject="c", from_address="c@x.com", labels=[]))
    skip.close()

    backend = LegacyBackend(training_db, skip_db, excluded=set())
    loaded = backend.load_index(_FakeEmbedder())

    assert loaded.stats.n_train == 2
    assert loaded.stats.n_skip == 1
    # skip_ids captures every sampled inbox id so the live loop won't re-classify.
    assert loaded.skip_ids == {"s1"}
    assert set(loaded.index.labels) == {"Tech", SKIP_LABEL}
    backend.close()


def test_load_index_missing_skip_db_is_ok(tmp_path):
    training_db, skip_db = _paths(tmp_path)

    train = MessageStore(training_db)
    train.save_message(Message(id="t1", subject="a", from_address="a@x.com", labels=["Tech"]))
    train.close()

    backend = LegacyBackend(training_db, skip_db, excluded=set())
    loaded = backend.load_index(_FakeEmbedder())

    assert loaded.stats.n_train == 1
    assert loaded.stats.n_skip == 0
    assert loaded.skip_ids == set()
    backend.close()


def test_load_index_empty_training_reports_zero(tmp_path):
    training_db, skip_db = _paths(tmp_path)
    backend = LegacyBackend(training_db, skip_db, excluded=set())
    loaded = backend.load_index(_FakeEmbedder())
    assert loaded.stats.n_train == 0
    backend.close()


def test_upsert_label_saves_and_drops_from_skip(tmp_path):
    training_db, skip_db = _paths(tmp_path)

    # msg1 starts in skip.
    skip = MessageStore(skip_db)
    skip.save_message(Message(id="msg1", subject="a", from_address="a@x.com", labels=[]))
    skip.close()

    backend = LegacyBackend(training_db, skip_db, excluded=set())
    msg = Message(id="msg1", subject="a", from_address="a@x.com", labels=["Tech"])
    backend.upsert_label(msg, "Label_1", np.ones(8, dtype=np.float32))
    backend.close()

    train = MessageStore(training_db)
    stored = train.load_all()
    train.close()
    assert len(stored) == 1
    assert stored[0].id == "msg1"
    assert stored[0].labels == ["Tech"]

    skip = MessageStore(skip_db)
    assert not skip.has_message("msg1")
    skip.close()


def test_upsert_skip_moves_from_training(tmp_path):
    training_db, skip_db = _paths(tmp_path)

    train = MessageStore(training_db)
    train.save_message(Message(id="msg1", subject="a", from_address="a@x.com", labels=["Tech"]))
    train.close()

    backend = LegacyBackend(training_db, skip_db, excluded=set())
    msg = Message(id="msg1", subject="a", from_address="a@x.com", labels=["Tech"])
    backend.upsert_skip(msg, np.ones(8, dtype=np.float32))
    backend.close()

    train = MessageStore(training_db)
    assert not train.has_message("msg1")
    train.close()

    skip = MessageStore(skip_db)
    stored = skip.load_all()
    skip.close()
    assert len(stored) == 1
    assert stored[0].id == "msg1"
    assert stored[0].labels == []  # skip examples carry no user label


def test_remove_drops_from_training(tmp_path):
    training_db, skip_db = _paths(tmp_path)

    train = MessageStore(training_db)
    train.save_message(Message(id="msg1", subject="a", from_address="a@x.com", labels=["Tech"]))
    train.close()

    backend = LegacyBackend(training_db, skip_db, excluded=set())
    backend.remove("msg1")
    backend.close()

    train = MessageStore(training_db)
    assert not train.has_message("msg1")
    train.close()


def test_history_cursor_is_process_local(tmp_path):
    """Legacy never persists the cursor: a fresh adapter starts with None,
    preserving today's fresh-watch restart behavior."""
    training_db, skip_db = _paths(tmp_path)

    backend = LegacyBackend(training_db, skip_db, excluded=set())
    assert backend.get_last_processed_history_id() is None
    backend.set_last_processed_history_id("12345")
    assert backend.get_last_processed_history_id() == "12345"
    backend.close()

    # A fresh adapter over the same files has no memory of the cursor.
    fresh = LegacyBackend(training_db, skip_db, excluded=set())
    assert fresh.get_last_processed_history_id() is None
    fresh.close()
