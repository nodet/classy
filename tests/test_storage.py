import os
import tempfile

import pytest

from gmail_classifier.models import Message
from gmail_classifier.storage import MessageStore


@pytest.fixture
def store(tmp_path):
    """Create a temporary MessageStore."""
    db_path = tmp_path / "test.db"
    return MessageStore(str(db_path))


def _make_message(id="msg1", labels=None, **kwargs):
    defaults = dict(
        subject="Test subject",
        from_name="Alice",
        from_address="alice@example.com",
        body_html="<p>body</p>",
        list_id="",
        date="2025-01-15",
    )
    defaults.update(kwargs)
    return Message(id=id, labels=labels or [], **defaults)


def test_storage_save_and_load_message(store):
    msg = _make_message(labels=["Tech"])
    store.save_message(msg)
    messages = store.load_all()
    assert len(messages) == 1
    assert messages[0].id == "msg1"
    assert messages[0].subject == "Test subject"
    assert messages[0].from_name == "Alice"
    assert messages[0].labels == ["Tech"]


def test_storage_save_multiple_messages(store):
    store.save_message(_make_message(id="m1", labels=["Tech"]))
    store.save_message(_make_message(id="m2", labels=["Travel"]))
    store.save_message(_make_message(id="m3", labels=["News"]))
    assert len(store.load_all()) == 3


def test_storage_load_by_label(store):
    store.save_message(_make_message(id="m1", labels=["Tech"]))
    store.save_message(_make_message(id="m2", labels=["Travel"]))
    store.save_message(_make_message(id="m3", labels=["Tech"]))
    tech_msgs = store.load_by_label("Tech")
    assert len(tech_msgs) == 2
    assert all("Tech" in m.labels for m in tech_msgs)


def test_storage_dedup_by_message_id(store):
    store.save_message(_make_message(id="m1", labels=["Tech"]))
    store.save_message(_make_message(id="m1", labels=["Tech", "Updated"]))
    messages = store.load_all()
    assert len(messages) == 1


def test_storage_message_with_multiple_labels(store):
    store.save_message(_make_message(id="m1", labels=["Tech", "Newsletters"]))
    assert len(store.load_by_label("Tech")) == 1
    assert len(store.load_by_label("Newsletters")) == 1


def test_storage_persists_to_disk(tmp_path):
    db_path = str(tmp_path / "persist.db")
    store1 = MessageStore(db_path)
    store1.save_message(_make_message(id="m1", labels=["Tech"]))
    store1.close()

    store2 = MessageStore(db_path)
    messages = store2.load_all()
    store2.close()
    assert len(messages) == 1
    assert messages[0].id == "m1"
