"""Tests for reacting to label changes in history events."""
from unittest.mock import MagicMock
import numpy as np

from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.label_change_handler import process_label_changes
from gmail_classifier.label_registry import LabelRegistry
from gmail_classifier.models import HistoryEvent, Message
from gmail_classifier.training_index import TrainingIndex

from tests.fakes import FakeBackend


def _make_raw_message(msg_id, subject="Test", label_ids=None):
    return {
        "id": msg_id,
        "labelIds": label_ids or ["INBOX"],
        "payload": {
            "headers": [
                {"name": "From", "value": "sender@example.com"},
                {"name": "Subject", "value": subject},
            ],
            "body": {"data": ""},
            "parts": [],
        },
    }


def test_label_added_updates_training():
    """When a user label is added, message is recorded as a labeled example."""
    events = [
        HistoryEvent(type="labelsAdded", message_id="msg1", label_ids=["Label_1"]),
    ]

    client = MagicMock()
    client.get_message.return_value = _make_raw_message("msg1", label_ids=["Label_1"])

    backend = FakeBackend()
    # Pre-populate skip with msg1
    backend.skipped["msg1"] = Message(id="msg1", subject="Test", from_address="a@x.com", labels=[])

    process_label_changes(
        events=events,
        client=client,
        backend=backend,
        label_id_to_name={"Label_1": "Tech"},
        user_label_ids={"Label_1"},
        excluded_labels=set(),
    )

    # Message should be labeled Tech and dropped from skip.
    assert "msg1" in backend.labeled
    assert backend.labeled["msg1"].labels == ["Tech"]
    assert "msg1" not in backend.skipped


def test_label_removed_moves_to_skip():
    """When a user label is removed and no user labels remain, message goes to skip."""
    events = [
        HistoryEvent(type="labelsRemoved", message_id="msg1", label_ids=["Label_1"]),
    ]

    client = MagicMock()
    # Message now only has INBOX (no user labels left)
    client.get_message.return_value = _make_raw_message("msg1", label_ids=["INBOX"])

    backend = FakeBackend()
    backend.labeled["msg1"] = Message(id="msg1", subject="Test", from_address="a@x.com", labels=["Tech"])

    process_label_changes(
        events=events,
        client=client,
        backend=backend,
        label_id_to_name={"Label_1": "Tech"},
        user_label_ids={"Label_1"},
        excluded_labels=set(),
    )

    # Message moved from labeled to skip.
    assert "msg1" not in backend.labeled
    assert "msg1" in backend.skipped


def test_label_moved_updates_training_not_skip():
    """When label A removed and label B added on same message, update label only."""
    events = [
        HistoryEvent(type="labelsRemoved", message_id="msg1", label_ids=["Label_1"]),
        HistoryEvent(type="labelsAdded", message_id="msg1", label_ids=["Label_2"]),
    ]

    client = MagicMock()
    client.get_message.return_value = _make_raw_message("msg1", label_ids=["Label_2"])

    backend = FakeBackend()
    backend.labeled["msg1"] = Message(id="msg1", subject="Test", from_address="a@x.com", labels=["Tech"])

    process_label_changes(
        events=events,
        client=client,
        backend=backend,
        label_id_to_name={"Label_1": "Tech", "Label_2": "Travel"},
        user_label_ids={"Label_1", "Label_2"},
        excluded_labels=set(),
    )

    # Message re-labeled Travel, never moved to skip.
    assert backend.labeled["msg1"].labels == ["Travel"]
    assert "msg1" not in backend.skipped


def test_excluded_label_changes_ignored():
    """Changes to excluded labels should be ignored."""
    events = [
        HistoryEvent(type="labelsAdded", message_id="msg1", label_ids=["Label_XLC"]),
    ]

    client = MagicMock()
    backend = FakeBackend()

    process_label_changes(
        events=events,
        client=client,
        backend=backend,
        label_id_to_name={"Label_XLC": "XLC"},
        user_label_ids={"Label_XLC"},
        excluded_labels={"XLC"},
    )

    # No fetch, no store changes
    client.get_message.assert_not_called()
    assert backend.labeled == {}
    assert backend.skipped == {}


# --- Tests for in-memory index updates ---


def test_label_added_updates_in_memory_index():
    """When a label is added and index is provided, index gets updated."""
    events = [
        HistoryEvent(type="labelsAdded", message_id="msg1", label_ids=["Label_1"]),
    ]

    client = MagicMock()
    client.get_message.return_value = _make_raw_message("msg1", label_ids=["Label_1"])

    backend = FakeBackend()

    # Start with an empty index (just one dummy entry)
    index = TrainingIndex(
        np.random.randn(1, 384).astype(np.float32),
        ["dummy"],
        ["dummy_id"],
    )

    embedder = MagicMock()
    embedder.embed.return_value = np.ones(384, dtype=np.float32)

    process_label_changes(
        events=events,
        client=client,
        backend=backend,
        label_id_to_name={"Label_1": "Tech"},
        user_label_ids={"Label_1"},
        excluded_labels=set(),
        index=index,
        embedder=embedder,
    )

    assert len(index) == 2
    assert "msg1" in index
    idx = index._id_to_idx["msg1"]
    assert index.labels[idx] == "Tech"


def test_label_removed_updates_in_memory_index_to_skip():
    """When a label is removed and no labels left, index entry becomes __skip__."""
    events = [
        HistoryEvent(type="labelsRemoved", message_id="msg1", label_ids=["Label_1"]),
    ]

    client = MagicMock()
    client.get_message.return_value = _make_raw_message("msg1", label_ids=["INBOX"])

    backend = FakeBackend()
    backend.labeled["msg1"] = Message(id="msg1", subject="Test", from_address="a@x.com", labels=["Tech"])

    # Index has msg1 as Tech
    embeddings = np.random.randn(2, 384).astype(np.float32)
    index = TrainingIndex(embeddings, ["Tech", "Travel"], ["msg1", "msg2"])

    embedder = MagicMock()
    embedder.embed.return_value = np.ones(384, dtype=np.float32)

    process_label_changes(
        events=events,
        client=client,
        backend=backend,
        label_id_to_name={"Label_1": "Tech"},
        user_label_ids={"Label_1"},
        excluded_labels=set(),
        index=index,
        embedder=embedder,
    )

    # msg1 should now be __skip__ in the index
    assert "msg1" in index
    idx = index._id_to_idx["msg1"]
    assert index.labels[idx] == SKIP_LABEL


# --- Tests for dynamic label discovery via LabelRegistry ---


def test_unknown_label_triggers_refresh_and_processes():
    """When a label ID is unknown and registry is provided, refresh discovers it."""
    events = [
        HistoryEvent(type="labelsAdded", message_id="msg1", label_ids=["Label_NEW"]),
    ]

    client = MagicMock()
    client.get_message.return_value = _make_raw_message("msg1", label_ids=["Label_NEW"])
    # First call: only L1. After refresh: L1 + Label_NEW.
    client.list_user_labels.side_effect = [
        [("L1", "Tech")],
        [("L1", "Tech"), ("Label_NEW", "Science")],
    ]

    registry = LabelRegistry(client, excluded=set())
    backend = FakeBackend()

    index = TrainingIndex(
        np.random.randn(1, 384).astype(np.float32),
        ["dummy"],
        ["dummy_id"],
    )
    embedder = MagicMock()
    embedder.embed.return_value = np.ones(384, dtype=np.float32)

    process_label_changes(
        events=events,
        client=client,
        backend=backend,
        label_id_to_name={},  # ignored when registry provided
        user_label_ids=set(),
        excluded_labels=set(),
        index=index,
        embedder=embedder,
        registry=registry,
    )

    # Registry should have refreshed and discovered the new label
    assert registry.is_known("Label_NEW")
    assert registry.get_name("Label_NEW") == "Science"

    # Message should be labeled under the new label
    assert backend.labeled["msg1"].labels == ["Science"]

    # In-memory index should have the new entry
    assert "msg1" in index
    idx = index._id_to_idx["msg1"]
    assert index.labels[idx] == "Science"


def test_unknown_label_still_unknown_after_refresh_is_skipped():
    """If label is still unknown after refresh, the event is silently skipped."""
    events = [
        HistoryEvent(type="labelsAdded", message_id="msg1", label_ids=["Label_GHOST"]),
    ]

    client = MagicMock()
    # Refresh still doesn't find it
    client.list_user_labels.return_value = [("L1", "Tech")]

    registry = LabelRegistry(client, excluded=set())
    backend = FakeBackend()

    process_label_changes(
        events=events,
        client=client,
        backend=backend,
        label_id_to_name={},
        user_label_ids=set(),
        excluded_labels=set(),
        registry=registry,
    )

    # No message fetched, nothing stored
    client.get_message.assert_not_called()
    assert backend.labeled == {}


def test_movements_summary_inbox_to_label():
    """Movement summary reports messages moved from inbox to a label."""
    events = [
        HistoryEvent(type="labelsAdded", message_id="msg1", label_ids=["Label_1"]),
        HistoryEvent(type="labelsAdded", message_id="msg2", label_ids=["Label_1"]),
        HistoryEvent(type="labelsAdded", message_id="msg3", label_ids=["Label_2"]),
    ]

    client = MagicMock()
    client.get_message.side_effect = [
        _make_raw_message("msg1", label_ids=["Label_1"]),
        _make_raw_message("msg2", label_ids=["Label_1"]),
        _make_raw_message("msg3", label_ids=["Label_2"]),
    ]

    backend = FakeBackend()

    movements = process_label_changes(
        events=events,
        client=client,
        backend=backend,
        label_id_to_name={"Label_1": "Tech", "Label_2": "Travel"},
        user_label_ids={"Label_1", "Label_2"},
        excluded_labels=set(),
    )

    # Should report 2 from inbox→Tech, 1 from inbox→Travel
    assert sorted(movements) == sorted([
        ("inbox", "Tech", 2),
        ("inbox", "Travel", 1),
    ])


def test_movements_summary_label_to_inbox():
    """Movement summary reports messages moved from a label to inbox (unlabeled)."""
    events = [
        HistoryEvent(type="labelsRemoved", message_id="msg1", label_ids=["Label_1"]),
        HistoryEvent(type="labelsRemoved", message_id="msg2", label_ids=["Label_1"]),
    ]

    client = MagicMock()
    # Both messages now have no user labels
    client.get_message.side_effect = [
        _make_raw_message("msg1", label_ids=["INBOX"]),
        _make_raw_message("msg2", label_ids=["INBOX"]),
    ]

    backend = FakeBackend()
    backend.labeled["msg1"] = Message(id="msg1", subject="A", from_address="a@x.com", labels=["Tech"])
    backend.labeled["msg2"] = Message(id="msg2", subject="B", from_address="b@x.com", labels=["Tech"])

    movements = process_label_changes(
        events=events,
        client=client,
        backend=backend,
        label_id_to_name={"Label_1": "Tech"},
        user_label_ids={"Label_1"},
        excluded_labels=set(),
    )

    assert movements == [("Tech", "inbox", 2)]


def test_movements_summary_label_to_label():
    """Movement summary reports messages moved between labels."""
    events = [
        HistoryEvent(type="labelsRemoved", message_id="msg1", label_ids=["Label_1"]),
        HistoryEvent(type="labelsAdded", message_id="msg1", label_ids=["Label_2"]),
    ]

    client = MagicMock()
    client.get_message.return_value = _make_raw_message("msg1", label_ids=["Label_2"])

    backend = FakeBackend()
    backend.labeled["msg1"] = Message(id="msg1", subject="A", from_address="a@x.com", labels=["Tech"])

    movements = process_label_changes(
        events=events,
        client=client,
        backend=backend,
        label_id_to_name={"Label_1": "Tech", "Label_2": "Travel"},
        user_label_ids={"Label_1", "Label_2"},
        excluded_labels=set(),
    )

    assert movements == [("Tech", "Travel", 1)]


def test_ignore_ids_skips_self_labeled_messages():
    """Messages in ignore_ids are skipped (classifier's own echo)."""
    events = [
        HistoryEvent(type="labelsAdded", message_id="msg1", label_ids=["Label_1"]),
        HistoryEvent(type="labelsAdded", message_id="msg2", label_ids=["Label_1"]),
    ]

    client = MagicMock()
    client.get_message.return_value = _make_raw_message("msg2", label_ids=["Label_1"])

    backend = FakeBackend()
    ignore_ids = {"msg1"}  # msg1 was labeled by the classifier

    movements = process_label_changes(
        events=events,
        client=client,
        backend=backend,
        label_id_to_name={"Label_1": "Tech"},
        user_label_ids={"Label_1"},
        excluded_labels=set(),
        ignore_ids=ignore_ids,
    )

    # Only msg2 should be processed (msg1 ignored)
    assert list(backend.labeled) == ["msg2"]
    assert movements == [("inbox", "Tech", 1)]

    # msg1 should be removed from ignore_ids (so future corrections work)
    assert "msg1" not in ignore_ids


def test_ignore_ids_allows_subsequent_user_correction():
    """After being ignored once, the same message ID can be processed (user correction)."""
    # First call: classifier labeled msg1, echo comes back → ignored
    events1 = [
        HistoryEvent(type="labelsAdded", message_id="msg1", label_ids=["Label_1"]),
    ]

    client = MagicMock()
    backend = FakeBackend()

    label_id_to_name = {"Label_1": "Tech", "Label_2": "Travel"}
    user_label_ids = {"Label_1", "Label_2"}
    ignore_ids = {"msg1"}

    process_label_changes(
        events=events1,
        client=client,
        backend=backend,
        label_id_to_name=label_id_to_name,
        user_label_ids=user_label_ids,
        excluded_labels=set(),
        ignore_ids=ignore_ids,
    )

    # msg1 consumed from ignore_ids
    assert "msg1" not in ignore_ids

    # Second call: user corrects msg1 from Tech to Travel
    events2 = [
        HistoryEvent(type="labelsRemoved", message_id="msg1", label_ids=["Label_1"]),
        HistoryEvent(type="labelsAdded", message_id="msg1", label_ids=["Label_2"]),
    ]
    client.get_message.return_value = _make_raw_message("msg1", label_ids=["Label_2"])

    movements = process_label_changes(
        events=events2,
        client=client,
        backend=backend,
        label_id_to_name=label_id_to_name,
        user_label_ids=user_label_ids,
        excluded_labels=set(),
        ignore_ids=ignore_ids,
    )

    # Should be processed now (user correction)
    assert backend.labeled["msg1"].labels == ["Travel"]
    assert movements == [("Tech", "Travel", 1)]


def test_new_excluded_label_is_ignored():
    """A newly discovered label that's in the excluded set is not processed."""
    events = [
        HistoryEvent(type="labelsAdded", message_id="msg1", label_ids=["Label_XLZ"]),
    ]

    client = MagicMock()
    client.list_user_labels.side_effect = [
        [("L1", "Tech")],
        [("L1", "Tech"), ("Label_XLZ", "XLZ")],
    ]

    registry = LabelRegistry(client, excluded={"XLZ"})
    backend = FakeBackend()

    process_label_changes(
        events=events,
        client=client,
        backend=backend,
        label_id_to_name={},
        user_label_ids=set(),
        excluded_labels=set(),
        registry=registry,
    )

    # Excluded label: no message fetch, no storage
    client.get_message.assert_not_called()
    assert backend.labeled == {}
