"""Tests for reacting to label changes in history events."""
from unittest.mock import MagicMock
import numpy as np

from gmail_classifier.label_change_handler import process_label_changes
from gmail_classifier.models import HistoryEvent, Message
from gmail_classifier.storage import MessageStore


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


def test_label_added_updates_training(tmp_path):
    """When a user label is added, message goes into training DB."""
    events = [
        HistoryEvent(type="labelsAdded", message_id="msg1", label_ids=["Label_1"]),
    ]

    client = MagicMock()
    client.get_message.return_value = _make_raw_message("msg1", label_ids=["Label_1"])

    training_store = MessageStore(str(tmp_path / "training.db"))
    skip_store = MessageStore(str(tmp_path / "skip.db"))

    # Pre-populate skip store with msg1
    skip_store.save_message(Message(id="msg1", subject="Test", from_address="a@x.com", labels=[]))

    label_id_to_name = {"Label_1": "Tech"}
    user_label_ids = {"Label_1"}

    process_label_changes(
        events=events,
        client=client,
        training_store=training_store,
        skip_store=skip_store,
        label_id_to_name=label_id_to_name,
        user_label_ids=user_label_ids,
        excluded_labels=set(),
    )

    # Message should be in training
    training_msgs = training_store.load_all()
    assert len(training_msgs) == 1
    assert training_msgs[0].id == "msg1"
    assert training_msgs[0].labels == ["Tech"]

    # Message should be removed from skip
    assert not skip_store.has_message("msg1")

    training_store.close()
    skip_store.close()


def test_label_removed_moves_to_skip(tmp_path):
    """When a user label is removed and message has no other user labels, it goes to skip."""
    events = [
        HistoryEvent(type="labelsRemoved", message_id="msg1", label_ids=["Label_1"]),
    ]

    client = MagicMock()
    # Message now only has INBOX (no user labels left)
    client.get_message.return_value = _make_raw_message("msg1", label_ids=["INBOX"])

    training_store = MessageStore(str(tmp_path / "training.db"))
    skip_store = MessageStore(str(tmp_path / "skip.db"))

    # Pre-populate training with msg1
    training_store.save_message(
        Message(id="msg1", subject="Test", from_address="a@x.com", labels=["Tech"])
    )

    label_id_to_name = {"Label_1": "Tech"}
    user_label_ids = {"Label_1"}

    process_label_changes(
        events=events,
        client=client,
        training_store=training_store,
        skip_store=skip_store,
        label_id_to_name=label_id_to_name,
        user_label_ids=user_label_ids,
        excluded_labels=set(),
    )

    # Message should be removed from training
    assert not training_store.has_message("msg1")

    # Message should be in skip
    skip_msgs = skip_store.load_all()
    assert len(skip_msgs) == 1
    assert skip_msgs[0].id == "msg1"

    training_store.close()
    skip_store.close()


def test_label_moved_updates_training_not_skip(tmp_path):
    """When label A removed and label B added on same message, update training only."""
    events = [
        HistoryEvent(type="labelsRemoved", message_id="msg1", label_ids=["Label_1"]),
        HistoryEvent(type="labelsAdded", message_id="msg1", label_ids=["Label_2"]),
    ]

    client = MagicMock()
    # Message now has Label_2
    client.get_message.return_value = _make_raw_message("msg1", label_ids=["Label_2"])

    training_store = MessageStore(str(tmp_path / "training.db"))
    skip_store = MessageStore(str(tmp_path / "skip.db"))

    # Pre-populate training with msg1 under Tech
    training_store.save_message(
        Message(id="msg1", subject="Test", from_address="a@x.com", labels=["Tech"])
    )

    label_id_to_name = {"Label_1": "Tech", "Label_2": "Travel"}
    user_label_ids = {"Label_1", "Label_2"}

    process_label_changes(
        events=events,
        client=client,
        training_store=training_store,
        skip_store=skip_store,
        label_id_to_name=label_id_to_name,
        user_label_ids=user_label_ids,
        excluded_labels=set(),
    )

    # Message should be in training under Travel
    training_msgs = training_store.load_all()
    assert len(training_msgs) == 1
    assert training_msgs[0].labels == ["Travel"]

    # Should NOT be in skip
    assert not skip_store.has_message("msg1")

    training_store.close()
    skip_store.close()


def test_excluded_label_changes_ignored(tmp_path):
    """Changes to excluded labels should be ignored."""
    events = [
        HistoryEvent(type="labelsAdded", message_id="msg1", label_ids=["Label_XLC"]),
    ]

    client = MagicMock()
    training_store = MessageStore(str(tmp_path / "training.db"))
    skip_store = MessageStore(str(tmp_path / "skip.db"))

    label_id_to_name = {"Label_XLC": "XLC"}
    user_label_ids = {"Label_XLC"}

    process_label_changes(
        events=events,
        client=client,
        training_store=training_store,
        skip_store=skip_store,
        label_id_to_name=label_id_to_name,
        user_label_ids=user_label_ids,
        excluded_labels={"XLC"},
    )

    # No fetch, no store changes
    client.get_message.assert_not_called()
    assert training_store.load_all() == []
    assert skip_store.load_all() == []

    training_store.close()
    skip_store.close()
