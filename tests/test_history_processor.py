"""Tests for history-based notification processing."""
from unittest.mock import MagicMock, patch
import numpy as np

from gmail_classifier.history_processor import process_history_events
from gmail_classifier.models import HistoryEvent, Message


def _make_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = np.zeros(384)
    return embedder


def _make_raw_message(msg_id, subject="Test", label_ids=None):
    """Create a raw Gmail API message dict."""
    return {
        "id": msg_id,
        "labelIds": label_ids or ["INBOX"],
        "payload": {
            "headers": [
                {"name": "From", "value": f"sender@example.com"},
                {"name": "Subject", "value": subject},
            ],
            "body": {"data": ""},
            "parts": [],
        },
    }


def test_process_new_inbox_message_classifies_it():
    """A new message added to INBOX should be classified."""
    from gmail_classifier.classifier import Action

    events = [
        HistoryEvent(type="messagesAdded", message_id="msg1", label_ids=["INBOX"]),
    ]

    client = MagicMock()
    client.get_message.return_value = _make_raw_message("msg1")

    embedder = _make_embedder()
    train_embeddings = np.zeros((10, 384))
    train_labels = ["Tech"] * 5 + ["__skip__"] * 5

    results = process_history_events(
        events=events,
        client=client,
        embedder=embedder,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        label_name_to_id={"Tech": "Label_1"},
        user_label_ids={"Label_1"},
        excluded_labels=set(),
        skip_ids=set(),
        k=5,
        dry_run=False,
    )

    # Message was fetched and classified
    client.get_message.assert_called_once_with("msg1")
    assert len(results) == 1
    assert results[0]["message_id"] == "msg1"


def test_process_skips_message_already_in_skip_pool():
    """Messages already in the skip pool should not be re-processed."""
    events = [
        HistoryEvent(type="messagesAdded", message_id="msg1", label_ids=["INBOX"]),
    ]

    client = MagicMock()
    embedder = _make_embedder()

    results = process_history_events(
        events=events,
        client=client,
        embedder=embedder,
        train_embeddings=np.zeros((10, 384)),
        train_labels=["Tech"] * 10,
        label_name_to_id={},
        user_label_ids=set(),
        excluded_labels=set(),
        skip_ids={"msg1"},
        k=5,
        dry_run=False,
    )

    client.get_message.assert_not_called()
    assert results == []


def test_process_skips_message_with_existing_user_label():
    """Messages that already have a user label should be skipped."""
    events = [
        HistoryEvent(type="messagesAdded", message_id="msg1",
                     label_ids=["INBOX", "Label_1"]),
    ]

    client = MagicMock()
    client.get_message.return_value = _make_raw_message(
        "msg1", label_ids=["INBOX", "Label_1"]
    )
    embedder = _make_embedder()

    results = process_history_events(
        events=events,
        client=client,
        embedder=embedder,
        train_embeddings=np.zeros((10, 384)),
        train_labels=["Tech"] * 10,
        label_name_to_id={"Tech": "Label_1"},
        user_label_ids={"Label_1"},
        excluded_labels=set(),
        skip_ids=set(),
        k=5,
        dry_run=False,
    )

    assert results == []


def test_process_deduplicates_events():
    """Multiple events for the same message should only classify once."""
    events = [
        HistoryEvent(type="messagesAdded", message_id="msg1", label_ids=["INBOX"]),
        HistoryEvent(type="messagesAdded", message_id="msg1", label_ids=["INBOX"]),
    ]

    client = MagicMock()
    client.get_message.return_value = _make_raw_message("msg1")
    embedder = _make_embedder()

    results = process_history_events(
        events=events,
        client=client,
        embedder=embedder,
        train_embeddings=np.zeros((10, 384)),
        train_labels=["__skip__"] * 10,
        label_name_to_id={},
        user_label_ids=set(),
        excluded_labels=set(),
        skip_ids=set(),
        k=5,
        dry_run=False,
    )

    client.get_message.assert_called_once_with("msg1")


def test_process_ignores_non_inbox_messages():
    """Messages added to non-INBOX labels should not be classified."""
    events = [
        HistoryEvent(type="messagesAdded", message_id="msg1", label_ids=["SENT"]),
    ]

    client = MagicMock()
    embedder = _make_embedder()

    results = process_history_events(
        events=events,
        client=client,
        embedder=embedder,
        train_embeddings=np.zeros((10, 384)),
        train_labels=["Tech"] * 10,
        label_name_to_id={},
        user_label_ids=set(),
        excluded_labels=set(),
        skip_ids=set(),
        k=5,
        dry_run=False,
    )

    client.get_message.assert_not_called()
    assert results == []


def test_process_excluded_label_not_applied():
    """Messages classified into an excluded label should not be labeled."""
    events = [
        HistoryEvent(type="messagesAdded", message_id="msg1", label_ids=["INBOX"]),
    ]

    client = MagicMock()
    client.get_message.return_value = _make_raw_message("msg1")

    embedder = _make_embedder()
    # All neighbors are XLC (excluded)
    train_embeddings = np.zeros((5, 384))
    train_labels = ["XLC"] * 5

    results = process_history_events(
        events=events,
        client=client,
        embedder=embedder,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        label_name_to_id={"XLC": "Label_XLC"},
        user_label_ids={"Label_XLC"},
        excluded_labels={"XLC"},
        skip_ids=set(),
        k=5,
        dry_run=False,
    )

    # Message was classified but label not applied
    client.apply_label.assert_not_called()
