import base64
from unittest.mock import MagicMock

from gmail_classifier.fetcher import fetch_messages_for_label
from gmail_classifier.gmail_client import GmailClient
from gmail_classifier.storage import MessageStore


def _b64url(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def _make_raw_message(id, subject="Test", from_val="A <a@x.com>", body="hello", labels=None):
    return {
        "id": id,
        "labelIds": labels or ["Label_1"],
        "payload": {
            "mimeType": "text/html",
            "headers": [
                {"name": "From", "value": from_val},
                {"name": "Subject", "value": subject},
                {"name": "Date", "value": "2025-01-15"},
            ],
            "body": {"data": _b64url(body)},
        },
    }


def test_fetch_and_store_messages_for_label(tmp_path):
    service = MagicMock()
    service.users().messages().list.return_value.execute.return_value = {
        "messages": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}],
    }
    service.users().messages().get.return_value.execute.side_effect = [
        _make_raw_message("m1", subject="First"),
        _make_raw_message("m2", subject="Second"),
        _make_raw_message("m3", subject="Third"),
    ]
    client = GmailClient(service)
    store = MessageStore(str(tmp_path / "test.db"))

    fetch_messages_for_label(client, store, label_id="Label_1", label_name="Tech")

    messages = store.load_all()
    assert len(messages) == 3
    assert messages[0].subject == "First"
    assert "Tech" in messages[0].labels


def test_fetch_skips_already_stored_messages(tmp_path):
    service = MagicMock()
    service.users().messages().list.return_value.execute.return_value = {
        "messages": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}],
    }
    # Only m2 and m3 should be fetched
    service.users().messages().get.return_value.execute.side_effect = [
        _make_raw_message("m2", subject="Second"),
        _make_raw_message("m3", subject="Third"),
    ]

    client = GmailClient(service)
    store = MessageStore(str(tmp_path / "test.db"))

    # Pre-store m1
    from gmail_classifier.models import Message
    store.save_message(Message(
        id="m1", subject="First", from_address="a@x.com", labels=["Tech"],
    ))

    fetch_messages_for_label(client, store, label_id="Label_1", label_name="Tech")

    messages = store.load_all()
    assert len(messages) == 3


def test_fetch_multiple_labels(tmp_path):
    service = MagicMock()
    # For simplicity, each label returns different messages
    service.users().messages().list.return_value.execute.side_effect = [
        {"messages": [{"id": "m1"}]},
        {"messages": [{"id": "m2"}]},
    ]
    service.users().messages().get.return_value.execute.side_effect = [
        _make_raw_message("m1", subject="Tech msg", labels=["Label_1"]),
        _make_raw_message("m2", subject="Travel msg", labels=["Label_2"]),
    ]

    client = GmailClient(service)
    store = MessageStore(str(tmp_path / "test.db"))

    fetch_messages_for_label(client, store, label_id="Label_1", label_name="Tech")
    fetch_messages_for_label(client, store, label_id="Label_2", label_name="Travel")

    assert len(store.load_by_label("Tech")) == 1
    assert len(store.load_by_label("Travel")) == 1


def test_fetch_removes_messages_no_longer_in_label(tmp_path):
    """If a message was unlabeled or moved to another label, remove it."""
    service = MagicMock()
    # Gmail now only has m2 and m3 under this label (m1 was removed)
    service.users().messages().list.return_value.execute.return_value = {
        "messages": [{"id": "m2"}, {"id": "m3"}],
    }
    # No new messages to fetch (m2 and m3 already stored)
    client = GmailClient(service)
    store = MessageStore(str(tmp_path / "test.db"))

    # Pre-store m1, m2, m3 all under "Tech"
    from gmail_classifier.models import Message
    store.save_message(Message(id="m1", subject="First", from_address="a@x.com", labels=["Tech"]))
    store.save_message(Message(id="m2", subject="Second", from_address="a@x.com", labels=["Tech"]))
    store.save_message(Message(id="m3", subject="Third", from_address="a@x.com", labels=["Tech"]))

    fetch_messages_for_label(client, store, label_id="Label_1", label_name="Tech")

    messages = store.load_by_label("Tech")
    assert len(messages) == 2
    assert {m.id for m in messages} == {"m2", "m3"}
