from unittest.mock import MagicMock, patch

from gmail_classifier.gmail_client import GmailClient


def test_list_user_labels():
    service = MagicMock()
    service.users().labels().list.return_value.execute.return_value = {
        "labels": [
            {"id": "Label_1", "name": "Tech", "type": "user"},
            {"id": "Label_2", "name": "Travel", "type": "user"},
        ]
    }
    client = GmailClient(service)
    labels = client.list_user_labels()
    assert labels == [("Label_1", "Tech"), ("Label_2", "Travel")]


def test_list_user_labels_excludes_system_labels():
    service = MagicMock()
    service.users().labels().list.return_value.execute.return_value = {
        "labels": [
            {"id": "INBOX", "name": "INBOX", "type": "system"},
            {"id": "SENT", "name": "SENT", "type": "system"},
            {"id": "Label_1", "name": "Tech", "type": "user"},
        ]
    }
    client = GmailClient(service)
    labels = client.list_user_labels()
    assert labels == [("Label_1", "Tech")]


def test_list_messages_by_label():
    service = MagicMock()
    service.users().messages().list.return_value.execute.return_value = {
        "messages": [{"id": "msg1"}, {"id": "msg2"}],
    }
    client = GmailClient(service)
    ids = client.list_message_ids(label_id="Label_1")
    assert ids == ["msg1", "msg2"]


def test_list_messages_by_label_pagination():
    service = MagicMock()
    # First call returns page 1 with a nextPageToken
    # Second call returns page 2 with no token
    service.users().messages().list.return_value.execute.side_effect = [
        {"messages": [{"id": "msg1"}, {"id": "msg2"}], "nextPageToken": "token123"},
        {"messages": [{"id": "msg3"}]},
    ]
    client = GmailClient(service)
    ids = client.list_message_ids(label_id="Label_1")
    assert ids == ["msg1", "msg2", "msg3"]


def test_list_messages_by_label_empty():
    service = MagicMock()
    service.users().messages().list.return_value.execute.return_value = {
        "resultSizeEstimate": 0,
    }
    client = GmailClient(service)
    ids = client.list_message_ids(label_id="Label_1")
    assert ids == []
