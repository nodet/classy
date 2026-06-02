from unittest.mock import MagicMock, patch

import pytest

from gmail_classifier.gmail_client import GmailClient
from gmail_classifier.models import HistoryExpiredError


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


def test_list_messages_with_max_results():
    service = MagicMock()
    # API returns 5 messages but we only want 3
    service.users().messages().list.return_value.execute.return_value = {
        "messages": [{"id": f"msg{i}"} for i in range(5)],
        "nextPageToken": "more",
    }
    client = GmailClient(service)
    ids = client.list_message_ids(label_id="Label_1", max_results=3)
    assert len(ids) == 3
    assert ids == ["msg0", "msg1", "msg2"]


def test_get_message():
    service = MagicMock()
    msg_resource = {
        "id": "msg1",
        "payload": {"headers": [{"name": "Subject", "value": "Hi"}]},
    }
    service.users().messages().get.return_value.execute.return_value = msg_resource
    client = GmailClient(service)
    result = client.get_message("msg1")
    assert result == msg_resource


def test_batch_get_messages():
    service = MagicMock()
    msg_resources = [
        {"id": f"msg{i}", "payload": {"headers": []}} for i in range(5)
    ]
    service.users().messages().get.return_value.execute.side_effect = msg_resources
    client = GmailClient(service)
    results = client.get_messages(["msg0", "msg1", "msg2", "msg3", "msg4"])
    assert len(results) == 5
    assert results[0]["id"] == "msg0"
    assert results[4]["id"] == "msg4"


def test_apply_label():
    service = MagicMock()
    client = GmailClient(service)
    client.apply_label("msg1", "Label_1")
    service.users().messages().modify.assert_called_once_with(
        userId="me", id="msg1", body={"addLabelIds": ["Label_1"]}
    )
    service.users().messages().modify.return_value.execute.assert_called_once()


def test_apply_label_with_archive():
    service = MagicMock()
    client = GmailClient(service)
    client.apply_label("msg1", "Label_1", archive=True)
    service.users().messages().modify.assert_called_once_with(
        userId="me", id="msg1",
        body={"addLabelIds": ["Label_1"], "removeLabelIds": ["INBOX"]}
    )
    service.users().messages().modify.return_value.execute.assert_called_once()


def test_get_message_labels():
    service = MagicMock()
    service.users().messages().get.return_value.execute.return_value = {
        "id": "msg1",
        "labelIds": ["INBOX", "UNREAD", "Label_1"],
    }
    client = GmailClient(service)
    labels = client.get_message_labels("msg1")
    assert labels == ["INBOX", "UNREAD", "Label_1"]


# --- history.list tests ---


def test_get_history_returns_messages_added():
    service = MagicMock()
    service.users().history().list.return_value.execute.return_value = {
        "history": [
            {
                "id": "100",
                "messagesAdded": [
                    {"message": {"id": "msg1", "labelIds": ["INBOX", "UNREAD"]}}
                ],
            }
        ],
        "historyId": "101",
    }
    client = GmailClient(service)
    events = client.get_history("99")
    assert len(events) == 1
    assert events[0].type == "messagesAdded"
    assert events[0].message_id == "msg1"
    assert events[0].label_ids == ["INBOX", "UNREAD"]


def test_get_history_returns_labels_added():
    service = MagicMock()
    service.users().history().list.return_value.execute.return_value = {
        "history": [
            {
                "id": "100",
                "labelsAdded": [
                    {"message": {"id": "msg2", "labelIds": ["Label_1"]},
                     "labelIds": ["Label_1"]}
                ],
            }
        ],
        "historyId": "101",
    }
    client = GmailClient(service)
    events = client.get_history("99")
    assert len(events) == 1
    assert events[0].type == "labelsAdded"
    assert events[0].message_id == "msg2"
    assert events[0].label_ids == ["Label_1"]


def test_get_history_returns_labels_removed():
    service = MagicMock()
    service.users().history().list.return_value.execute.return_value = {
        "history": [
            {
                "id": "100",
                "labelsRemoved": [
                    {"message": {"id": "msg3", "labelIds": ["INBOX"]},
                     "labelIds": ["Label_2"]}
                ],
            }
        ],
        "historyId": "101",
    }
    client = GmailClient(service)
    events = client.get_history("99")
    assert len(events) == 1
    assert events[0].type == "labelsRemoved"
    assert events[0].message_id == "msg3"
    assert events[0].label_ids == ["Label_2"]


def test_get_history_paginates():
    service = MagicMock()
    service.users().history().list.return_value.execute.side_effect = [
        {
            "history": [
                {"id": "100", "messagesAdded": [
                    {"message": {"id": "msg1", "labelIds": ["INBOX"]}}
                ]}
            ],
            "historyId": "101",
            "nextPageToken": "token1",
        },
        {
            "history": [
                {"id": "101", "messagesAdded": [
                    {"message": {"id": "msg2", "labelIds": ["INBOX"]}}
                ]}
            ],
            "historyId": "102",
        },
    ]
    client = GmailClient(service)
    events = client.get_history("99")
    assert len(events) == 2
    assert events[0].message_id == "msg1"
    assert events[1].message_id == "msg2"


def test_get_history_raises_on_expired_id():
    from googleapiclient.errors import HttpError

    service = MagicMock()
    resp = MagicMock()
    resp.status = 404
    service.users().history().list.return_value.execute.side_effect = HttpError(
        resp=resp, content=b"Not Found"
    )
    client = GmailClient(service)
    with pytest.raises(HistoryExpiredError):
        client.get_history("1")


def test_get_history_empty():
    service = MagicMock()
    service.users().history().list.return_value.execute.return_value = {
        "historyId": "100",
    }
    client = GmailClient(service)
    events = client.get_history("99")
    assert events == []


# --- watch tests ---


def test_watch_returns_history_id_and_expiration():
    service = MagicMock()
    service.users().watch.return_value.execute.return_value = {
        "historyId": "12345",
        "expiration": "1734567890000",
    }
    client = GmailClient(service)
    history_id, expiration = client.watch("projects/myproj/topics/gmail-notifications")
    assert history_id == "12345"
    assert expiration == 1734567890000
    service.users().watch.assert_called_once_with(
        userId="me",
        body={
            "topicName": "projects/myproj/topics/gmail-notifications",
            "labelIds": ["INBOX"],
        },
    )
