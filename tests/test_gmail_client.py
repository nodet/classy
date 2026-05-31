from unittest.mock import MagicMock

from gmail_classifier.gmail_client import GmailClient


def _make_mock_service(labels_response=None, messages_list_responses=None, messages_get_responses=None):
    """Create a mock Gmail API service."""
    service = MagicMock()

    if labels_response is not None:
        service.users().labels().list.return_value.execute.return_value = labels_response

    return service


def test_list_user_labels():
    service = _make_mock_service(labels_response={
        "labels": [
            {"id": "Label_1", "name": "Tech", "type": "user"},
            {"id": "Label_2", "name": "Travel", "type": "user"},
        ]
    })
    client = GmailClient(service)
    labels = client.list_user_labels()
    assert labels == [("Label_1", "Tech"), ("Label_2", "Travel")]


def test_list_user_labels_excludes_system_labels():
    service = _make_mock_service(labels_response={
        "labels": [
            {"id": "INBOX", "name": "INBOX", "type": "system"},
            {"id": "SENT", "name": "SENT", "type": "system"},
            {"id": "Label_1", "name": "Tech", "type": "user"},
        ]
    })
    client = GmailClient(service)
    labels = client.list_user_labels()
    assert labels == [("Label_1", "Tech")]
