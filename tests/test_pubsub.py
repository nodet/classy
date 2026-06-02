"""Tests for PubSub subscriber wrapper."""
import json
from unittest.mock import MagicMock, patch

from gmail_classifier.pubsub import PubSubSubscriber, PubSubNotification


def test_pull_returns_decoded_notifications():
    mock_client = MagicMock()
    msg_data = json.dumps({"emailAddress": "user@gmail.com", "historyId": "123"}).encode()
    mock_message = MagicMock()
    mock_message.message.data = msg_data
    mock_message.ack_id = "ack1"

    mock_client.pull.return_value.received_messages = [mock_message]

    subscriber = PubSubSubscriber(
        subscription_path="projects/classy-498012/subscriptions/gmail-notifications-sub",
        client=mock_client,
    )
    notifications = subscriber.pull(timeout=30)

    assert len(notifications) == 1
    assert notifications[0].email == "user@gmail.com"
    assert notifications[0].history_id == "123"
    # Verify ack was called
    mock_client.acknowledge.assert_called_once_with(
        subscription="projects/classy-498012/subscriptions/gmail-notifications-sub",
        ack_ids=["ack1"],
    )


def test_pull_returns_empty_on_no_messages():
    mock_client = MagicMock()
    mock_client.pull.return_value.received_messages = []

    subscriber = PubSubSubscriber(
        subscription_path="projects/classy-498012/subscriptions/gmail-notifications-sub",
        client=mock_client,
    )
    notifications = subscriber.pull(timeout=30)

    assert notifications == []
    mock_client.acknowledge.assert_not_called()


def test_pull_returns_empty_on_timeout():
    from google.api_core.exceptions import DeadlineExceeded

    mock_client = MagicMock()
    mock_client.pull.side_effect = DeadlineExceeded("timeout")

    subscriber = PubSubSubscriber(
        subscription_path="projects/classy-498012/subscriptions/gmail-notifications-sub",
        client=mock_client,
    )
    notifications = subscriber.pull(timeout=30)

    assert notifications == []


def test_pull_multiple_notifications():
    mock_client = MagicMock()
    messages = []
    for i in range(3):
        msg = MagicMock()
        msg.message.data = json.dumps(
            {"emailAddress": "user@gmail.com", "historyId": str(100 + i)}
        ).encode()
        msg.ack_id = f"ack{i}"
        messages.append(msg)

    mock_client.pull.return_value.received_messages = messages

    subscriber = PubSubSubscriber(
        subscription_path="projects/classy-498012/subscriptions/gmail-notifications-sub",
        client=mock_client,
    )
    notifications = subscriber.pull(timeout=30)

    assert len(notifications) == 3
    assert notifications[0].history_id == "100"
    assert notifications[2].history_id == "102"
    mock_client.acknowledge.assert_called_once_with(
        subscription="projects/classy-498012/subscriptions/gmail-notifications-sub",
        ack_ids=["ack0", "ack1", "ack2"],
    )
