"""Pub/Sub subscriber wrapper for Gmail push notifications."""
import json
from dataclasses import dataclass
from typing import List


@dataclass
class PubSubNotification:
    """A decoded Gmail push notification."""
    email: str
    history_id: str


class PubSubSubscriber:
    """Wraps google.cloud.pubsub_v1.SubscriberClient for Gmail notifications."""

    def __init__(self, subscription_path: str, client=None):
        self._subscription_path = subscription_path
        if client is None:
            from google.cloud.pubsub_v1 import SubscriberClient
            client = SubscriberClient()
        self._client = client

    def pull(self, timeout: int = 60) -> List[PubSubNotification]:
        """Pull notifications from the subscription.

        Returns decoded notifications. Acknowledges received messages.
        Returns empty list on timeout or no messages.
        """
        from google.api_core.exceptions import DeadlineExceeded

        try:
            response = self._client.pull(
                subscription=self._subscription_path,
                max_messages=100,
                timeout=timeout,
            )
        except DeadlineExceeded:
            return []

        messages = response.received_messages
        if not messages:
            return []

        notifications = []
        ack_ids = []
        for msg in messages:
            data = json.loads(msg.message.data)
            notifications.append(PubSubNotification(
                email=data.get("emailAddress", ""),
                history_id=data.get("historyId", ""),
            ))
            ack_ids.append(msg.ack_id)

        self._client.acknowledge(
            subscription=self._subscription_path,
            ack_ids=ack_ids,
        )

        return notifications
