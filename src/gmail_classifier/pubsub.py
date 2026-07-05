"""Pub/Sub subscriber wrapper for Gmail push notifications."""
import json
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PubSubNotification:
    """A decoded Gmail push notification."""
    email: str
    history_id: str


class PubSubSubscriber:
    """Wraps google.cloud.pubsub_v1.SubscriberClient for Gmail notifications."""

    def __init__(self, subscription_path: str, client=None, credentials=None):
        self._subscription_path = subscription_path
        if client is None:
            from google.cloud.pubsub_v1 import SubscriberClient
            client = SubscriberClient(credentials=credentials)
        self._client = client

    def pull(self, timeout: int = 60) -> Tuple[List[PubSubNotification], List[str]]:
        """Pull notifications from the subscription.

        Returns ``(notifications, ack_ids)``. Does **not** acknowledge -- the
        caller must call :meth:`ack` only after the notifications' Gmail history
        has been durably processed, so a crash mid-processing leaves the
        messages un-acked for redelivery. Returns ``([], [])`` on timeout or no
        messages.
        """
        from google.api_core.exceptions import DeadlineExceeded

        try:
            response = self._client.pull(
                subscription=self._subscription_path,
                max_messages=100,
                timeout=timeout,
            )
        except DeadlineExceeded:
            return [], []

        messages = response.received_messages
        if not messages:
            return [], []

        notifications = []
        ack_ids = []
        for msg in messages:
            data = json.loads(msg.message.data)
            notifications.append(PubSubNotification(
                email=data.get("emailAddress", ""),
                history_id=data.get("historyId", ""),
            ))
            ack_ids.append(msg.ack_id)

        return notifications, ack_ids

    def ack(self, ack_ids: List[str]) -> None:
        """Acknowledge previously-pulled messages by their ack ids.

        Call this only after the pulled notifications' history has been
        processed (and, on a durable-cursor backend, persisted), so an
        interrupted run redelivers rather than dropping mail. No-op on an
        empty list.
        """
        if not ack_ids:
            return
        self._client.acknowledge(
            subscription=self._subscription_path,
            ack_ids=ack_ids,
        )

    def close(self) -> None:
        """Close the underlying gRPC channel and release its resources."""
        self._client.close()
