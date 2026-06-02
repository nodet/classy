from typing import List, Tuple

from gmail_classifier.models import HistoryEvent, HistoryExpiredError


class GmailClient:
    """Thin wrapper around the Gmail API service object."""

    def __init__(self, service):
        self._service = service

    def list_user_labels(self) -> List[Tuple[str, str]]:
        """List user-created labels. Returns [(id, name), ...]."""
        response = self._service.users().labels().list(userId="me").execute()
        labels = response.get("labels", [])
        return [
            (l["id"], l["name"])
            for l in labels
            if l.get("type") == "user"
        ]

    def list_message_ids(self, label_id: str, max_results: int = 0) -> List[str]:
        """List message IDs with the given label. Handles pagination.

        Args:
            label_id: Gmail label ID to filter by.
            max_results: Maximum number of IDs to return (0 = no limit).
                         Gmail returns most recent first.
        """
        ids = []
        page_token = None
        while True:
            kwargs = {"userId": "me", "labelIds": [label_id]}
            if page_token:
                kwargs["pageToken"] = page_token
            if max_results:
                # Request at most what we still need (Gmail caps at 500 per page)
                kwargs["maxResults"] = min(max_results - len(ids), 500)
            response = self._service.users().messages().list(**kwargs).execute()
            messages = response.get("messages", [])
            ids.extend(m["id"] for m in messages)
            if max_results and len(ids) >= max_results:
                ids = ids[:max_results]
                break
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return ids

    def get_message(self, message_id: str) -> dict:
        """Get a single message by ID."""
        return self._service.users().messages().get(
            userId="me", id=message_id, format="full"
        ).execute()

    def get_messages(self, message_ids: List[str]) -> List[dict]:
        """Get multiple messages by ID."""
        return [self.get_message(mid) for mid in message_ids]

    def apply_label(self, message_id: str, label_id: str, archive: bool = False):
        """Add a label to a message, optionally archiving it."""
        body = {"addLabelIds": [label_id]}
        if archive:
            body["removeLabelIds"] = ["INBOX"]
        self._service.users().messages().modify(
            userId="me", id=message_id, body=body
        ).execute()

    def get_message_labels(self, message_id: str) -> List[str]:
        """Get the label IDs currently on a message (minimal fetch)."""
        result = self._service.users().messages().get(
            userId="me", id=message_id, format="minimal"
        ).execute()
        return result.get("labelIds", [])

    def get_history(self, start_history_id: str) -> List[HistoryEvent]:
        """Get mailbox changes since the given history ID.

        Returns a list of HistoryEvents for messagesAdded, labelsAdded,
        and labelsRemoved. Raises HistoryExpiredError if the history ID
        is too old.
        """
        from googleapiclient.errors import HttpError

        events = []
        page_token = None
        while True:
            kwargs = {"userId": "me", "startHistoryId": start_history_id}
            if page_token:
                kwargs["pageToken"] = page_token
            try:
                response = self._service.users().history().list(**kwargs).execute()
            except HttpError as e:
                if e.resp.status == 404:
                    raise HistoryExpiredError(
                        f"History ID {start_history_id} is too old"
                    ) from e
                raise

            for record in response.get("history", []):
                for added in record.get("messagesAdded", []):
                    msg = added["message"]
                    events.append(HistoryEvent(
                        type="messagesAdded",
                        message_id=msg["id"],
                        label_ids=msg.get("labelIds", []),
                    ))
                for added in record.get("labelsAdded", []):
                    events.append(HistoryEvent(
                        type="labelsAdded",
                        message_id=added["message"]["id"],
                        label_ids=added.get("labelIds", []),
                    ))
                for removed in record.get("labelsRemoved", []):
                    events.append(HistoryEvent(
                        type="labelsRemoved",
                        message_id=removed["message"]["id"],
                        label_ids=removed.get("labelIds", []),
                    ))

            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return events

    def watch(self, topic_name: str) -> Tuple[str, int]:
        """Register for push notifications via Gmail Watch API.

        No labelIds filter — get notifications for all mailbox changes
        (new messages, label adds/removes on any message, etc.).

        Returns (history_id, expiration_ms).
        """
        result = self._service.users().watch(
            userId="me",
            body={
                "topicName": topic_name,
            },
        ).execute()
        return result["historyId"], int(result["expiration"])
