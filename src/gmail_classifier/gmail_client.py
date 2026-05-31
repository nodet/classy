from typing import List, Tuple


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

    def apply_label(self, message_id: str, label_id: str):
        """Add a label to a message."""
        self._service.users().messages().modify(
            userId="me", id=message_id, body={"addLabelIds": [label_id]}
        ).execute()

    def get_message_labels(self, message_id: str) -> List[str]:
        """Get the label IDs currently on a message (minimal fetch)."""
        result = self._service.users().messages().get(
            userId="me", id=message_id, format="minimal"
        ).execute()
        return result.get("labelIds", [])
