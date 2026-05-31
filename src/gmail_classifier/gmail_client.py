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

    def list_message_ids(self, label_id: str) -> List[str]:
        """List all message IDs with the given label. Handles pagination."""
        ids = []
        page_token = None
        while True:
            kwargs = {"userId": "me", "labelIds": [label_id]}
            if page_token:
                kwargs["pageToken"] = page_token
            response = self._service.users().messages().list(**kwargs).execute()
            messages = response.get("messages", [])
            ids.extend(m["id"] for m in messages)
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return ids
