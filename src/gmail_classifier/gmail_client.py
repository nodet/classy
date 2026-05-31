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
