"""Mutable label registry with lazy refresh on unknown IDs."""
from typing import Dict, Optional, Set

from gmail_classifier.gmail_client import GmailClient


class LabelRegistry:
    """Maps between label names and Gmail IDs, with on-demand refresh."""

    def __init__(self, client: GmailClient, excluded: Set[str]):
        self._client = client
        self._excluded = excluded
        self.refresh()

    def refresh(self):
        """Re-fetch label list from Gmail API."""
        user_labels = self._client.list_user_labels()
        self.name_to_id: Dict[str, str] = {name: lid for lid, name in user_labels}
        self.id_to_name: Dict[str, str] = {lid: name for lid, name in user_labels}
        self.user_label_ids: Set[str] = {lid for lid, _ in user_labels}

    def is_known(self, label_id: str) -> bool:
        return label_id in self.id_to_name

    def is_excluded(self, label_id: str) -> bool:
        name = self.id_to_name.get(label_id)
        return name in self._excluded if name else False

    def get_name(self, label_id: str) -> Optional[str]:
        return self.id_to_name.get(label_id)

    def get_id(self, label_name: str) -> Optional[str]:
        return self.name_to_id.get(label_name)

    def ensure_known(self, label_id: str) -> bool:
        """Ensure a label ID is known; refresh from API if not.

        Returns True if the label is known (possibly after refresh),
        False if still unknown after refresh.
        """
        if label_id in self.id_to_name:
            return True
        self.refresh()
        return label_id in self.id_to_name
