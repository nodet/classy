"""Mutable label registry with lazy refresh on unknown IDs."""
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

from gmail_classifier.gmail_client import GmailClient


@dataclass
class LabelDiff:
    """Changes detected between two consecutive label refreshes."""
    deleted: Dict[str, str] = field(default_factory=dict)
    renamed: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.deleted and not self.renamed


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
        names = [n for n in self.name_to_id if n not in self._excluded]
        self.max_label_width: int = max((len(n) for n in names), default=0)

    def is_known(self, label_id: str) -> bool:
        return label_id in self.id_to_name

    def is_excluded(self, label_id: str) -> bool:
        name = self.id_to_name.get(label_id)
        return name in self._excluded if name else False

    def get_name(self, label_id: str) -> Optional[str]:
        return self.id_to_name.get(label_id)

    def get_id(self, label_name: str) -> Optional[str]:
        return self.name_to_id.get(label_name)

    def refresh_with_diff(self) -> LabelDiff:
        """Refresh from the API and return what changed.

        Deleted: IDs that were known before but are now absent.
        Renamed: IDs still present but mapping to a different name.
        """
        old = dict(self.id_to_name)
        self.refresh()
        diff = LabelDiff()
        for lid, old_name in old.items():
            if lid not in self.id_to_name:
                if old_name not in self._excluded:
                    diff.deleted[lid] = old_name
            elif self.id_to_name[lid] != old_name:
                diff.renamed[lid] = (old_name, self.id_to_name[lid])
        return diff

    def ensure_known(self, label_id: str) -> bool:
        """Ensure a label ID is known; refresh from API if not.

        Returns True if the label is known (possibly after refresh),
        False if still unknown after refresh.
        """
        if label_id in self.id_to_name:
            return True
        self.refresh()
        return label_id in self.id_to_name
