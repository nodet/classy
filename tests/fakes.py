"""Shared test fixtures/fakes for the storage seam."""
from typing import Dict, Optional

import numpy as np

from gmail_classifier.models import Message


class FakeBackend:
    """In-memory ``StorageBackend`` for handler tests.

    Records label/skip upserts in dicts keyed by message id, mirroring the
    legacy adapter's semantics (labeled-wins/last-write-wins on a given id).
    Tests assert on ``labeled`` / ``skipped`` instead of on concrete stores.
    """

    def __init__(self):
        self.labeled: Dict[str, Message] = {}
        self.skipped: Dict[str, Message] = {}
        self.upsert_label_calls = []
        self.upsert_skip_calls = []
        self.removed = []
        self._history_id: Optional[str] = None

    def upsert_label(self, message: Message, label_id: str,
                     vec: Optional[np.ndarray] = None) -> None:
        self.labeled[message.id] = message
        self.skipped.pop(message.id, None)
        self.upsert_label_calls.append((message, label_id, vec))

    def upsert_skip(self, message: Message,
                    vec: Optional[np.ndarray] = None) -> None:
        message.labels = []
        self.skipped[message.id] = message
        self.labeled.pop(message.id, None)
        self.upsert_skip_calls.append((message, vec))

    def remove(self, message_id: str) -> None:
        self.labeled.pop(message_id, None)
        self.removed.append(message_id)

    def get_last_processed_history_id(self) -> Optional[str]:
        return self._history_id

    def set_last_processed_history_id(self, history_id: str) -> None:
        self._history_id = history_id

    def close(self) -> None:
        pass
