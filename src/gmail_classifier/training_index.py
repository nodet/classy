"""Mutable training index for KNN classification."""
from typing import Dict, List

import numpy as np


class TrainingIndex:
    """Wraps training embeddings + labels with add/remove support."""

    def __init__(self, embeddings: np.ndarray, labels: List[str], ids: List[str]):
        assert len(embeddings) == len(labels) == len(ids)
        self.embeddings = embeddings
        self.labels = list(labels)
        self._ids = list(ids)
        self._id_to_idx: Dict[str, int] = {mid: i for i, mid in enumerate(ids)}

    def __len__(self):
        return len(self.labels)

    def __contains__(self, message_id: str) -> bool:
        return message_id in self._id_to_idx

    def add(self, message_id: str, embedding: np.ndarray, label: str):
        """Add or replace a message in the index."""
        if message_id in self._id_to_idx:
            # Replace in-place
            idx = self._id_to_idx[message_id]
            self.embeddings[idx] = embedding
            self.labels[idx] = label
        else:
            # Append
            self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
            self.labels.append(label)
            self._ids.append(message_id)
            self._id_to_idx[message_id] = len(self._ids) - 1

    def remove(self, message_id: str):
        """Remove a message from the index. No-op if not present."""
        if message_id not in self._id_to_idx:
            return

        idx = self._id_to_idx[message_id]
        last_idx = len(self._ids) - 1

        if idx != last_idx:
            # Swap with last element for O(1) removal
            self.embeddings[idx] = self.embeddings[last_idx]
            self.labels[idx] = self.labels[last_idx]
            self._ids[idx] = self._ids[last_idx]
            self._id_to_idx[self._ids[idx]] = idx

        # Remove last element
        self.embeddings = self.embeddings[:last_idx]
        self.labels.pop()
        del self._id_to_idx[message_id]
        self._ids.pop()
