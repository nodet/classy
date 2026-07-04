"""Mutable training index for KNN classification."""
from typing import Dict, Iterable, List, Tuple

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

    def add_many(self, items: Iterable[Tuple[str, np.ndarray, str]]):
        """Add or replace many messages with a single reallocation.

        Equivalent to calling ``add`` once per item, but appends every
        genuinely-new row with ONE ``np.vstack`` instead of one per message.
        Each per-message ``vstack`` reallocates (and transiently doubles) the
        whole matrix; on a bulk relabel of N messages that is N reallocations
        copying ~N*rows, which spikes RSS and fragments the glibc arena. Batching
        collapses that to a single copy of ``current + N`` rows.

        Existing ids are replaced in place (cheap, no reallocation). If an id
        repeats within one batch the last entry wins, matching sequential
        ``add``. Does not sort or dedupe against anything but the current index.
        """
        new_ids: List[str] = []
        new_rows: List[np.ndarray] = []
        new_labels: List[str] = []
        pending: Dict[str, int] = {}  # id -> position in new_* for within-batch dupes

        for message_id, embedding, label in items:
            row = np.asarray(embedding, dtype=self.embeddings.dtype).reshape(1, -1)
            if message_id in self._id_to_idx:
                # Replace in-place, exactly like add().
                idx = self._id_to_idx[message_id]
                self.embeddings[idx] = row
                self.labels[idx] = label
            elif message_id in pending:
                # Repeated new id within this batch: last write wins.
                j = pending[message_id]
                new_rows[j] = row
                new_labels[j] = label
            else:
                pending[message_id] = len(new_ids)
                new_ids.append(message_id)
                new_rows.append(row)
                new_labels.append(label)

        if not new_rows:
            return

        self.embeddings = np.vstack([self.embeddings, *new_rows])
        base = len(self._ids)
        for offset, (message_id, label) in enumerate(zip(new_ids, new_labels)):
            self._ids.append(message_id)
            self.labels.append(label)
            self._id_to_idx[message_id] = base + offset

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
