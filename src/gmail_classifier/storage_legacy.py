"""Legacy storage backend: today's three-DB layout behind the seam.

Wraps the existing ``MessageStore`` (training + skip) and ``EmbeddingCache``
(``embeddings.db``) so the runtime can talk to a ``StorageBackend`` without
naming concrete stores. This is a *move/wrap* of the code that used to live
inline in ``classify_and_label.py`` and the handlers -- behavior is unchanged.

The live-adaptation writes map 1:1 to today's store operations:

- ``upsert_label(msg, ...)``  -> ``training.save_message(msg)`` + drop from skip
- ``upsert_skip(msg, ...)``   -> drop from training + ``skip.save_message(msg)``
  (with ``labels`` emptied, as the two old call sites did)
- ``remove(id)``             -> ``training.delete_messages([id])``

The history cursor is deliberately **process-local, never persisted**: the
legacy service re-``watch()``es from a fresh ``historyId`` every boot, so a
fresh adapter starts with no cursor -- preserving today's fresh-watch restart
semantics byte-for-byte. Only a future ``state`` backend persists it.

Bodies are stored exactly as today (this is the body-preserving backend); the
``label_id``/``vec`` arguments are part of the shared seam for the state
backend and are ignored here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Set

import numpy as np

from gmail_classifier.embedding_cache import EmbeddingCache
from gmail_classifier.embeddings import Embedder
from gmail_classifier.models import Message
from gmail_classifier.storage import MessageStore
from gmail_classifier.storage_backend import LoadedIndex
from gmail_classifier.training import assemble_training_index


class LegacyBackend:
    """``StorageBackend`` over ``training.db`` / ``inbox_sample.db`` /
    ``embeddings.db``."""

    def __init__(self, training_db: str, skip_db: str, excluded: Set[str]):
        self._training_db = training_db
        self._skip_db = skip_db
        self._excluded = excluded
        # embeddings.db lives beside training.db, as main() has always assumed.
        self._cache_path = Path(training_db).parent / "embeddings.db"

        # Long-lived connections for the live-adaptation writes. Opened lazily
        # so a read-only/index-only use never touches the files for writing.
        self._training_store: Optional[MessageStore] = None
        self._skip_store: Optional[MessageStore] = None

        # Process-local history cursor (never persisted -- see module docstring).
        self._history_id: Optional[str] = None

    # --- index load ------------------------------------------------------

    def load_index(self, embedder: Embedder) -> LoadedIndex:
        """Build the runtime KNN index from the training + skip DBs.

        Mirrors the original ``main()`` sequence: load all training messages,
        load the skip sample (if the file exists), then assemble the
        cache-backed index (config exclusion + labeled-wins-over-skip dedup).
        """
        train_store = MessageStore(self._training_db)
        train_messages = train_store.load_all()
        train_store.close()

        skip_messages = []
        if Path(self._skip_db).exists():
            skip_store = MessageStore(self._skip_db)
            skip_messages = skip_store.load_all()
            skip_store.close()

        cache = EmbeddingCache(str(self._cache_path))
        index, skip_ids, stats = assemble_training_index(
            train_messages, skip_messages,
            excluded=self._excluded, embedder=embedder, cache=cache,
        )
        cache.close()
        return LoadedIndex(index=index, skip_ids=skip_ids, stats=stats)

    # --- live-adaptation writes -----------------------------------------

    def _training(self) -> MessageStore:
        if self._training_store is None:
            self._training_store = MessageStore(self._training_db)
        return self._training_store

    def _skip(self) -> MessageStore:
        if self._skip_store is None:
            self._skip_store = MessageStore(self._skip_db)
        return self._skip_store

    def upsert_label(
        self, message: Message, label_id: str, vec: Optional[np.ndarray] = None
    ) -> None:
        """Persist a labeled example (caller has set ``message.labels``) and
        drop it from the skip pool if present."""
        self._training().save_message(message)
        skip = self._skip()
        if skip.has_message(message.id):
            skip.delete_messages([message.id])

    def upsert_skip(self, message: Message, vec: Optional[np.ndarray] = None) -> None:
        """Move a message to the skip pool: drop any training row, save it with
        empty labels. The training delete is a no-op for a never-labeled
        inbox message, matching the two old call sites."""
        self._training().delete_messages([message.id])
        message.labels = []
        self._skip().save_message(message)

    def remove(self, message_id: str) -> None:
        """Drop a message from training (used when a label move is handled by a
        separate ``upsert_label`` for the new label)."""
        self._training().delete_messages([message_id])

    # --- history cursor (process-local, never persisted) ----------------

    def get_last_processed_history_id(self) -> Optional[str]:
        return self._history_id

    def set_last_processed_history_id(self, history_id: str) -> None:
        # In-memory only: a fresh adapter returns None, so restarts fresh-watch.
        self._history_id = history_id

    # --- lifecycle -------------------------------------------------------

    def close(self) -> None:
        if self._training_store is not None:
            self._training_store.close()
            self._training_store = None
        if self._skip_store is not None:
            self._skip_store.close()
            self._skip_store = None
