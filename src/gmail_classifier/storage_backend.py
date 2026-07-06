"""The ``StorageBackend`` seam: how the runtime loads and persists its index.

Both backends share one classification path and differ only in *where the
index comes from and where label/skip changes are written*. The runtime
(``classify_and_label.py`` and the label/inbox handlers) talks only to this
interface, never to a concrete store, so a second backend can be added without
touching the classification logic.

Phase 1 wires up **only** the ``legacy`` adapter (``storage_legacy.py``), which
wraps today's ``MessageStore`` + ``EmbeddingCache`` over the three DB files and
behaves byte-for-byte as before. The ``state`` backend (single ``state.db``,
derived-only) plugs in behind the same interface in a later phase.

The interface is a ``typing.Protocol`` for documentation and structural typing;
adapters are plain classes that satisfy it, and tests use fakes that do too.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Set, Tuple, runtime_checkable

import numpy as np

from gmail_classifier.models import Message
from gmail_classifier.training import AssemblyStats
from gmail_classifier.training_index import TrainingIndex


@dataclass
class LoadedIndex:
    """What a backend returns from :meth:`StorageBackend.load_index`.

    ``skip_ids`` is the set of message ids the live loop must not re-classify
    (on legacy, every sampled inbox id, matching today's behavior). ``stats``
    carries the counts the caller logs at startup.
    """
    index: TrainingIndex
    skip_ids: Set[str]
    stats: AssemblyStats


@runtime_checkable
class StorageBackend(Protocol):
    """The narrow interface the runtime talks to for index load + persistence.

    Live-adaptation callers pass the parsed :class:`Message` plus its vector.
    The legacy adapter uses the ``Message`` (with ``labels`` already set by the
    caller) to perform today's body-preserving ``save_message``; a future state
    adapter stores only ``message.id``, ``label_id``, and ``vec`` and discards
    the body. ``vec`` is accepted everywhere but only the state backend needs it.
    """

    def load_index(self, embedder) -> LoadedIndex:
        """Build the runtime KNN index and the set of already-seen ids."""
        ...

    def upsert_label(
        self, message: Message, label_id: str, vec: Optional[np.ndarray] = None
    ) -> None:
        """Record ``message`` as a labeled training example, drop it from skip.

        The caller sets ``message.labels`` to the intended label name(s); the
        legacy adapter persists that. ``label_id``/``vec`` are for the state
        backend.
        """
        ...

    def upsert_skip(self, message: Message, vec: Optional[np.ndarray] = None) -> None:
        """Record ``message`` as a ``__skip__`` example (removes any label row)."""
        ...

    def remove(self, message_id: str) -> None:
        """Drop a message from the labeled set (no-op if absent)."""
        ...

    def park_pending(self, message_id: str, history_id: str) -> None:
        """Park a genuinely-new, pre-maturity message: record it as awaiting a
        mature model, with no label row and no archive. Idempotent. On legacy
        this is a no-op (legacy has no maturity gate)."""
        ...

    def get_pending(self) -> List[Tuple[str, str, str]]:
        """The parked ``(message_id, first_seen_history_id, reason)`` rows to
        drain once the model matures. Empty on legacy."""
        ...

    def remove_pending(self, message_id: str) -> None:
        """Drop one parked row after it has been drained. Idempotent no-op on
        legacy."""
        ...

    def get_last_processed_history_id(self) -> Optional[str]:
        """Durable Gmail history cursor, or ``None`` (legacy: process-local)."""
        ...

    def set_last_processed_history_id(self, history_id: str) -> None:
        """Advance the history cursor (legacy: process-local, never persisted)."""
        ...

    def close(self) -> None:
        """Release any open resources."""
        ...
