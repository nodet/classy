"""Progressive cold bootstrap: build the index *in* the live pubsub loop.

Phase 4 bootstraps by blocking -- ``bootstrap_index`` fetches the whole corpus
before the service goes live. Phase 5 turns that into the live-from-the-first-
second design: the boundary is pinned, the loop starts immediately, and one
**bounded round-robin batch** of the corpus is embedded *between* ``run_iteration``
calls. Live mail (accumulating on the subscription since ``watch()``) preempts
bootstrap between batches, so a notification arriving mid-bootstrap is serviced
before the corpus finishes.

Why single-threaded, not a background thread: ``TrainingIndex.add`` reassigns
``self.embeddings`` via ``np.vstack``; a concurrent ``classify`` reading it
mid-vstack races on a half-built array, and two threads would share one
embedder and compete for the e2-micro's single core. So the driver is a plain
iterator the loop pumps -- no lock, no index race, one embedder caller.

Index growth uses ``TrainingIndex.add_many`` once per batch (a single
``np.vstack`` for the whole batch), never one append per message.

This module owns *building* the index. The maturity gate (does the growing index
justify labeling new mail yet?) and ``pending_new`` park/drain live alongside it
in the classification path; the driver only exposes the maturity gate it planned
so the loop can consult it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set, Tuple

import numpy as np

from gmail_classifier import bootstrap as _bootstrap
from gmail_classifier.maturity import MaturityGate, build_gate
from gmail_classifier.state_store import StateStore
from gmail_classifier.training_index import TrainingIndex

# Per-batch bounds: process at most this many messages, or run for at most this
# many wall-clock seconds, before yielding back to the loop to service any
# pending live notification. Small enough that live mail is never starved.
BATCH_MAX_MESSAGES = 25
BATCH_MAX_SECONDS = 5.0


def _noop(*_args, **_kwargs) -> None:
    pass


@dataclass
class ProgressiveBootstrap:
    """Drives a resumable, batched cold bootstrap that grows a live index.

    Construct with the already-pinned boundary (call
    ``bootstrap.ensure_boundary`` first, so the read-only boundary is fixed
    before any message is read), then call :meth:`run_batch` repeatedly from the
    pubsub loop until :attr:`done`. Each batch embeds up to ``BATCH_MAX_MESSAGES``
    still-unbuilt messages (or runs ``BATCH_MAX_SECONDS``), persists each vector,
    and appends them to ``index`` with a single ``add_many``.
    """
    client: object
    embedder: object
    store: StateStore
    index: TrainingIndex
    excluded: Set[str]
    max_per_label: int
    gmail_account_id: Optional[str] = None
    log: Callable[..., None] = _noop
    now: Callable[[], float] = None  # injectable clock (monotonic-ish); test hook
    batch_max_messages: int = BATCH_MAX_MESSAGES
    batch_max_seconds: float = BATCH_MAX_SECONDS

    # Internal state, populated on first run_batch (lazy so construction is cheap
    # and the Gmail listing happens inside the loop, not before it goes live).
    _worklist: List[Tuple[str, str, Optional[str], str]] = field(default=None, repr=False)
    _pos: int = 0
    _built: int = 0
    gate: Optional[MaturityGate] = None
    done: bool = False
    _finalized: bool = False

    def _ensure_planned(self) -> None:
        if self._worklist is not None:
            return
        worklist, avail_labels, avail_skip = _bootstrap.plan_bootstrap_worklist(
            self.client, self.excluded, self.max_per_label)
        self._worklist = worklist
        self.gate = build_gate(avail_labels, avail_skip)
        self.log(
            f"Bootstrap: {len(worklist)} messages planned; maturity targets "
            f"{self.gate.label_targets}, skip {self.gate.skip_target}")

    def _clock(self) -> float:
        if self.now is not None:
            return self.now()
        import time
        return time.monotonic()

    def run_batch(self) -> int:
        """Embed one bounded batch and grow the live index. Returns the number
        of messages actually embedded this batch (0 when already done).

        Resumable and idempotent: ids already embedded (a resumed bootstrap, or
        a message labelled live in between) are skipped without a fetch, and do
        not count against the batch budget -- so a batch always makes real
        forward progress rather than burning its budget re-checking cached ids."""
        if self.done:
            return 0
        self._ensure_planned()

        deadline = self._clock() + self.batch_max_seconds
        additions: List[Tuple[str, np.ndarray, str]] = []

        while self._pos < len(self._worklist):
            if len(additions) >= self.batch_max_messages:
                break
            if additions and self._clock() >= deadline:
                break
            mid, label_id, label_name, source = self._worklist[self._pos]
            self._pos += 1
            vec = _bootstrap._fetch_embed_persist(
                self.client, self.embedder, self.store,
                mid, label_id, label_name, source)
            if vec is None:
                continue  # already embedded (resume / raced with a live label)
            additions.append(
                (mid, vec, _bootstrap.index_label_for(label_id, label_name)))

        if additions:
            self.index.add_many(additions)
            self._built += len(additions)

        if self._pos >= len(self._worklist):
            self._finalize()

        return len(additions)

    def _finalize(self) -> None:
        """Stamp the store complete exactly once, when the work-list is drained."""
        if self._finalized:
            return
        _bootstrap.finalize_bootstrap(
            self.store, self.embedder, excluded=self.excluded,
            gmail_account_id=self.gmail_account_id, n=self._built, log=self.log)
        self._finalized = True
        self.done = True

    def is_mature(self) -> bool:
        """Whether the growing index is broad enough to label new mail. Before
        the first batch (gate not yet planned) this is conservatively False --
        nothing should be labelled until we know the corpus shape."""
        if self.gate is None:
            return False
        return self.gate.is_open_for_index(self.index.labels)
