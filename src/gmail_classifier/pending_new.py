"""Maturity-gate parking and draining of genuinely-new pre-maturity mail.

While the progressive bootstrap is still warming the index, new post-boundary
mail must not be labelled or archived (an early mistake is semi-irreversible --
the live path *archives* at >= 0.80). But it must not be dropped or written as
``__skip__`` either, because ``__skip__`` is a final verdict. Instead it is
**parked** in ``pending_new``: recorded by id + history id, with no label row.

When the maturity gate opens, the parked mail is **drained** through the normal
classifier exactly as if it had just arrived, then each row is removed. Draining
is idempotent: ``pending_new`` stores no body, so each parked id is re-fetched
from Gmail; a crash mid-drain simply re-drains the rows that were not yet
removed on the next mature pass.

Both helpers are pure orchestration over the ``StorageBackend`` seam and a
message processor, so they are unit-testable with fakes. On legacy the backend's
pending methods are no-ops, so parking is a no-op and draining finds nothing --
legacy keeps classifying new mail immediately, as today.
"""
from __future__ import annotations

from typing import Callable, List

from gmail_classifier.history_processor import collect_new_inbox_ids
from gmail_classifier.models import HistoryEvent


def park_ids(ids: List[str], skip_ids, backend, history_id: str) -> None:
    """Park each id in ``pending_new`` and mark it seen in ``skip_ids`` so the
    same session doesn't reconsider it before draining. Shared by
    :func:`park_immature_mail` (pre-maturity live mail) and gap catchup
    (read-only resync's downtime backlog) -- both need "record it now, drain
    it whenever it's safe to classify" with the same crash-safety story."""
    for mid in ids:
        backend.park_pending(mid, history_id)
        skip_ids.add(mid)


def park_immature_mail(events, skip_ids, backend, history_id: str) -> List[str]:
    """Park every genuinely-new inbox id in ``events`` for later classification.

    Returns the parked ids. ``skip_ids`` is updated so the same session does not
    re-consider them before maturity (they are already accounted for). Parking
    is idempotent at the store level, and label-change events are untouched --
    the caller still processes those, so user corrections keep growing the index
    even before maturity."""
    new_ids = collect_new_inbox_ids(events, skip_ids)
    park_ids(new_ids, skip_ids, backend, history_id)
    return new_ids


def drain_pending(backend, process_ids: Callable[[List[str]], None]) -> List[str]:
    """Classify every parked id through ``process_ids`` and clear its row.

    ``process_ids`` takes a list of message ids and runs the normal
    classification path over them (fetch → classify → label/skip). Each row is
    removed only after ``process_ids`` returns, so a crash mid-drain re-drains
    the survivors next time. Returns the drained ids (empty when nothing was
    parked -- the steady-state and legacy case)."""
    pending = backend.get_pending()
    if not pending:
        return []
    ids = [row[0] for row in pending]
    process_ids(ids)
    for mid in ids:
        backend.remove_pending(mid)
    return ids


def events_for_ids(ids: List[str]) -> List[HistoryEvent]:
    """Synthesize ``messagesAdded``/INBOX events for parked ids so they flow
    through the same ``process_history_events`` path as freshly-arrived mail."""
    return [
        HistoryEvent(type="messagesAdded", message_id=mid, label_ids=["INBOX"])
        for mid in ids
    ]
