"""Tests for pending_new park/drain orchestration.

Guards the plan's "Unit -- maturity gate" drain bullets at the orchestration
level: pre-maturity new mail is parked (not labelled, not __skip__), and once
mature it is drained through normal classification and removed idempotently.
The gate decision itself is tested in test_maturity; the store table in
test_state_store.
"""
from gmail_classifier.models import HistoryEvent
from gmail_classifier.pending_new import (
    drain_pending,
    events_for_ids,
    park_immature_mail,
)


class _FakeBackend:
    """Records park/get/remove against an in-memory pending table."""
    def __init__(self):
        self.pending = {}  # id -> (history_id, reason)

    def park_pending(self, message_id, history_id):
        # Idempotent: first park wins (mirrors INSERT OR IGNORE).
        self.pending.setdefault(message_id, (history_id, "immature"))

    def get_pending(self):
        return [(mid, hid, reason) for mid, (hid, reason) in self.pending.items()]

    def remove_pending(self, message_id):
        self.pending.pop(message_id, None)


def _added(mid):
    return HistoryEvent(type="messagesAdded", message_id=mid, label_ids=["INBOX"])


def test_park_records_new_inbox_ids_only():
    backend = _FakeBackend()
    events = [
        _added("n1"),
        _added("n2"),
        HistoryEvent(type="labelsAdded", message_id="lbl", label_ids=["Label_1"]),
        HistoryEvent(type="messagesAdded", message_id="sent", label_ids=["SENT"]),
    ]
    skip_ids = set()
    parked = park_immature_mail(events, skip_ids, backend, history_id="500")

    assert parked == ["n1", "n2"]                 # only INBOX additions
    assert set(backend.pending) == {"n1", "n2"}
    assert backend.pending["n1"] == ("500", "immature")
    # Parked ids are marked seen so the same session doesn't re-park them.
    assert skip_ids == {"n1", "n2"}


def test_park_is_idempotent_and_skips_known():
    backend = _FakeBackend()
    backend.park_pending("n1", "400")             # already parked earlier
    skip_ids = {"known"}
    parked = park_immature_mail(
        [_added("n1"), _added("known"), _added("n2")],
        skip_ids, backend, history_id="500")

    assert parked == ["n1", "n2"]                 # "known" filtered out
    assert backend.pending["n1"] == ("400", "immature")  # first park wins
    assert set(backend.pending) == {"n1", "n2"}


def test_drain_processes_then_removes_each_row():
    backend = _FakeBackend()
    backend.park_pending("n1", "500")
    backend.park_pending("n2", "500")

    processed = []
    drained = drain_pending(backend, process_ids=lambda ids: processed.append(list(ids)))

    assert drained == ["n1", "n2"]
    assert processed == [["n1", "n2"]]            # classified in one pass
    assert backend.pending == {}                  # all rows cleared


def test_drain_empty_is_noop():
    backend = _FakeBackend()
    calls = []
    drained = drain_pending(backend, process_ids=lambda ids: calls.append(ids))
    assert drained == []
    assert calls == []                            # processor not invoked


def test_drain_leaves_rows_if_processing_raises():
    """A crash mid-drain must leave the parked rows so the next mature pass
    re-drains them (idempotent)."""
    backend = _FakeBackend()
    backend.park_pending("n1", "500")

    def boom(ids):
        raise RuntimeError("classify blew up")

    try:
        drain_pending(backend, process_ids=boom)
    except RuntimeError:
        pass
    assert set(backend.pending) == {"n1"}         # not removed


def test_events_for_ids_are_inbox_added():
    events = events_for_ids(["a", "b"])
    assert [e.message_id for e in events] == ["a", "b"]
    assert all(e.type == "messagesAdded" and e.label_ids == ["INBOX"] for e in events)
