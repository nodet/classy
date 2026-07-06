"""Integration tests for the pending_new drain wiring in classify_and_label.

These exercise the *real* script helpers (``_classify_new_ids``, ``_make_drain``,
``_run_pubsub_mode``) rather than the pure pending_new orchestration (covered in
test_pending_new). They guard two regressions:

  P1 -- a message parked pre-maturity is added to ``skip_ids``; the drain must
        still classify it, not let ``collect_new_inbox_ids`` filter it right back
        out and clear the row without doing anything.
  P2 -- pending rows stranded by a crash after bootstrap_status="complete" must
        be drained on the next WARM startup, where no bootstrap controller runs.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gmail_classifier.pending_new import park_immature_mail
from gmail_classifier.training_index import TrainingIndex

_SPEC = importlib.util.spec_from_file_location(
    "classify_and_label",
    Path(__file__).resolve().parent.parent / "scripts" / "classify_and_label.py",
)
cal = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cal)


class _FakeEmbedder:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    dimension = 8

    def embed(self, text):
        return np.full(self.dimension, 1.0, dtype=np.float32)


class _FakeClient:
    def __init__(self):
        self.get_calls = []

    def watch(self, topic):
        return ("500", 10**18)

    def get_message(self, mid):
        self.get_calls.append(mid)
        # No user labels, empty body -> classifier returns NO_LABEL, so the
        # drain records a skip example (proving it processed the id).
        return {"id": mid, "labelIds": ["INBOX"],
                "payload": {"headers": [], "body": {}}}


class _FakeRegistry:
    """Minimal LabelRegistry stand-in for the classification path."""
    name_to_id = {"A": "L_A"}
    user_label_ids = {"L_A"}
    id_to_name = {"L_A": "A"}
    _excluded = set()
    max_label_width = 8


class _PendingBackend:
    """In-memory pending_new + skip recorder behind the StorageBackend seam."""
    def __init__(self, pending=None):
        # pending: id -> (history_id, reason)
        self.pending = dict(pending or {})
        self.skipped = []          # ids handed to upsert_skip
        self._cursor = "400"       # a durable cursor so state won't fail closed

    # pending_new
    def park_pending(self, message_id, history_id):
        self.pending.setdefault(message_id, (history_id, "immature"))

    def get_pending(self):
        return [(mid, hid, reason) for mid, (hid, reason) in self.pending.items()]

    def remove_pending(self, message_id):
        self.pending.pop(message_id, None)

    # skip recording (NO_LABEL results)
    def upsert_skip(self, msg, embedding):
        self.skipped.append(msg.id if hasattr(msg, "id") else msg)

    # cursor (state path)
    def get_last_processed_history_id(self):
        return self._cursor

    def set_last_processed_history_id(self, hid):
        self._cursor = hid


def _empty_index():
    # An index with a real skip mass so NO_LABEL is the classification outcome.
    embs = np.full((3, 8), 1.0, dtype=np.float32)
    return TrainingIndex(embs, ["__skip__", "__skip__", "__skip__"],
                         ["s1", "s2", "s3"])


def _args():
    return SimpleNamespace(k=5, dry_run=False, storage="state",
                           mode="pubsub", once=True, max_messages=50)


# --------------------------------------------------------------------------
# P1: a parked id (now in skip_ids) is still classified when drained.
# --------------------------------------------------------------------------

def test_drain_classifies_parked_ids_despite_skip_ids():
    """park_immature_mail adds parked ids to skip_ids; the drain must classify
    them anyway. Before the fix, collect_new_inbox_ids filtered them back out and
    the drain cleared the rows without fetching a single message."""
    from gmail_classifier.pending_new import events_for_ids

    backend = _PendingBackend()
    skip_ids = set()
    events = events_for_ids(["n1", "n2"])
    parked = park_immature_mail(events, skip_ids, backend, history_id="500")
    assert parked == ["n1", "n2"]
    assert skip_ids == {"n1", "n2"}          # parking marked them seen

    client = _FakeClient()
    drain = cal._make_drain(
        _args(), client, _FakeEmbedder(), _empty_index(), _FakeRegistry(),
        skip_ids, set(), backend)
    drain()

    # Both parked messages were actually fetched + classified (not filtered out),
    # recorded as skip examples, and their pending rows cleared.
    assert client.get_calls == ["n1", "n2"]
    assert set(backend.skipped) == {"n1", "n2"}
    assert backend.pending == {}


# --------------------------------------------------------------------------
# P2: a WARM startup drains rows stranded by a crash mid-drain.
# --------------------------------------------------------------------------

def test_warm_startup_drains_stranded_pending_rows():
    """A store marked complete but with leftover pending_new rows reads WARM on
    the next boot (plan is None -> no controller). The startup drain must still
    classify + clear those rows, or they sit forever."""
    backend = _PendingBackend(pending={"n1": ("500", "immature")})
    client = _FakeClient()
    skip_ids = set()

    # plan=None models a warm start; once=True returns right after the drain.
    cal._run_pubsub_mode(
        _args(), client, credentials=None, embedder=_FakeEmbedder(),
        index=_empty_index(), registry=_FakeRegistry(), skip_ids=skip_ids,
        backend=backend, plan=None)

    assert client.get_calls == ["n1"]        # the stranded row was classified
    assert backend.skipped == ["n1"]
    assert backend.pending == {}             # and cleared


def test_warm_startup_with_no_pending_is_noop():
    """No stranded rows -> the drain fetches nothing (the steady-state case)."""
    backend = _PendingBackend()
    client = _FakeClient()

    cal._run_pubsub_mode(
        _args(), client, credentials=None, embedder=_FakeEmbedder(),
        index=_empty_index(), registry=_FakeRegistry(), skip_ids=set(),
        backend=backend, plan=None)

    assert client.get_calls == []
