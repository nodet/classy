"""Tests for the state backend's startup recovery paths in pubsub mode.

The startup path decides (via classify_gap) whether to replay from the durable
cursor (REPLAY) or run a read-only resync (RESYNC). These tests drive
_run_pubsub_mode with --once so the loop returns immediately after startup.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gmail_classifier import classifier as _classifier
from gmail_classifier.storage_state import (
    STATE_SCHEMA_VERSION,
    WARM_RECOVERY_WINDOW,
    StateBackend,
)
from gmail_classifier.training_index import TrainingIndex

_SPEC = importlib.util.spec_from_file_location(
    "classify_and_label",
    Path(__file__).resolve().parent.parent / "scripts" / "classify_and_label.py",
)
cal = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cal)


class _FakeEmbedder:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    dimension = 384

    def embed(self, text):
        return np.full(8, 1.0, dtype=np.float32)


# Default arrival timestamp (epoch seconds) for inbox ids given as bare
# strings, chosen far in the future so plain ["i1", "i2"] fixtures keep
# looking like "arrived after any past cutoff" -- tests that need to exercise
# the after_ts gap-catchup boundary pass explicit (id, ts) tuples instead.
_DEFAULT_MSG_TS = 9_999_999_999


class _ResyncClient:
    """Records watch/apply_label and serves a small mailbox for read-only resync.

    ``inbox`` is a list of ids, or ``(id, sent_ts)`` tuples (epoch seconds) for
    tests that need to exercise the ``after_ts`` gap-catchup boundary.
    """
    def __init__(self, labels=None, inbox=None):
        self._labels = labels or {}
        self._inbox = [
            m if isinstance(m, tuple) else (m, _DEFAULT_MSG_TS)
            for m in (inbox or [])
        ]
        self.watch_calls = 0
        self.applied = []

    def watch(self, topic):
        self.watch_calls += 1
        return ("900", 10**18)

    def list_message_ids(self, label_id, max_results=0):
        if label_id == "INBOX":
            return []
        ids = self._labels.get(label_id, ("", []))[1]
        return list(ids[:max_results]) if max_results else list(ids)

    def list_user_labels(self):
        return [(lid, name) for lid, (name, _ids) in self._labels.items()]

    def list_unlabeled_inbox_ids(self, max_results=0, after_ts=None):
        labeled = set()
        for _name, ids in self._labels.values():
            labeled.update(ids)
        unlabeled = [
            mid for mid, ts in self._inbox
            if mid not in labeled and (after_ts is None or ts > after_ts)
        ]
        return list(unlabeled[:max_results]) if max_results else list(unlabeled)

    def get_message(self, mid):
        return {"id": mid, "payload": {"headers": [], "body": {}}, "labelIds": []}

    def apply_label(self, *a, **k):
        self.applied.append((a, k))


class _Registry:
    """Mirrors the fields ``LabelRegistry.refresh()`` derives from the Gmail
    client, so tests that trigger classification (gap catchup) have a usable
    registry. ``client=None`` (the no-classification case) leaves everything
    empty."""
    def __init__(self, client=None):
        self._excluded = set()
        user_labels = client.list_user_labels() if client is not None else []
        self.name_to_id = {name: lid for lid, name in user_labels}
        self.id_to_name = {lid: name for lid, name in user_labels}
        self.user_label_ids = {lid for lid, _ in user_labels}
        self.max_label_width = max((len(n) for n in self.name_to_id), default=0)


def _pubsub_args():
    return SimpleNamespace(
        once=True, max_messages=50, k=5, dry_run=False, max_per_label=200,
    )


def _seed_warm_state(path, *, cursor, last_at, now_ms):
    """Seed a complete/WARM state.db with a durable cursor + timestamp."""
    backend = StateBackend(path, excluded=set(), now_ms=lambda: now_ms)
    store = backend.store
    store.upsert_embedding("a1", np.full(8, 1.0, dtype=np.float32))
    store.upsert_label("a1", "L_A", "A", source="user")
    store.set_meta("state_schema_version", STATE_SCHEMA_VERSION)
    store.set_meta("gmail_account_id", "me@x.com")
    store.set_meta("bootstrap_boundary_history_id", "400")
    store.set_meta("bootstrap_status", "complete")
    if cursor is not None:
        store.set_meta("last_processed_history_id", cursor)
    if last_at is not None:
        store.set_meta("last_processed_at", str(last_at))
    return backend


def test_state_within_window_resumes_without_resync(tmp_path):
    """A recent cursor (gap within WARM_RECOVERY_WINDOW) replays from the durable
    cursor: no read-only resync, no re-pin beyond the initial watch."""
    path = str(tmp_path / "state.db")
    backend = _seed_warm_state(path, cursor="410", last_at=1000, now_ms=1000)
    client = _ResyncClient(labels={"L_A": ("A", ["a1"])}, inbox=["i1"])
    index = TrainingIndex(np.empty((0, 0), dtype=np.float32), [], [])

    cal._run_pubsub_mode(
        _pubsub_args(), client, credentials=None,
        embedder=_FakeEmbedder(), index=index, registry=_Registry(client),
        skip_ids=set(), backend=backend,
    )
    assert client.watch_calls == 1
    assert backend.store.get_meta("bootstrap_boundary_history_id") == "400"
    assert client.applied == []
    backend.close()


def test_state_past_window_runs_read_only_resync(tmp_path, monkeypatch):
    """A cursor older than the window takes the read-only resync: it re-pins a
    fresh boundary, ingests Gmail's current label state read-only, and THEN
    classifies the inbox backlog that arrived after the last known-good state
    (gap catchup) -- unlike cold bootstrap, a crash recovery is expected to
    catch up on mail, not leave it stranded."""
    monkeypatch.setattr(_classifier, "MIN_EXAMPLES_PER_LABEL", 1)
    path = str(tmp_path / "state.db")
    now = 10 * WARM_RECOVERY_WINDOW
    backend = _seed_warm_state(path, cursor="410", last_at=0, now_ms=now)
    # 5 examples for A so it robustly outweighs a tied identical-embedding
    # vote from any residual skip competitor (see the fake embedder note).
    client = _ResyncClient(labels={"L_A": ("A", ["a1", "a2", "a3", "a4", "a5"])},
                           inbox=["i1", "i2", "i3"])
    index = TrainingIndex(np.empty((0, 0), dtype=np.float32), [], [])

    cal._run_pubsub_mode(
        _pubsub_args(), client, credentials=None,
        embedder=_FakeEmbedder(), index=index, registry=_Registry(client),
        skip_ids=set(), backend=backend,
    )
    assert backend.store.get_meta("bootstrap_boundary_history_id") == "900"
    assert backend.store.get_last_processed_history_id() == "900"
    assert {a[0] for a, _k in client.applied} == {"i1", "i2", "i3"}
    assert backend.store.get_bootstrap_status() == "complete"
    backend.close()


def test_state_absent_cursor_resyncs_not_fails_closed(tmp_path):
    """A warm-looking state.db with no durable cursor takes the read-only resync
    and starts cleanly from the fresh boundary. No ``last_processed_at`` means
    there is no known-good timestamp to bound gap catchup by, so -- unlike the
    past-window case -- this must NOT classify the inbox backlog: Gmail's
    current state is the only ground truth available (same safety guarantee as
    cold bootstrap)."""
    path = str(tmp_path / "state.db")
    backend = _seed_warm_state(path, cursor=None, last_at=None, now_ms=1000)
    client = _ResyncClient(labels={"L_A": ("A", ["a1"])}, inbox=["i1"])
    index = TrainingIndex(np.empty((0, 0), dtype=np.float32), [], [])

    cal._run_pubsub_mode(
        _pubsub_args(), client, credentials=None,
        embedder=_FakeEmbedder(), index=index, registry=_Registry(client),
        skip_ids=set(), backend=backend,
    )
    assert backend.store.get_last_processed_history_id() == "900"
    assert client.applied == []
    backend.close()


def test_gap_catchup_skips_known_messages(tmp_path, monkeypatch):
    """A gap-window message already in skip_ids (e.g. from a previously
    interrupted catchup) is not re-classified."""
    monkeypatch.setattr(_classifier, "MIN_EXAMPLES_PER_LABEL", 1)
    path = str(tmp_path / "state.db")
    now = 10 * WARM_RECOVERY_WINDOW
    backend = _seed_warm_state(path, cursor="410", last_at=0, now_ms=now)
    client = _ResyncClient(labels={"L_A": ("A", ["a1", "a2", "a3", "a4", "a5"])},
                           inbox=["i1", "i2", "i3"])
    index = TrainingIndex(np.empty((0, 0), dtype=np.float32), [], [])

    cal._run_pubsub_mode(
        _pubsub_args(), client, credentials=None,
        embedder=_FakeEmbedder(), index=index, registry=_Registry(client),
        skip_ids={"i2"}, backend=backend,
    )
    assert {a[0] for a, _k in client.applied} == {"i1", "i3"}
    backend.close()


def test_gap_catchup_respects_after_ts_boundary(tmp_path, monkeypatch):
    """A message older than the seeded last_processed_at is NOT caught up, even
    though it is unlabeled and not in skip_ids -- only genuinely new mail from
    after the last known-good state is a "gap" message."""
    monkeypatch.setattr(_classifier, "MIN_EXAMPLES_PER_LABEL", 1)
    path = str(tmp_path / "state.db")
    last_at_ms = 5_000_000
    now = last_at_ms + 10 * WARM_RECOVERY_WINDOW
    backend = _seed_warm_state(path, cursor="410", last_at=last_at_ms, now_ms=now)
    client = _ResyncClient(
        labels={"L_A": ("A", ["a1", "a2", "a3", "a4", "a5"])},
        inbox=[("old", 4000), ("new", 6000)],  # after_ts == 5000
    )
    index = TrainingIndex(np.empty((0, 0), dtype=np.float32), [], [])

    cal._run_pubsub_mode(
        _pubsub_args(), client, credentials=None,
        embedder=_FakeEmbedder(), index=index, registry=_Registry(client),
        skip_ids=set(), backend=backend,
    )
    assert {a[0] for a, _k in client.applied} == {"new"}
    backend.close()


def test_gap_catchup_parks_instead_of_classifying_during_cold_bootstrap(tmp_path):
    """Gap catchup must defer to the progressive-bootstrap maturity gate, not
    bypass it: while a cold bootstrap plan is in flight (controller not yet
    built, but ``plan`` says one is coming), gap messages are parked like any
    other pre-maturity mail, not classified/labeled directly."""
    path = str(tmp_path / "state.db")
    now = 10 * WARM_RECOVERY_WINDOW
    backend = _seed_warm_state(path, cursor="410", last_at=0, now_ms=now)
    client = _ResyncClient(labels={"L_A": ("A", ["a1", "a2", "a3", "a4", "a5"])},
                           inbox=["i1", "i2", "i3"])
    index = TrainingIndex(np.empty((0, 0), dtype=np.float32), [], [])
    plan = cal.BootstrapPlan(excluded=set(), max_per_label=200,
                             gmail_account_id="me@x.com")

    cal._run_pubsub_mode(
        _pubsub_args(), client, credentials=None,
        embedder=_FakeEmbedder(), index=index, registry=_Registry(client),
        skip_ids=set(), backend=backend, plan=plan,
    )
    assert client.applied == []  # not classified/labeled directly
    pending_ids = {mid for mid, _hid, _reason in backend.get_pending()}
    assert pending_ids == {"i1", "i2", "i3"}  # parked for the maturity drain
    backend.close()
