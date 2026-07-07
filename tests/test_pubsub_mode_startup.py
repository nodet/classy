"""Tests for the state backend's startup recovery paths in pubsub mode.

The startup path decides (via classify_gap) whether to replay from the durable
cursor (REPLAY) or run a read-only resync (RESYNC). These tests drive
_run_pubsub_mode with --once so the loop returns immediately after startup.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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


class _ResyncClient:
    """Records watch/apply_label and serves a small mailbox for read-only resync."""
    def __init__(self, labels=None, inbox=None):
        self._labels = labels or {}
        self._inbox = inbox or []
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

    def list_unlabeled_inbox_ids(self, max_results=0):
        labeled = set()
        for _name, ids in self._labels.values():
            labeled.update(ids)
        unlabeled = [m for m in self._inbox if m not in labeled]
        return list(unlabeled[:max_results]) if max_results else list(unlabeled)

    def get_message(self, mid):
        return {"id": mid, "payload": {"headers": [], "body": {}}, "labelIds": []}

    def apply_label(self, *a, **k):
        self.applied.append((a, k))


class _Registry:
    _excluded = set()


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
        embedder=_FakeEmbedder(), index=index, registry=_Registry(),
        skip_ids=set(), backend=backend,
    )
    assert client.watch_calls == 1
    assert backend.store.get_meta("bootstrap_boundary_history_id") == "400"
    assert client.applied == []
    backend.close()


def test_state_past_window_runs_read_only_resync(tmp_path):
    """A cursor older than the window takes the read-only resync: it re-pins a
    fresh boundary and never labels/archives the accumulated inbox backlog."""
    path = str(tmp_path / "state.db")
    now = 10 * WARM_RECOVERY_WINDOW
    backend = _seed_warm_state(path, cursor="410", last_at=0, now_ms=now)
    client = _ResyncClient(labels={"L_A": ("A", ["a1"])},
                           inbox=["i1", "i2", "i3"])
    index = TrainingIndex(np.empty((0, 0), dtype=np.float32), [], [])

    cal._run_pubsub_mode(
        _pubsub_args(), client, credentials=None,
        embedder=_FakeEmbedder(), index=index, registry=_Registry(),
        skip_ids=set(), backend=backend,
    )
    assert backend.store.get_meta("bootstrap_boundary_history_id") == "900"
    assert backend.store.get_last_processed_history_id() == "900"
    assert client.applied == []
    assert backend.store.get_bootstrap_status() == "complete"
    backend.close()


def test_state_absent_cursor_resyncs_not_fails_closed(tmp_path):
    """A warm-looking state.db with no durable cursor takes the read-only resync
    and starts cleanly from the fresh boundary."""
    path = str(tmp_path / "state.db")
    backend = _seed_warm_state(path, cursor=None, last_at=None, now_ms=1000)
    client = _ResyncClient(labels={"L_A": ("A", ["a1"])}, inbox=["i1"])
    index = TrainingIndex(np.empty((0, 0), dtype=np.float32), [], [])

    cal._run_pubsub_mode(
        _pubsub_args(), client, credentials=None,
        embedder=_FakeEmbedder(), index=index, registry=_Registry(),
        skip_ids=set(), backend=backend,
    )
    assert backend.store.get_last_processed_history_id() == "900"
    assert client.applied == []
    backend.close()
