"""Tests that the labeling inbox sweep in pubsub mode is legacy-only.

The startup "initial inbox check" (and the HistoryExpiredError inbox-poll
fallback) list the current INBOX and classify whatever is unlabeled. That is
correct catch-up behavior for legacy, but unsafe for the state backend: inbox
listing carries no per-message historyId, so it cannot enforce the read-only
boundary, and a sweep could label/archive pre-boundary backlog. State catches
up via history replay from its durable cursor instead.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gmail_classifier.state_store import (
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


class _SweepClient:
    """Records the startup inbox listing so a test can assert it did/didn't run."""
    def __init__(self):
        self.list_calls = 0

    def watch(self, topic):
        return ("500", 10**18)

    def list_message_ids(self, label_id, max_results=0):
        self.list_calls += 1
        return []  # empty -> _check_inbox returns right after listing


class _CursorBackend:
    """A durable cursor stand-in exposing the slice of StateStore that the state
    startup path reaches through ``backend.store``: the cursor, its timestamp,
    and an injected clock.

    ``now_ms``/``last_at`` control ``classify_gap``: with the defaults the gap is
    0 (REPLAY, resume from the cursor). A test that wants the read-only resync
    path drops the cursor (``cursor=None``) or pushes ``last_at`` outside the
    recovery window."""
    def __init__(self, cursor=None, last_at=0, now_ms=0):
        self._cursor = cursor
        self._last_at = last_at
        self._now_ms = now_ms

    # The state path reaches the store's cursor/clock via ``backend.store``.
    @property
    def store(self):
        return self

    def now_ms(self):
        return self._now_ms

    def get_last_processed_history_id(self):
        return self._cursor
    def set_last_processed_history_id(self, hid):
        self._cursor = hid
        self._last_at = self._now_ms
    def get_last_processed_at(self):
        return self._last_at
    def get_pending(self):
        return []  # no stranded pending_new rows in these sweep tests


def _pubsub_args(storage):
    return SimpleNamespace(
        storage=storage, mode="pubsub", once=True,
        max_messages=50, k=5, dry_run=False, max_per_label=200,
    )


def test_startup_inbox_sweep_runs_on_legacy():
    client = _SweepClient()
    cal._run_pubsub_mode(
        _pubsub_args("legacy"), client, credentials=None,
        embedder=_FakeEmbedder(), index=object(), registry=object(),
        skip_ids=set(), backend=_CursorBackend(),
    )
    assert client.list_calls == 1  # legacy sweeps the inbox at startup


def test_startup_inbox_sweep_skipped_on_state():
    """State must not run the labeling inbox sweep -- it would label/archive
    pre-boundary backlog that history replay is designed to leave untouched."""
    client = _SweepClient()
    cal._run_pubsub_mode(
        _pubsub_args("state"), client, credentials=None,
        embedder=_FakeEmbedder(), index=object(), registry=object(),
        skip_ids=set(), backend=_CursorBackend(cursor="400"),
    )
    assert client.list_calls == 0  # no inbox sweep on the state path


class _ResyncClient:
    """A fuller state client: records watch/apply_label and serves a small
    mailbox so the read-only resync can list + reconcile. ``list_calls`` still
    tracks the labeling inbox sweep (must stay 0 on state)."""
    def __init__(self, labels=None, inbox=None):
        self._labels = labels or {}
        self._inbox = inbox or []
        self.list_calls = 0
        self.watch_calls = 0
        self.applied = []

    def watch(self, topic):
        self.watch_calls += 1
        return ("900", 10**18)

    def list_message_ids(self, label_id, max_results=0):
        # The labeling inbox sweep peeks INBOX; on state it must never run.
        if label_id == "INBOX":
            self.list_calls += 1
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
        _pubsub_args("state"), client, credentials=None,
        embedder=_FakeEmbedder(), index=index, registry=_Registry(),
        skip_ids=set(), backend=backend,
    )
    # Only the initial watch ran (the resume-from-cursor path re-watches once at
    # startup); no resync re-pin, and the boundary is unchanged.
    assert client.watch_calls == 1
    assert backend.store.get_meta("bootstrap_boundary_history_id") == "400"
    assert client.applied == []       # nothing labeled
    assert client.list_calls == 0     # no inbox sweep
    backend.close()


def test_state_past_window_runs_read_only_resync(tmp_path):
    """A cursor older than the window takes the read-only resync: it re-pins a
    fresh boundary and never labels/archives the accumulated inbox backlog."""
    path = str(tmp_path / "state.db")
    now = 10 * WARM_RECOVERY_WINDOW
    backend = _seed_warm_state(path, cursor="410", last_at=0, now_ms=now)
    # A backlog full of inbox mail that a naive replay-and-classify would archive.
    client = _ResyncClient(labels={"L_A": ("A", ["a1"])},
                           inbox=["i1", "i2", "i3"])
    index = TrainingIndex(np.empty((0, 0), dtype=np.float32), [], [])

    cal._run_pubsub_mode(
        _pubsub_args("state"), client, credentials=None,
        embedder=_FakeEmbedder(), index=index, registry=_Registry(),
        skip_ids=set(), backend=backend,
    )
    # Resync re-pinned the boundary to the fresh watch id, and NEVER labeled the
    # backlog.
    assert backend.store.get_meta("bootstrap_boundary_history_id") == "900"
    assert backend.store.get_last_processed_history_id() == "900"
    assert client.applied == []       # backlog never labeled/archived
    assert client.list_calls == 0     # no labeling inbox sweep
    # Store stays complete (repin must NOT reset to in_progress -> no re-bootstrap).
    assert backend.store.get_bootstrap_status() == "complete"
    backend.close()


def test_state_absent_cursor_resyncs_not_fails_closed(tmp_path):
    """A warm-looking state.db with no durable cursor no longer SystemExits
    (Phase 6 removed that fail-closed stop): it takes the read-only resync and
    starts cleanly from the fresh boundary. Replaces the old
    test_state_without_cursor_fails_closed."""
    path = str(tmp_path / "state.db")
    backend = _seed_warm_state(path, cursor=None, last_at=None, now_ms=1000)
    client = _ResyncClient(labels={"L_A": ("A", ["a1"])}, inbox=["i1"])
    index = TrainingIndex(np.empty((0, 0), dtype=np.float32), [], [])

    # Must NOT raise SystemExit.
    cal._run_pubsub_mode(
        _pubsub_args("state"), client, credentials=None,
        embedder=_FakeEmbedder(), index=index, registry=_Registry(),
        skip_ids=set(), backend=backend,
    )
    assert backend.store.get_last_processed_history_id() == "900"  # re-pinned
    assert client.applied == []       # never labels the backlog
    assert client.list_calls == 0     # no inbox sweep
    backend.close()


def test_legacy_without_cursor_uses_watch_boundary():
    """Legacy has no durable cursor and legitimately adopts the fresh watch id;
    it must not be caught by the state fail-closed guard."""
    client = _SweepClient()
    cal._run_pubsub_mode(
        _pubsub_args("legacy"), client, credentials=None,
        embedder=_FakeEmbedder(), index=object(), registry=object(),
        skip_ids=set(), backend=_CursorBackend(cursor=None),
    )
    assert client.list_calls == 1  # started normally, swept the inbox
