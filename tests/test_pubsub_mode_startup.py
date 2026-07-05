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

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "classify_and_label",
    Path(__file__).resolve().parent.parent / "scripts" / "classify_and_label.py",
)
cal = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cal)


class _FakeEmbedder:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    dimension = 384


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
    """A durable cursor stand-in. The state path fails closed without one, so a
    test that wants to reach the sweep gate must supply a resume cursor."""
    def __init__(self, cursor=None):
        self._cursor = cursor
    def get_last_processed_history_id(self):
        return self._cursor
    def set_last_processed_history_id(self, hid):
        self._cursor = hid


def _pubsub_args(storage):
    return SimpleNamespace(
        storage=storage, mode="pubsub", once=True,
        max_messages=50, k=5, dry_run=False,
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


def test_state_without_cursor_fails_closed():
    """A warm-looking state.db with no durable cursor must NOT start at the
    fresh watch boundary -- that would silently skip all history before the
    watch (and the state path also skips the startup sweep, so it would never
    be seen). Fail closed instead of adopting an arbitrary boundary."""
    client = _SweepClient()
    with pytest.raises(SystemExit):
        cal._run_pubsub_mode(
            _pubsub_args("state"), client, credentials=None,
            embedder=_FakeEmbedder(), index=object(), registry=object(),
            skip_ids=set(), backend=_CursorBackend(cursor=None),
        )
    assert client.list_calls == 0  # bailed before any inbox work


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
