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


class _NoCursorBackend:
    def get_last_processed_history_id(self):
        return None
    def set_last_processed_history_id(self, hid):
        pass


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
        skip_ids=set(), backend=_NoCursorBackend(),
    )
    assert client.list_calls == 1  # legacy sweeps the inbox at startup


def test_startup_inbox_sweep_skipped_on_state():
    """State must not run the labeling inbox sweep -- it would label/archive
    pre-boundary backlog that history replay is designed to leave untouched."""
    client = _SweepClient()
    cal._run_pubsub_mode(
        _pubsub_args("state"), client, credentials=None,
        embedder=_FakeEmbedder(), index=object(), registry=object(),
        skip_ids=set(), backend=_NoCursorBackend(),
    )
    assert client.list_calls == 0  # no inbox sweep on the state path
