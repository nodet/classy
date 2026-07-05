"""Tests that --storage state is rejected outside pubsub mode.

Poll mode is the labeling inbox path: it lists the current INBOX and applies
labels with archive=True every interval. That is correct for legacy but unsafe
for the state backend, which must advance only via history replay from its
durable cursor and never archive pre-boundary backlog. --mode defaults to poll
while --storage/$CLASSY_STORAGE can be state, so the combination is rejected at
argument-validation time (before any Gmail I/O).
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


def test_state_poll_mode_is_rejected():
    args = SimpleNamespace(storage="state", mode="poll")
    with pytest.raises(SystemExit) as exc:
        cal._validate_storage_mode(args)
    assert exc.value.code == 1


def test_state_pubsub_mode_is_allowed():
    args = SimpleNamespace(storage="state", mode="pubsub")
    cal._validate_storage_mode(args)  # no raise


def test_legacy_poll_mode_is_allowed():
    args = SimpleNamespace(storage="legacy", mode="poll")
    cal._validate_storage_mode(args)  # no raise
