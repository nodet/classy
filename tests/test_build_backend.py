"""Tests for the storage-backend selector in classify_and_label.

Focus: the state path validates a seeded state.db against the *actual* Gmail
account id (from the authenticated client), not against the store's own stored
value -- so a copied/stale DB from another mailbox is rejected.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from gmail_classifier.storage_state import (
    STATE_SCHEMA_VERSION,
    StateStore,
    compute_excluded_hash,
    compute_ml_fingerprint,
)

_SPEC = importlib.util.spec_from_file_location(
    "classify_and_label",
    Path(__file__).resolve().parent.parent / "scripts" / "classify_and_label.py",
)
cal = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cal)


class _FakeEmbedder:
    """Stand-in with the two attributes the ML fingerprint reads."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    dimension = 384


class _FakeClient:
    def __init__(self, email):
        self._email = email
        self.profile_calls = 0

    def get_profile_email(self):
        self.profile_calls += 1
        return self._email


def _seed_warm(path, *, account, excluded=frozenset()):
    """Seed a complete, fingerprint-matching state.db for the given account."""
    emb = _FakeEmbedder()
    store = StateStore(path)
    store.set_meta("bootstrap_status", "complete")
    store.set_meta("state_schema_version", STATE_SCHEMA_VERSION)
    store.set_meta("ml_fingerprint",
                   compute_ml_fingerprint(emb.model_name, emb.dimension))
    store.set_meta("excluded_labels_hash", compute_excluded_hash(excluded))
    store.set_meta("gmail_account_id", account)
    store.close()


def _args(state_db):
    return SimpleNamespace(state_db=state_db)


def test_state_warm_start_accepts_matching_account(tmp_path):
    path = str(tmp_path / "state.db")
    _seed_warm(path, account="me@gmail.com")
    client = _FakeClient("me@gmail.com")

    backend, plan = cal._build_backend(_args(path), set(), client, _FakeEmbedder())
    assert client.profile_calls == 1  # the ACTUAL mailbox was queried
    assert plan is None               # warm store: no bootstrap deferred
    backend.close()


def test_state_rejects_mismatched_account(tmp_path):
    """A state.db seeded for another account must NOT warm-start, even though
    schema/ML/exclusion all match -- guards against a copied/stale DB bringing
    the wrong label ids and history cursor."""
    path = str(tmp_path / "state.db")
    _seed_warm(path, account="other@gmail.com")
    client = _FakeClient("me@gmail.com")

    with pytest.raises(SystemExit) as exc:
        cal._build_backend(_args(path), set(), client, _FakeEmbedder())
    assert exc.value.code == 1
    assert client.profile_calls == 1


