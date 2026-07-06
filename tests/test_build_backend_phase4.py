"""Phase 4: _build_backend now runs the Gmail-backed state paths.

Where Phase 3 failed closed on anything but WARM, the state selector now:
- BOOTSTRAP -> cold-fetch from Gmail into the empty store, then return warm;
- RECONCILE -> cheap membership fix in place;
- REBUILD  -> build state.rebuild.db, atomically swap, reopen;
- INCOMPATIBLE -> still fails closed (schema/account mismatch, no safe recovery).

Drives _build_backend with the same fake client/embedder style as the other
selector tests, seeding state.db to steer decide_startup down each branch.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.state_store import (
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
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    dimension = 8

    def embed(self, text):
        return np.full(self.dimension, 1.0, dtype=np.float32)


class _FakeClient:
    def __init__(self, email="me@x.com", labels=None, inbox=None, watch_id="1000"):
        self._email = email
        self._labels = labels or {}
        self._inbox = inbox or []
        self._watch_id = watch_id
        self.watch_calls = 0
        self.get_calls = []

    def get_profile_email(self):
        return self._email

    def watch(self, topic):
        self.watch_calls += 1
        return (self._watch_id, 10**18)

    def list_user_labels(self):
        return [(lid, name) for lid, (name, _ids) in self._labels.items()]

    def list_message_ids(self, label_id, max_results=0):
        ids = self._inbox if label_id == "INBOX" else self._labels.get(label_id, ("", []))[1]
        return list(ids[:max_results]) if max_results else list(ids)

    def get_message(self, mid):
        self.get_calls.append(mid)
        return {"id": mid, "payload": {"headers": [], "body": {}}, "labelIds": []}


def _args(tmp_path, **over):
    base = dict(
        storage="state",
        state_db=str(tmp_path / "state.db"),
        training_db="unused", skip_db="unused",
        max_per_label=200,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _ml(emb):
    return compute_ml_fingerprint(emb.model_name, emb.dimension)


def _seed(path, **meta):
    store = StateStore(path)
    for k, v in meta.items():
        store.set_meta(k, v)
    return store


# --------------------------------------------------------------------------
# BOOTSTRAP: empty store -> cold fetch
# --------------------------------------------------------------------------

def test_bootstrap_decision_fetches_and_warms(tmp_path):
    client = _FakeClient(labels={"L_A": ("A", ["a1", "a2"])}, inbox=["i1"])
    emb = _FakeEmbedder()
    backend = cal._build_backend(_args(tmp_path), set(), client, emb)

    assert client.watch_calls == 1  # boundary pinned
    by_id = {mid: label for mid, _v, label in backend.store.iter_index()}
    assert by_id == {"a1": "A", "a2": "A", "i1": SKIP_LABEL}
    assert backend.store.get_bootstrap_status() == "complete"
    backend.close()


# --------------------------------------------------------------------------
# RECONCILE: excluded-label set changed -> cheap membership fix
# --------------------------------------------------------------------------

def test_reconcile_decision_updates_membership_in_place(tmp_path):
    path = str(tmp_path / "state.db")
    emb = _FakeEmbedder()
    store = _seed(
        path,
        bootstrap_status="complete",
        state_schema_version=STATE_SCHEMA_VERSION,
        ml_fingerprint=_ml(emb),
        gmail_account_id="me@x.com",
        excluded_labels_hash=compute_excluded_hash(set()),  # was: nothing excluded
    )
    store.upsert_label("a1", "L_A", "A", source="user")
    store.upsert_embedding("a1", np.full(8, 1.0, dtype=np.float32))
    store.upsert_label("b1", "L_B", "B", source="user")
    store.upsert_embedding("b1", np.full(8, 1.0, dtype=np.float32))
    store.close()

    client = _FakeClient(labels={"L_A": ("A", ["a1"]), "L_B": ("B", ["b1"])})
    # Now exclude B -> excluded hash changes -> RECONCILE.
    backend = cal._build_backend(_args(tmp_path), {"B"}, client, emb)

    assert backend.store.known_ids() == {"a1"}
    assert client.get_calls == []  # retained id not re-fetched
    assert backend.store.get_meta("excluded_labels_hash") == compute_excluded_hash({"B"})
    backend.close()


# --------------------------------------------------------------------------
# REBUILD: ML fingerprint changed -> rebuild + atomic swap
# --------------------------------------------------------------------------

def test_rebuild_decision_swaps_in_reembedded_store(tmp_path):
    path = str(tmp_path / "state.db")
    emb = _FakeEmbedder()
    store = _seed(
        path,
        bootstrap_status="complete",
        state_schema_version=STATE_SCHEMA_VERSION,
        ml_fingerprint="stale-fingerprint",   # != current -> REBUILD
        gmail_account_id="me@x.com",
        excluded_labels_hash=compute_excluded_hash(set()),
    )
    store.upsert_label("a1", "L_A", "A", source="user")
    store.upsert_embedding("a1", np.full(8, 9.0, dtype=np.float32))  # stale vec
    store.set_last_processed_history_id("4242")
    store.close()

    client = _FakeClient(labels={"L_A": ("A", ["a1"])})
    backend = cal._build_backend(_args(tmp_path), set(), client, emb)

    # Swapped-in store has the current fingerprint and carried the cursor.
    assert backend.store.get_meta("ml_fingerprint") == _ml(emb)
    assert backend.store.get_last_processed_history_id() == "4242"
    assert backend.store.known_ids() == {"a1"}
    assert client.watch_calls == 0  # rebuild must NOT re-pin a fresh boundary
    # The rebuild sidecar file was consumed by the swap.
    assert not (tmp_path / "state.rebuild.db").exists()
    backend.close()


# --------------------------------------------------------------------------
# INCOMPATIBLE: still fails closed
# --------------------------------------------------------------------------

def test_incompatible_account_still_fails_closed(tmp_path):
    path = str(tmp_path / "state.db")
    emb = _FakeEmbedder()
    _seed(
        path,
        bootstrap_status="complete",
        state_schema_version=STATE_SCHEMA_VERSION,
        ml_fingerprint=_ml(emb),
        gmail_account_id="other@x.com",  # mismatch
        excluded_labels_hash=compute_excluded_hash(set()),
    ).close()

    client = _FakeClient(email="me@x.com", labels={"L_A": ("A", ["a1"])})
    with pytest.raises(SystemExit) as exc:
        cal._build_backend(_args(tmp_path), set(), client, emb)
    assert exc.value.code == 1
    assert client.get_calls == []  # never fetched anything
