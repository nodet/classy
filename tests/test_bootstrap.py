"""Tests for bootstrap.py: cold bootstrap, ML rebuild, exclusion reconcile.

All drive fakes -- a Gmail client recording calls and returning canned message
resources, a fake embedder, and an on-disk StateStore (tmp_path). No network,
no FastEmbed. Guards the plan's "Unit -- bootstrap.py" and "Unit -- read-only
boundary (cold path safety)" gates for Phase 4.
"""
import numpy as np
import pytest

from gmail_classifier import bootstrap
from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.state_store import (
    STATE_SCHEMA_VERSION,
    StateStore,
    compute_ml_fingerprint,
)


class _FakeEmbedder:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    dimension = 8

    def __init__(self):
        self.embed_calls = 0

    def embed(self, text):
        self.embed_calls += 1
        # Deterministic non-zero vector; content-independent (fine for tests).
        return np.full(self.dimension, 1.0, dtype=np.float32)


class _FakeClient:
    """Records watch/list/get calls and serves canned message resources.

    ``labels`` maps label_id -> (label_name, [message_ids]); ``inbox`` is the
    list of INBOX ids. ``get_message`` returns a minimal Gmail resource.
    """

    def __init__(self, labels=None, inbox=None, watch_id="1000"):
        self._labels = labels or {}
        self._inbox = inbox or []
        self._watch_id = watch_id
        self.watch_calls = 0
        self.get_calls = []
        self.max_concurrent_bodies = 0
        self._live_bodies = 0

    # -- collaborators used by bootstrap --
    def watch(self, topic):
        self.watch_calls += 1
        return (self._watch_id, 10**18)

    def list_user_labels(self):
        return [(lid, name) for lid, (name, _ids) in self._labels.items()]

    def list_message_ids(self, label_id, max_results=0):
        if label_id == "INBOX":
            ids = self._inbox
        else:
            ids = self._labels.get(label_id, ("", []))[1]
        return list(ids[:max_results]) if max_results else list(ids)

    def get_message(self, mid):
        self.get_calls.append(mid)
        # Track that only one raw body is "live" at a time: bootstrap must embed
        # and discard before fetching the next. We can't see the discard
        # directly, so we approximate by asserting get is never re-entered.
        self._live_bodies += 1
        self.max_concurrent_bodies = max(self.max_concurrent_bodies, self._live_bodies)
        self._live_bodies -= 1
        return {"id": mid, "payload": {"headers": [], "body": {}}, "labelIds": []}


def _store(tmp_path, name="state.db"):
    return StateStore(str(tmp_path / name))


# --------------------------------------------------------------------------
# Cold bootstrap
# --------------------------------------------------------------------------

def test_bootstrap_builds_index_from_labels_and_skip(tmp_path):
    client = _FakeClient(
        labels={"L_A": ("A", ["a1", "a2"]), "L_B": ("B", ["b1"])},
        inbox=["i1", "i2"],
    )
    store = _store(tmp_path)
    bootstrap.bootstrap_index(
        client, _FakeEmbedder(), store,
        excluded=set(), max_per_label=200, topic="topic",
    )

    by_id = {mid: label for mid, _vec, label in store.iter_index()}
    assert by_id == {
        "a1": "A", "a2": "A", "b1": "B",
        "i1": SKIP_LABEL, "i2": SKIP_LABEL,
    }
    assert store.get_bootstrap_status() == "complete"
    assert store.get_meta("ml_fingerprint") == compute_ml_fingerprint(
        "sentence-transformers/all-MiniLM-L6-v2", 8)
    assert store.get_meta("state_schema_version") == STATE_SCHEMA_VERSION
    store.close()


def test_bootstrap_persists_account_id_before_complete(tmp_path):
    """gmail_account_id must be written by the time the store is marked complete,
    or decide_startup rejects the freshly built store as INCOMPATIBLE."""
    client = _FakeClient(labels={"L_A": ("A", ["a1"])}, inbox=[])
    store = _store(tmp_path)
    bootstrap.bootstrap_index(
        client, _FakeEmbedder(), store,
        excluded=set(), max_per_label=200, topic="topic",
        gmail_account_id="me@x.com",
    )
    assert store.get_bootstrap_status() == "complete"
    assert store.get_meta("gmail_account_id") == "me@x.com"
    store.close()


def test_bootstrap_watch_pins_boundary_before_any_get(tmp_path):
    """watch() must run before the first get_message, so the pinned historyId
    boundary reflects start-of-service, not end-of-bootstrap."""
    calls = []

    client = _FakeClient(labels={"L_A": ("A", ["a1"])}, inbox=[], watch_id="777")
    orig_watch, orig_get = client.watch, client.get_message
    client.watch = lambda t: (calls.append("watch"), orig_watch(t))[1]
    client.get_message = lambda mid: (calls.append("get"), orig_get(mid))[1]

    store = _store(tmp_path)
    bootstrap.bootstrap_index(
        client, _FakeEmbedder(), store,
        excluded=set(), max_per_label=200, topic="topic",
    )
    assert calls[0] == "watch"
    assert "get" in calls
    # Boundary + cursor are both pinned to the watch id.
    assert store.get_meta("bootstrap_boundary_history_id") == "777"
    assert store.get_last_processed_history_id() == "777"
    assert store.get_last_processed_at() is not None
    store.close()


def test_bootstrap_never_labels_or_archives(tmp_path):
    """Read-only boundary: bootstrap only reads mail to embed it; it must never
    call apply_label (there is no apply_label call anywhere in the path)."""
    client = _FakeClient(labels={"L_A": ("A", ["a1"])}, inbox=["i1"])
    # apply_label would raise if bootstrap ever tried to label/archive.
    client.apply_label = lambda *a, **k: pytest.fail("bootstrap must not label")
    store = _store(tmp_path)
    bootstrap.bootstrap_index(
        client, _FakeEmbedder(), store,
        excluded=set(), max_per_label=200, topic="topic",
    )
    store.close()


def test_bootstrap_excludes_configured_labels_at_source(tmp_path):
    client = _FakeClient(
        labels={"L_A": ("A", ["a1"]), "L_X": ("XLC", ["x1", "x2"])},
        inbox=[],
    )
    store = _store(tmp_path)
    bootstrap.bootstrap_index(
        client, _FakeEmbedder(), store,
        excluded={"XLC"}, max_per_label=200, topic="topic",
    )
    labels = {label for _mid, _vec, label in store.iter_index()}
    assert "XLC" not in labels
    assert store.known_ids() == {"a1"}  # x1/x2 never fetched
    assert "x1" not in client.get_calls and "x2" not in client.get_calls
    store.close()


def test_bootstrap_labeled_wins_over_skip(tmp_path):
    """An INBOX id that also carries a user label is recorded under the label,
    never __skip__."""
    client = _FakeClient(
        labels={"L_A": ("A", ["shared"])},
        inbox=["shared", "i1"],  # 'shared' is both labeled and in inbox
    )
    store = _store(tmp_path)
    bootstrap.bootstrap_index(
        client, _FakeEmbedder(), store,
        excluded=set(), max_per_label=200, topic="topic",
    )
    by_id = {mid: label for mid, _vec, label in store.iter_index()}
    assert by_id["shared"] == "A"
    assert by_id["i1"] == SKIP_LABEL
    assert store.skip_vote_ids() == {"i1"}
    store.close()


def test_bootstrap_max_per_label_bounds_each_set(tmp_path):
    client = _FakeClient(
        labels={"L_A": ("A", ["a1", "a2", "a3"])},
        inbox=["i1", "i2", "i3", "i4"],
    )
    store = _store(tmp_path)
    bootstrap.bootstrap_index(
        client, _FakeEmbedder(), store,
        excluded=set(), max_per_label=2, topic="topic",
    )
    # 2 from the label, 2 from the skip pool.
    labeled = [m for m, _v, l in store.iter_index() if l == "A"]
    skips = store.skip_vote_ids()
    assert len(labeled) == 2
    assert len(skips) == 2
    store.close()


def test_bootstrap_is_resumable_skips_embedded(tmp_path):
    """A second run does NOT re-fetch ids already embedded."""
    client = _FakeClient(labels={"L_A": ("A", ["a1", "a2"])}, inbox=[])
    emb = _FakeEmbedder()
    store = _store(tmp_path)
    bootstrap.bootstrap_index(client, emb, store, excluded=set(),
                              max_per_label=200, topic="topic")
    first_gets = list(client.get_calls)
    assert set(first_gets) == {"a1", "a2"}

    client.get_calls.clear()
    emb.embed_calls = 0
    # Re-run over the SAME store.
    bootstrap.bootstrap_index(client, emb, store, excluded=set(),
                              max_per_label=200, topic="topic")
    assert client.get_calls == []  # nothing re-fetched
    assert emb.embed_calls == 0    # nothing re-embedded
    store.close()


def test_bootstrap_resume_does_not_re_watch(tmp_path):
    """On resume (boundary already pinned) watch() is NOT called again -- moving
    the boundary would skip mail that arrived during the first attempt."""
    client = _FakeClient(labels={"L_A": ("A", ["a1"])}, inbox=[], watch_id="500")
    store = _store(tmp_path)
    # First (interrupted-then-complete) run pins the boundary.
    bootstrap.bootstrap_index(client, _FakeEmbedder(), store, excluded=set(),
                              max_per_label=200, topic="topic")
    assert client.watch_calls == 1

    bootstrap.bootstrap_index(client, _FakeEmbedder(), store, excluded=set(),
                              max_per_label=200, topic="topic")
    assert client.watch_calls == 1  # not re-watched
    assert store.get_meta("bootstrap_boundary_history_id") == "500"
    store.close()


def test_bootstrap_does_not_rewatch_when_boundary_set_but_status_missing(tmp_path):
    """Narrow crash window: a boundary is pinned but bootstrap_status is absent.
    watch() must NOT run again (that would move the boundary past mail that
    arrived after the first pin); the existing boundary is reused."""
    client = _FakeClient(labels={"L_A": ("A", ["a1"])}, inbox=[], watch_id="900")
    store = _store(tmp_path)
    # Simulate the crash state: boundary present, status never written.
    store.set_meta("bootstrap_boundary_history_id", "111")
    store.set_last_processed_history_id("111")
    assert store.get_bootstrap_status() is None

    bootstrap.bootstrap_index(client, _FakeEmbedder(), store, excluded=set(),
                              max_per_label=200, topic="topic")

    assert client.watch_calls == 0  # boundary already pinned -> no re-watch
    assert store.get_meta("bootstrap_boundary_history_id") == "111"  # unchanged
    store.close()


def test_pin_bootstrap_boundary_is_atomic(tmp_path):
    """pin_bootstrap_boundary writes boundary, cursor, and status together."""
    store = _store(tmp_path)
    store.pin_bootstrap_boundary("321")
    assert store.get_meta("bootstrap_boundary_history_id") == "321"
    assert store.get_last_processed_history_id() == "321"
    assert store.get_last_processed_at() is not None
    assert store.get_bootstrap_status() == "in_progress"
    store.close()


def test_bootstrap_one_body_at_a_time(tmp_path):
    client = _FakeClient(
        labels={"L_A": ("A", ["a1", "a2", "a3"])}, inbox=["i1", "i2"],
    )
    store = _store(tmp_path)
    bootstrap.bootstrap_index(client, _FakeEmbedder(), store, excluded=set(),
                              max_per_label=200, topic="topic")
    assert client.max_concurrent_bodies == 1
    store.close()


def test_bootstrap_front_loads_skip_pool(tmp_path, monkeypatch):
    """The first persisted rows are skip seeds, before the round-robin proper --
    so an interrupted boot already has skip mass for the confidence denominator.
    """
    monkeypatch.setattr(bootstrap, "SKIP_FRONTLOAD", 2)
    client = _FakeClient(
        labels={"L_A": ("A", ["a1", "a2"])},
        inbox=["i1", "i2", "i3"],
    )
    store = _store(tmp_path)
    bootstrap.bootstrap_index(client, _FakeEmbedder(), store, excluded=set(),
                              max_per_label=200, topic="topic")
    # The first two fetched ids are the front-loaded skip seeds.
    assert client.get_calls[:2] == ["i1", "i2"]
    store.close()


# --------------------------------------------------------------------------
# ML rebuild
# --------------------------------------------------------------------------

def test_rebuild_re_embeds_existing_labels_and_carries_cursor(tmp_path):
    old = _store(tmp_path, "state.db")
    # Seed an old store: labels + (stale) vectors + a cursor.
    old.upsert_label("a1", "L_A", "A", source="user")
    old.upsert_embedding("a1", np.full(8, 9.0, dtype=np.float32))
    old.upsert_label("s1", SKIP_LABEL, SKIP_LABEL, source="auto")
    old.upsert_embedding("s1", np.full(8, 9.0, dtype=np.float32))
    old.set_meta("gmail_account_id", "me@x.com")
    old.set_meta("state_schema_version", STATE_SCHEMA_VERSION)
    old.set_last_processed_history_id("4242")

    rebuild = _store(tmp_path, "state.rebuild.db")
    client = _FakeClient(labels={"L_A": ("A", ["a1"])}, inbox=[])
    bootstrap.rebuild_index(client, _FakeEmbedder(), old, rebuild)

    # Same label map, re-fetched from Gmail (so both ids re-embedded).
    by_id = {mid: label for mid, _v, label in rebuild.iter_index()}
    assert by_id == {"a1": "A", "s1": SKIP_LABEL}
    assert set(client.get_calls) == {"a1", "s1"}
    # Cursor + account carried forward; watch NOT called (no re-pin).
    assert rebuild.get_last_processed_history_id() == "4242"
    assert rebuild.get_meta("gmail_account_id") == "me@x.com"
    assert client.watch_calls == 0
    # New fingerprint written, marked complete.
    assert rebuild.get_meta("ml_fingerprint") == compute_ml_fingerprint(
        "sentence-transformers/all-MiniLM-L6-v2", 8)
    assert rebuild.get_bootstrap_status() == "complete"
    old.close()
    rebuild.close()


def test_atomic_swap_replaces_file(tmp_path):
    state_path = str(tmp_path / "state.db")
    rebuild_path = str(tmp_path / "state.rebuild.db")
    StateStore(state_path).close()
    r = StateStore(rebuild_path)
    r.set_meta("marker", "rebuilt")
    r.close()

    bootstrap.atomic_swap_state_db(state_path, rebuild_path)
    import os
    assert not os.path.exists(rebuild_path)
    reopened = StateStore(state_path)
    assert reopened.get_meta("marker") == "rebuilt"
    reopened.close()


# --------------------------------------------------------------------------
# Exclusion reconcile (cheap membership change, no re-embed of retained ids)
# --------------------------------------------------------------------------

def test_reconcile_removes_now_excluded_without_re_embedding(tmp_path):
    store = _store(tmp_path)
    store.upsert_label("a1", "L_A", "A", source="user")
    store.upsert_embedding("a1", np.full(8, 1.0, dtype=np.float32))
    store.upsert_label("b1", "L_B", "B", source="user")
    store.upsert_embedding("b1", np.full(8, 1.0, dtype=np.float32))

    emb = _FakeEmbedder()
    client = _FakeClient(labels={"L_A": ("A", ["a1"]), "L_B": ("B", ["b1"])})
    # Now exclude B.
    removed, added = bootstrap.reconcile_exclusions(
        client, emb, store, excluded={"B"}, max_per_label=200)

    assert removed == 1 and added == 0
    assert store.known_ids() == {"a1"}   # B row gone
    assert store.has_embedding("b1")     # embedding retained, just dropped from join
    assert client.get_calls == []        # nothing re-fetched
    assert emb.embed_calls == 0          # nothing re-embedded
    store.close()


def test_reconcile_bootstraps_newly_included_label(tmp_path):
    store = _store(tmp_path)
    store.upsert_label("a1", "L_A", "A", source="user")
    store.upsert_embedding("a1", np.full(8, 1.0, dtype=np.float32))

    emb = _FakeEmbedder()
    client = _FakeClient(labels={"L_A": ("A", ["a1"]), "L_C": ("C", ["c1", "c2"])})
    # Previously C was excluded; now it is included.
    removed, added = bootstrap.reconcile_exclusions(
        client, emb, store, excluded=set(), max_per_label=200)

    assert removed == 0 and added == 2
    by_id = {mid: label for mid, _v, label in store.iter_index()}
    assert by_id == {"a1": "A", "c1": "C", "c2": "C"}
    assert set(client.get_calls) == {"c1", "c2"}  # only the new label fetched
    store.close()


def test_round_robin_interleaves_labels():
    """Order cycles A,B,C,A,B,C -- so labels cross the eligibility line together
    rather than one label finishing before the next starts."""
    a = [("a1",), ("a2",), ("a3",)]
    b = [("b1",), ("b2",)]
    c = [("c1",)]
    order = [item[0] for item in bootstrap._round_robin([a, b, c])]
    assert order == ["a1", "b1", "c1", "a2", "b2", "a3"]
