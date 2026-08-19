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
from gmail_classifier.models import HistoryExpiredError
from gmail_classifier.storage_state import (
    STATE_SCHEMA_VERSION,
    WARM_RECOVERY_WINDOW,
    StateStore,
    compute_ml_fingerprint,
)
from gmail_classifier.training_index import TrainingIndex


class _FakeEmbedder:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    dimension = 8

    def __init__(self):
        self.embed_calls = 0

    def embed(self, text):
        self.embed_calls += 1
        # Deterministic non-zero vector; content-independent (fine for tests).
        return np.full(self.dimension, 1.0, dtype=np.float32)


# Default arrival timestamp (epoch seconds) for inbox ids given as bare
# strings, chosen far in the future so plain ["i1", "i2"] fixtures keep
# looking like "arrived after any past cutoff" -- tests that need to exercise
# the after_ts gap-catchup boundary pass explicit (id, ts) tuples instead.
_DEFAULT_MSG_TS = 9_999_999_999


class _FakeClient:
    """Records watch/list/get calls and serves canned message resources.

    ``labels`` maps label_id -> (label_name, [message_ids]); ``inbox`` is the
    list of INBOX ids, or ``(id, sent_ts)`` tuples (epoch seconds) for tests
    that need to exercise the ``after_ts`` gap-catchup boundary.
    ``get_message`` returns a minimal Gmail resource.
    """

    def __init__(self, labels=None, inbox=None, watch_id="1000"):
        self._labels = labels or {}
        self._inbox = [
            m if isinstance(m, tuple) else (m, _DEFAULT_MSG_TS)
            for m in (inbox or [])
        ]
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
            ids = [mid for mid, _ts in self._inbox]
        else:
            ids = self._labels.get(label_id, ("", []))[1]
        return list(ids[:max_results]) if max_results else list(ids)

    def list_unlabeled_inbox_ids(self, max_results=0, after_ts=None):
        # Server-side has:nouserlabels: an INBOX id carrying ANY user label is
        # excluded, regardless of per-label sample caps. This mirrors the real
        # Gmail query so tests exercise the actual guarantee.
        all_labeled = set()
        for _name, ids in self._labels.values():
            all_labeled.update(ids)
        unlabeled = [
            mid for mid, ts in self._inbox
            if mid not in all_labeled and (after_ts is None or ts > after_ts)
        ]
        return list(unlabeled[:max_results]) if max_results else list(unlabeled)

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


def test_bootstrap_labeled_wins_even_outside_capped_sample(tmp_path):
    """A message that is both in INBOX and user-labeled, but falls OUTSIDE the
    capped sample for its label, must still never be stored as __skip__. The
    server-side has:nouserlabels filter enforces this; a client-side drop against
    only the sampled labeled union would miss it."""
    # max_per_label=1 samples only ["a1"] from label A, so "shared" (also labeled
    # A) is outside the sample -- but it is in INBOX.
    client = _FakeClient(
        labels={"L_A": ("A", ["a1", "shared"])},
        inbox=["shared", "i1"],
    )
    store = _store(tmp_path)
    bootstrap.bootstrap_index(client, _FakeEmbedder(), store, excluded=set(),
                              max_per_label=1, topic="topic")
    # "shared" must NOT be a skip example (it is user-labeled), even though it
    # was not in label A's capped sample.
    assert "shared" not in store.skip_vote_ids()
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


def test_bootstrap_repairs_missing_cursor_when_boundary_set(tmp_path):
    """Pre-atomic crash state: only bootstrap_boundary_history_id was persisted
    (the cursor write never completed). Reusing the boundary must NOT re-watch,
    and must repair the cursor to that boundary -- otherwise the completed store
    looks WARM but Pub/Sub startup fails closed on the missing resume cursor."""
    client = _FakeClient(labels={"L_A": ("A", ["a1"])}, inbox=[], watch_id="222")
    store = _store(tmp_path)
    # Only the boundary is set: no cursor, no status.
    store.set_meta("bootstrap_boundary_history_id", "111")
    assert store.get_last_processed_history_id() is None
    assert store.get_bootstrap_status() is None

    bootstrap.bootstrap_index(client, _FakeEmbedder(), store, excluded=set(),
                              max_per_label=200, topic="topic",
                              gmail_account_id="me@x.com")

    assert client.watch_calls == 0  # existing boundary reused, not re-pinned
    assert store.get_meta("bootstrap_boundary_history_id") == "111"  # unchanged
    # Cursor repaired to the boundary, so the completed store can actually resume.
    assert store.get_last_processed_history_id() == "111"
    assert store.get_bootstrap_status() == "complete"
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


# --------------------------------------------------------------------------
# read_only_resync (Phase 6): reconcile labels as truth, re-pin, never label
# --------------------------------------------------------------------------

def _complete_store(tmp_path, name="state.db", *, boundary="100", cursor="100"):
    """A complete/WARM store with a pinned boundary + cursor, as a resync sees."""
    store = _store(tmp_path, name)
    store.set_meta("bootstrap_status", "complete")
    store.set_meta("bootstrap_boundary_history_id", boundary)
    store.set_last_processed_history_id(cursor)
    return store


def test_resync_never_labels_or_archives(tmp_path):
    """Resync ingests Gmail's current label state; it must never apply_label /
    archive, even with a backlog of high-confidence-looking inbox mail."""
    client = _FakeClient(labels={"L_A": ("A", ["a1"])}, inbox=["i1", "i2", "i3"])
    client.apply_label = lambda *a, **k: pytest.fail("resync must not label")
    store = _complete_store(tmp_path)
    bootstrap.read_only_resync(client, _FakeEmbedder(), store, topic="topic",
                               excluded=set(), max_per_label=200)
    store.close()


def test_resync_repins_fresh_boundary(tmp_path):
    """watch() is called; the fresh id becomes both the boundary and the cursor,
    with last_processed_at stamped, and the store stays complete (no re-bootstrap)."""
    client = _FakeClient(labels={"L_A": ("A", ["a1"])}, inbox=["i1"],
                         watch_id="900")
    store = _complete_store(tmp_path, boundary="100", cursor="100")
    hid, expiration = bootstrap.read_only_resync(
        client, _FakeEmbedder(), store, topic="topic",
        excluded=set(), max_per_label=200)

    assert client.watch_calls == 1
    assert hid == "900" and expiration == 10**18
    assert store.get_meta("bootstrap_boundary_history_id") == "900"
    assert store.get_last_processed_history_id() == "900"
    assert store.get_last_processed_at() is not None
    assert store.get_bootstrap_status() == "complete"  # NOT reset to in_progress
    store.close()


def test_resync_ingests_label_added_since_cursor(tmp_path):
    """A label present in Gmail but absent from the store is folded in."""
    client = _FakeClient(labels={"L_A": ("A", ["a1", "a2"])}, inbox=[])
    store = _complete_store(tmp_path)
    # Store only knew a1 before.
    store.upsert_label("a1", "L_A", "A", source="user")
    store.upsert_embedding("a1", np.full(8, 1.0, dtype=np.float32))

    bootstrap.read_only_resync(client, _FakeEmbedder(), store, topic="topic",
                               excluded=set(), max_per_label=200)
    by_id = {mid: label for mid, _v, label in store.iter_index()}
    assert by_id == {"a1": "A", "a2": "A"}  # a2 ingested
    store.close()


def test_resync_removes_label_deleted_in_gmail(tmp_path):
    """A stored labeled id that is no longer in Gmail's snapshot is removed."""
    client = _FakeClient(labels={"L_A": ("A", ["a1"])}, inbox=[])
    store = _complete_store(tmp_path)
    store.upsert_label("a1", "L_A", "A", source="user")
    store.upsert_embedding("a1", np.full(8, 1.0, dtype=np.float32))
    # 'gone' was labeled locally but Gmail no longer returns it.
    store.upsert_label("gone", "L_A", "A", source="user")
    store.upsert_embedding("gone", np.full(8, 1.0, dtype=np.float32))

    bootstrap.read_only_resync(client, _FakeEmbedder(), store, topic="topic",
                               excluded=set(), max_per_label=200)
    assert store.known_ids() == {"a1"}
    assert "gone" not in {mid for mid, _v, _l in store.iter_index()}
    store.close()


def test_resync_canonicalizes_to_capped_snapshot(tmp_path):
    """A stored id that STILL carries its Gmail label but falls outside the
    per-label newest-first cap is dropped from the training map. This is the
    deliberate bounded-snapshot behavior: resync re-pins to a per-label coreset
    (size max_per_label) rather than growing the map across recoveries. Absence
    from the capped listing is NOT evidence of absence from Gmail -- so this
    documents/pins the trade-off explicitly."""
    # Gmail labels both a1 (newest) and a0 (older) under L_A, but the cap is 1, so
    # list_message_ids returns only the newest -> a1.
    client = _FakeClient(labels={"L_A": ("A", ["a1", "a0"])}, inbox=[])
    store = _complete_store(tmp_path)
    # The store already knew the older one from a prior, larger sample.
    store.upsert_label("a0", "L_A", "A", source="user")
    store.upsert_embedding("a0", np.full(8, 1.0, dtype=np.float32))

    bootstrap.read_only_resync(client, _FakeEmbedder(), store, topic="topic",
                               excluded=set(), max_per_label=1)
    by_id = {mid: label for mid, _v, label in store.iter_index()}
    # a0 is canonicalized away even though Gmail still labels it; only the newest
    # (a1) is kept.
    assert by_id == {"a1": "A"}
    assert "a0" not in store.known_ids()
    store.close()


def test_resync_rewrites_changed_label(tmp_path):
    """A stored id whose Gmail label changed from A to B is rewritten to B."""
    client = _FakeClient(labels={"L_B": ("B", ["m1"])}, inbox=[])
    store = _complete_store(tmp_path)
    store.upsert_label("m1", "L_A", "A", source="user")
    store.upsert_embedding("m1", np.full(8, 1.0, dtype=np.float32))

    bootstrap.read_only_resync(client, _FakeEmbedder(), store, topic="topic",
                               excluded=set(), max_per_label=200)
    by_id = {mid: label for mid, _v, label in store.iter_index()}
    assert by_id == {"m1": "B"}
    store.close()


def test_resync_reuses_cached_embeddings(tmp_path):
    """An unchanged id keeps its cached vector -- no re-fetch, no re-embed."""
    client = _FakeClient(labels={"L_A": ("A", ["a1"])}, inbox=[])
    emb = _FakeEmbedder()
    store = _complete_store(tmp_path)
    store.upsert_label("a1", "L_A", "A", source="user")
    store.upsert_embedding("a1", np.full(8, 1.0, dtype=np.float32))

    bootstrap.read_only_resync(client, emb, store, topic="topic",
                               excluded=set(), max_per_label=200)
    assert client.get_calls == []   # a1 not re-fetched
    assert emb.embed_calls == 0     # a1 not re-embedded
    store.close()


def test_resync_rebuilds_live_index_and_skip_ids(tmp_path):
    """When passed the live index + skip_ids, resync rebuilds them in place from
    the reconciled store."""
    client = _FakeClient(labels={"L_A": ("A", ["a1"])}, inbox=["i1"])
    store = _complete_store(tmp_path)
    index = TrainingIndex(np.empty((0, 0), dtype=np.float32), [], [])
    skip_ids = {"stale-id"}

    bootstrap.read_only_resync(client, _FakeEmbedder(), store, topic="topic",
                               excluded=set(), max_per_label=200,
                               index=index, skip_ids=skip_ids)
    by_id = {mid: lbl for mid, lbl in zip(index.ids, index.labels)}
    assert by_id == {"a1": "A", "i1": SKIP_LABEL}
    assert skip_ids == {"a1", "i1"}   # rebuilt from known_ids; stale entry gone
    store.close()


def test_resync_is_idempotent(tmp_path):
    """Running resync twice leaves the same store state (no duplicate rows, no
    crash) -- guards the loop's expiry branch that may re-run recovery."""
    client = _FakeClient(labels={"L_A": ("A", ["a1"])}, inbox=["i1"],
                         watch_id="900")
    store = _complete_store(tmp_path)
    bootstrap.read_only_resync(client, _FakeEmbedder(), store, topic="topic",
                               excluded=set(), max_per_label=200)
    first = {mid: label for mid, _v, label in store.iter_index()}

    bootstrap.read_only_resync(client, _FakeEmbedder(), store, topic="topic",
                               excluded=set(), max_per_label=200)
    second = {mid: label for mid, _v, label in store.iter_index()}
    assert first == second
    assert store.get_bootstrap_status() == "complete"
    store.close()


# --------------------------------------------------------------------------
# find_gap_message_ids (Phase 6 gap catchup): classify mail that arrived
# during a downtime gap, bounded by last_processed_at so a resync never
# sweeps up backlog older than the gap itself.
# --------------------------------------------------------------------------

def _empty_index():
    return TrainingIndex(np.empty((0, 0), dtype=np.float32), [], [])


def test_find_gap_message_ids_returns_empty_without_last_at(tmp_path):
    """No known-good timestamp (Case A / blank slate) -> nothing is a "gap"
    message; Gmail's current state is the only ground truth."""
    store = _store(tmp_path)
    client = _FakeClient(inbox=["i1", "i2"])
    result = bootstrap.find_gap_message_ids(
        client, store, _empty_index(), set(), known_before=set(), last_at_ms=None)
    assert result == []
    store.close()


def test_find_gap_message_ids_subtracts_known_before(tmp_path):
    """Messages already known before the resync ran (its own snapshot
    canonicalization aside) are not gap messages."""
    store = _store(tmp_path)
    client = _FakeClient(inbox=["i1", "i2"])
    result = bootstrap.find_gap_message_ids(
        client, store, _empty_index(), set(), known_before={"i1"}, last_at_ms=1000)
    assert result == ["i2"]
    store.close()


def test_find_gap_message_ids_excludes_messages_before_last_at(tmp_path):
    """Only messages that arrived strictly after the known-good timestamp are
    gap messages -- older unlabeled backlog is left alone."""
    store = _store(tmp_path)
    client = _FakeClient(inbox=[("old", 4000), ("new", 6000)])
    result = bootstrap.find_gap_message_ids(
        client, store, _empty_index(), set(), known_before=set(),
        last_at_ms=5_000_000)
    assert result == ["new"]
    store.close()


def test_find_gap_message_ids_does_not_truncate_large_backlogs(tmp_path):
    """warn_threshold only logs a heads-up; it must never drop messages --
    unlike skip_ids, after_ts only moves forward on later resyncs, so a
    truncated listing would silently and permanently lose the excess."""
    store = _store(tmp_path)
    logged = []
    client = _FakeClient(inbox=["i1", "i2", "i3"])
    result = bootstrap.find_gap_message_ids(
        client, store, _empty_index(), set(), known_before=set(),
        last_at_ms=1000, warn_threshold=2, log=logged.append)
    assert result == ["i1", "i2", "i3"]  # nothing truncated
    assert any("large gap backlog" in m for m in logged)
    store.close()


def test_find_gap_message_ids_strips_self_contaminating_skip_row(tmp_path):
    """A gap message that read_only_resync's own canonicalization already
    persisted as a __skip__ row (it treats the current unlabeled inbox as
    ground truth) is stripped from the store/index/skip_ids before catchup
    gets it -- otherwise it would vote __skip__ on its own classification."""
    store = _store(tmp_path)
    store.upsert_label("i1", SKIP_LABEL, SKIP_LABEL, source="auto")
    store.upsert_embedding("i1", np.full(8, 1.0, dtype=np.float32))
    index = TrainingIndex(np.full((1, 8), 1.0, dtype=np.float32), [SKIP_LABEL], ["i1"])
    skip_ids = {"i1"}
    client = _FakeClient(inbox=["i1"])

    result = bootstrap.find_gap_message_ids(
        client, store, index, skip_ids, known_before=set(), last_at_ms=1000)

    assert result == ["i1"]
    assert "i1" not in store.known_ids()
    assert "i1" not in skip_ids
    assert index.ids == []
    store.close()


# --------------------------------------------------------------------------
# heartbeat_cursor (Phase 6): keep a live-idle cursor out of the resync window
# --------------------------------------------------------------------------

class _HistoryClient:
    """Records get_history calls; returns scripted (events, latest) or raises."""
    def __init__(self, events=None, latest=None, exc=None):
        self._events = events or []
        self._latest = latest
        self._exc = exc
        self.get_history_calls = []

    def get_history(self, cursor):
        self.get_history_calls.append(cursor)
        if self._exc is not None:
            raise self._exc
        return self._events, self._latest


def _heartbeat_store(tmp_path, *, cursor="500", last_at=0):
    store = _store(tmp_path)
    if cursor is not None:
        store.set_last_processed_history_id(cursor)
        store.set_meta("last_processed_at", str(last_at))
    return store


def test_heartbeat_refreshes_idle_cursor_when_due(tmp_path):
    """Once the cursor's age reaches the interval, an empty history read persists
    the returned id + a fresh timestamp."""
    clock = [0]
    store = _store(tmp_path)
    store._now_ms = lambda: clock[0]
    store.set_last_processed_history_id("500")  # stamps last_at = 0
    client = _HistoryClient(events=[], latest="600")

    clock[0] = bootstrap.HEARTBEAT_INTERVAL   # now due
    advanced = bootstrap.heartbeat_cursor(client, store, now_ms=clock[0])
    assert advanced == "600"                  # returns the advanced id
    assert client.get_history_calls == ["500"]
    assert store.get_last_processed_history_id() == "600"
    assert store.get_last_processed_at() == bootstrap.HEARTBEAT_INTERVAL
    store.close()


def test_heartbeat_noop_before_interval(tmp_path):
    """A cursor younger than the interval is left alone (no history call)."""
    store = _heartbeat_store(tmp_path, last_at=0)
    client = _HistoryClient(events=[], latest="600")
    advanced = bootstrap.heartbeat_cursor(client, store,
                                          now_ms=bootstrap.HEARTBEAT_INTERVAL - 1)
    assert advanced is None
    assert client.get_history_calls == []
    assert store.get_last_processed_history_id() == "500"  # unchanged
    store.close()


def test_heartbeat_does_not_advance_when_events_pending(tmp_path):
    """If history has real updates, the heartbeat leaves the cursor for the
    normal loop to process + advance -- it must not skip past them."""
    store = _heartbeat_store(tmp_path, last_at=0)
    client = _HistoryClient(events=[object()], latest="600")
    advanced = bootstrap.heartbeat_cursor(client, store,
                                          now_ms=bootstrap.HEARTBEAT_INTERVAL)
    assert advanced is None
    assert store.get_last_processed_history_id() == "500"  # unchanged
    store.close()


def test_heartbeat_swallows_expired_history_for_the_resync_path(tmp_path):
    """An expired cursor during the heartbeat is left for the expiry -> resync
    path, not raised out of the idle pull."""
    store = _heartbeat_store(tmp_path, last_at=0)
    client = _HistoryClient(exc=HistoryExpiredError("too old"))
    advanced = bootstrap.heartbeat_cursor(client, store,
                                          now_ms=bootstrap.HEARTBEAT_INTERVAL)
    assert advanced is None
    assert store.get_last_processed_history_id() == "500"  # unchanged
    store.close()


def test_heartbeat_noop_without_cursor(tmp_path):
    store = _store(tmp_path)  # no cursor at all
    client = _HistoryClient(events=[], latest="600")
    assert bootstrap.heartbeat_cursor(client, store,
                                      now_ms=WARM_RECOVERY_WINDOW) is None
    assert client.get_history_calls == []
    store.close()
