"""Tests for the state backend: StateStore, StateBackend, and startup dispatch.

Drive an on-disk SQLite state.db (tmp_path) with a fake embedder -- no network,
no FastEmbed. Covers the plan's "Unit -- storage_state.py" and "Unit -- startup
dispatch" gates for Phase 3 (warm path only).
"""
import numpy as np
import pytest

from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.models import Message
from gmail_classifier.storage_state import (
    STATE_SCHEMA_VERSION,
    WARM_RECOVERY_WINDOW,
    GapDecision,
    StartupDecision,
    StateBackend,
    StateStore,
    classify_gap,
    compute_excluded_hash,
    compute_ml_fingerprint,
    decide_startup,
)


def _vec(seed, dim=8):
    """Deterministic non-zero vector."""
    return np.full(dim, float(seed), dtype=np.float32)


# --------------------------------------------------------------------------
# StateStore: labels / embeddings / join
# --------------------------------------------------------------------------

def test_upsert_label_is_last_write_wins(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store.upsert_label("m1", "Label_1", "Tech")
    store.upsert_label("m1", "Label_2", "Banque")

    # One row per id; the second upsert overwrites the first.
    known = store.known_ids()
    assert known == {"m1"}
    rows = list(store._conn.execute(
        "SELECT label_id, label_name FROM labels WHERE message_id = 'm1'"
    ))
    assert rows == [("Label_2", "Banque")]
    store.close()


def test_iter_index_only_joins_ids_in_both_tables(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))

    # m1: both embedding + label -> included.
    store.upsert_embedding("m1", _vec(1))
    store.upsert_label("m1", "Label_1", "Tech")
    # m2: embedding but no label -> excluded (orphan).
    store.upsert_embedding("m2", _vec(2))
    # m3: label but no embedding -> excluded (orphan).
    store.upsert_label("m3", "Label_1", "Tech")

    joined = {mid for mid, _, _ in store.iter_index()}
    assert joined == {"m1"}
    store.close()


def test_iter_index_maps_skip_rows_to_skip_label(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store.upsert_embedding("m1", _vec(1))
    store.upsert_label("m1", SKIP_LABEL, SKIP_LABEL, source="auto")
    store.upsert_embedding("m2", _vec(2))
    store.upsert_label("m2", "Label_1", "Tech")

    by_id = {mid: label for mid, _, label in store.iter_index()}
    assert by_id == {"m1": SKIP_LABEL, "m2": "Tech"}
    store.close()


def test_known_ids_includes_skip_and_real_labels(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store.upsert_label("m1", "Label_1", "Tech")
    store.upsert_label("m2", SKIP_LABEL, SKIP_LABEL, source="auto")

    assert store.known_ids() == {"m1", "m2"}
    assert store.skip_vote_ids() == {"m2"}
    store.close()


def test_embedded_ids_and_has_embedding(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store.upsert_embedding("m1", _vec(1))
    assert store.has_embedding("m1")
    assert not store.has_embedding("m2")
    assert store.embedded_ids() == {"m1"}
    store.close()


def test_empty_store_reports_empty(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    assert store.known_ids() == set()
    assert list(store.iter_index()) == []
    assert store.get_bootstrap_status() is None
    store.close()


# --------------------------------------------------------------------------
# StateStore: meta / fingerprint / durable cursor
# --------------------------------------------------------------------------

def test_meta_round_trips_and_fresh_is_none(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    assert store.get_meta("ml_fingerprint") is None
    store.set_meta("ml_fingerprint", "abc")
    assert store.get_meta("ml_fingerprint") == "abc"
    store.close()


def test_cursor_round_trips_and_stamps_timestamp(tmp_path):
    clock = [1000]
    store = StateStore(str(tmp_path / "state.db"), now_ms=lambda: clock[0])
    assert store.get_last_processed_history_id() is None
    assert store.get_last_processed_at() is None

    store.set_last_processed_history_id("500")
    assert store.get_last_processed_history_id() == "500"
    # last_processed_at is stamped in the SAME write as the cursor.
    assert store.get_last_processed_at() == 1000

    clock[0] = 2000
    store.set_last_processed_history_id("600")
    assert store.get_last_processed_at() == 2000
    store.close()


def test_cursor_is_durable_across_reopen(tmp_path):
    path = str(tmp_path / "state.db")
    store = StateStore(path)
    store.set_last_processed_history_id("777")
    store.close()

    reopened = StateStore(path)
    assert reopened.get_last_processed_history_id() == "777"
    reopened.close()


def test_malformed_timestamp_reads_as_none(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store.set_meta("last_processed_at", "not-a-number")
    assert store.get_last_processed_at() is None
    store.close()


def test_pending_new_insert_drain_idempotent(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store.add_pending("m1", "100", "immature")
    store.add_pending("m1", "100", "immature")  # duplicate -> no-op
    assert store.get_pending() == [("m1", "100", "immature")]

    store.remove_pending("m1")
    store.remove_pending("m1")  # already gone -> no-op
    assert store.get_pending() == []
    store.close()


# --------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------

def test_ml_fingerprint_changes_with_model_and_dim():
    a = compute_ml_fingerprint("model-x", 384)
    assert a == compute_ml_fingerprint("model-x", 384)
    assert a != compute_ml_fingerprint("model-y", 384)
    assert a != compute_ml_fingerprint("model-x", 512)


def test_excluded_hash_is_order_insensitive_and_normalized():
    assert compute_excluded_hash(["A", "B"]) == compute_excluded_hash(["B", "A"])
    assert compute_excluded_hash([" A ", "B"]) == compute_excluded_hash(["A", "B"])
    assert compute_excluded_hash([]) != compute_excluded_hash(["A"])
    # Empty/whitespace names are dropped.
    assert compute_excluded_hash(["A", "", "  "]) == compute_excluded_hash(["A"])


# --------------------------------------------------------------------------
# Startup dispatch (pure decision)
# --------------------------------------------------------------------------

def _seed_complete(store, *, ml, excluded_hash, schema=STATE_SCHEMA_VERSION,
                   account="acct-1"):
    store.set_meta("bootstrap_status", "complete")
    store.set_meta("state_schema_version", schema)
    store.set_meta("ml_fingerprint", ml)
    store.set_meta("excluded_labels_hash", excluded_hash)
    store.set_meta("gmail_account_id", account)


def test_dispatch_empty_store_bootstraps(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    d = decide_startup(store, schema_version=STATE_SCHEMA_VERSION,
                       ml_fingerprint="ml", excluded_hash="ex",
                       gmail_account_id="acct-1")
    assert d is StartupDecision.BOOTSTRAP
    store.close()


def test_dispatch_in_progress_bootstraps(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    store.set_meta("bootstrap_status", "in_progress")
    d = decide_startup(store, schema_version=STATE_SCHEMA_VERSION,
                       ml_fingerprint="ml", excluded_hash="ex",
                       gmail_account_id="acct-1")
    assert d is StartupDecision.BOOTSTRAP
    store.close()


def test_dispatch_all_match_is_warm(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    _seed_complete(store, ml="ml-1", excluded_hash="ex-1")
    d = decide_startup(store, schema_version=STATE_SCHEMA_VERSION,
                       ml_fingerprint="ml-1", excluded_hash="ex-1",
                       gmail_account_id="acct-1")
    assert d is StartupDecision.WARM
    store.close()


def test_dispatch_ml_mismatch_is_rebuild(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    _seed_complete(store, ml="ml-old", excluded_hash="ex-1")
    d = decide_startup(store, schema_version=STATE_SCHEMA_VERSION,
                       ml_fingerprint="ml-new", excluded_hash="ex-1",
                       gmail_account_id="acct-1")
    assert d is StartupDecision.REBUILD
    store.close()


def test_dispatch_excluded_mismatch_is_reconcile(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    _seed_complete(store, ml="ml-1", excluded_hash="ex-old")
    d = decide_startup(store, schema_version=STATE_SCHEMA_VERSION,
                       ml_fingerprint="ml-1", excluded_hash="ex-new",
                       gmail_account_id="acct-1")
    assert d is StartupDecision.RECONCILE
    store.close()


def test_dispatch_ml_checked_before_excluded(tmp_path):
    """A stale vector must be rebuilt regardless of membership -- ML wins."""
    store = StateStore(str(tmp_path / "state.db"))
    _seed_complete(store, ml="ml-old", excluded_hash="ex-old")
    d = decide_startup(store, schema_version=STATE_SCHEMA_VERSION,
                       ml_fingerprint="ml-new", excluded_hash="ex-new",
                       gmail_account_id="acct-1")
    assert d is StartupDecision.REBUILD
    store.close()


def test_dispatch_schema_mismatch_is_incompatible(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    _seed_complete(store, ml="ml-1", excluded_hash="ex-1", schema="old")
    d = decide_startup(store, schema_version=STATE_SCHEMA_VERSION,
                       ml_fingerprint="ml-1", excluded_hash="ex-1",
                       gmail_account_id="acct-1")
    assert d is StartupDecision.INCOMPATIBLE
    store.close()


def test_dispatch_account_mismatch_is_incompatible(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    _seed_complete(store, ml="ml-1", excluded_hash="ex-1", account="acct-1")
    d = decide_startup(store, schema_version=STATE_SCHEMA_VERSION,
                       ml_fingerprint="ml-1", excluded_hash="ex-1",
                       gmail_account_id="acct-2")
    assert d is StartupDecision.INCOMPATIBLE
    store.close()


def test_dispatch_missing_account_is_incompatible(tmp_path):
    """A store that never recorded its account id is not bound to this mailbox,
    so it must NOT warm-start even when every other fingerprint matches."""
    store = StateStore(str(tmp_path / "state.db"))
    _seed_complete(store, ml="ml-1", excluded_hash="ex-1", account="acct-1")
    store.set_meta("gmail_account_id", None)  # absent stored account
    d = decide_startup(store, schema_version=STATE_SCHEMA_VERSION,
                       ml_fingerprint="ml-1", excluded_hash="ex-1",
                       gmail_account_id="acct-1")
    assert d is StartupDecision.INCOMPATIBLE
    store.close()


def test_dispatch_empty_account_is_incompatible(tmp_path):
    """An empty stored account id is treated the same as a missing one."""
    store = StateStore(str(tmp_path / "state.db"))
    _seed_complete(store, ml="ml-1", excluded_hash="ex-1", account="")
    d = decide_startup(store, schema_version=STATE_SCHEMA_VERSION,
                       ml_fingerprint="ml-1", excluded_hash="ex-1",
                       gmail_account_id="acct-1")
    assert d is StartupDecision.INCOMPATIBLE
    store.close()


# --------------------------------------------------------------------------
# StateBackend: seam adapter (warm load + live-adaptation writes, no body)
# --------------------------------------------------------------------------

class _FakeEmbedder:
    """Warm path never embeds; this just proves load_index ignores it."""
    def embed(self, text):
        raise AssertionError("warm load_index must not embed")


def test_backend_load_index_builds_from_join(tmp_path):
    path = str(tmp_path / "state.db")
    store = StateStore(path)
    store.upsert_embedding("t1", _vec(1))
    store.upsert_label("t1", "Label_1", "Tech")
    store.upsert_embedding("t2", _vec(2))
    store.upsert_label("t2", "Label_1", "Tech")
    store.upsert_embedding("s1", _vec(3))
    store.upsert_label("s1", SKIP_LABEL, SKIP_LABEL, source="auto")
    store.close()

    backend = StateBackend(path, excluded=set())
    loaded = backend.load_index(_FakeEmbedder())

    assert loaded.stats.n_train == 2
    assert loaded.stats.n_skip == 1
    assert loaded.stats.n_dropped == 0
    assert loaded.skip_ids == {"t1", "t2", "s1"}
    assert set(loaded.index.labels) == {"Tech", SKIP_LABEL}
    assert loaded.index.embeddings.shape == (3, 8)
    backend.close()


def test_backend_load_index_empty_is_zero(tmp_path):
    backend = StateBackend(str(tmp_path / "state.db"), excluded=set())
    loaded = backend.load_index(_FakeEmbedder())
    assert loaded.stats.n_train == 0
    assert loaded.stats.n_skip == 0
    assert loaded.skip_ids == set()
    assert len(loaded.index) == 0
    backend.close()


def test_backend_upsert_label_stores_id_label_vec_no_body(tmp_path):
    path = str(tmp_path / "state.db")
    backend = StateBackend(path, excluded=set())
    msg = Message(id="m1", subject="secret subject", from_address="a@x.com",
                  body_html="<p>secret body</p>", labels=["Tech"])
    backend.upsert_label(msg, "Label_1", _vec(1))
    backend.close()

    # Reopen raw and prove no body/subject/sender column exists anywhere.
    store = StateStore(path)
    assert store.known_ids() == {"m1"}
    assert store.has_embedding("m1")
    row = store._conn.execute(
        "SELECT label_id, label_name FROM labels WHERE message_id='m1'"
    ).fetchone()
    assert row == ("Label_1", "Tech")
    # The labels table has no body/subject/sender columns at all.
    cols = {c[1] for c in store._conn.execute("PRAGMA table_info(labels)")}
    assert cols == {"message_id", "label_id", "label_name", "source"}
    store.close()


def test_backend_upsert_skip_records_skip_row(tmp_path):
    path = str(tmp_path / "state.db")
    backend = StateBackend(path, excluded=set())
    msg = Message(id="m1", subject="s", from_address="a@x.com", labels=[])
    backend.upsert_skip(msg, _vec(1))
    backend.close()

    store = StateStore(path)
    assert store.skip_vote_ids() == {"m1"}
    store.close()


def test_backend_upsert_label_supersedes_skip(tmp_path):
    """Labeled wins over skip: a later label upsert replaces the skip row."""
    path = str(tmp_path / "state.db")
    backend = StateBackend(path, excluded=set())
    msg = Message(id="m1", subject="s", from_address="a@x.com", labels=[])
    backend.upsert_skip(msg, _vec(1))
    msg.labels = ["Tech"]
    backend.upsert_label(msg, "Label_1", _vec(1))
    backend.close()

    store = StateStore(path)
    assert store.skip_vote_ids() == set()  # skip row gone
    assert store.known_ids() == {"m1"}
    store.close()


def test_backend_remove_drops_label_keeps_embedding(tmp_path):
    path = str(tmp_path / "state.db")
    backend = StateBackend(path, excluded=set())
    msg = Message(id="m1", subject="s", from_address="a@x.com", labels=["Tech"])
    backend.upsert_label(msg, "Label_1", _vec(1))
    backend.remove("m1")
    backend.close()

    store = StateStore(path)
    assert store.known_ids() == set()
    # Embedding cache is retained; it's just excluded from the join now.
    assert store.has_embedding("m1")
    store.close()


def test_backend_cursor_is_durable(tmp_path):
    """Unlike legacy, a fresh state adapter over the same file resumes."""
    path = str(tmp_path / "state.db")
    backend = StateBackend(path, excluded=set())
    assert backend.get_last_processed_history_id() is None
    backend.set_last_processed_history_id("999")
    backend.close()

    fresh = StateBackend(path, excluded=set())
    assert fresh.get_last_processed_history_id() == "999"
    fresh.close()


# --------------------------------------------------------------------------
# classify_gap: pure warm-restart recovery decision (Phase 6)
# --------------------------------------------------------------------------

def _warm_store_with_cursor(tmp_path, *, cursor="500", last_at=1000):
    """A store with a durable cursor + explicit last_processed_at, for the
    gap-decision cases (classify_gap reads only those two meta keys)."""
    store = StateStore(str(tmp_path / "state.db"))
    if cursor is not None:
        store.set_meta("last_processed_history_id", cursor)
    if last_at is not None:
        store.set_meta("last_processed_at", str(last_at))
    return store


def test_classify_gap_zero_is_replay(tmp_path):
    store = _warm_store_with_cursor(tmp_path, last_at=1000)
    assert classify_gap(store, now_ms=1000) is GapDecision.REPLAY
    store.close()


def test_classify_gap_at_window_boundary_is_replay(tmp_path):
    """The window boundary is inclusive: an exact-window gap is still the genuine
    short-outage case."""
    store = _warm_store_with_cursor(tmp_path, last_at=0)
    assert classify_gap(store, now_ms=WARM_RECOVERY_WINDOW) is GapDecision.REPLAY
    store.close()


def test_classify_gap_just_past_window_is_resync(tmp_path):
    store = _warm_store_with_cursor(tmp_path, last_at=0)
    assert classify_gap(store, now_ms=WARM_RECOVERY_WINDOW + 1) is GapDecision.RESYNC
    store.close()


def test_classify_gap_missing_timestamp_is_resync(tmp_path):
    """Cursor present but no last_processed_at -> no trustworthy gap -> fail safe."""
    store = _warm_store_with_cursor(tmp_path, last_at=None)
    assert store.get_last_processed_at() is None
    assert classify_gap(store, now_ms=1000) is GapDecision.RESYNC
    store.close()


def test_classify_gap_malformed_timestamp_is_resync(tmp_path):
    store = _warm_store_with_cursor(tmp_path, last_at=None)
    store.set_meta("last_processed_at", "not-a-number")
    assert classify_gap(store, now_ms=1000) is GapDecision.RESYNC
    store.close()


def test_classify_gap_future_timestamp_is_resync(tmp_path):
    """A last_processed_at ahead of now (clock skew) must not license replaying
    an unbounded gap."""
    store = _warm_store_with_cursor(tmp_path, last_at=5000)
    assert classify_gap(store, now_ms=1000) is GapDecision.RESYNC
    store.close()


def test_classify_gap_absent_cursor_is_resync(tmp_path):
    """The case that used to SystemExit: a warm-looking store with no cursor is
    now recovered read-only, not rejected."""
    store = _warm_store_with_cursor(tmp_path, cursor=None, last_at=1000)
    assert store.get_last_processed_history_id() is None
    assert classify_gap(store, now_ms=1000) is GapDecision.RESYNC
    store.close()


def test_repin_boundary_atomic_and_keeps_complete(tmp_path):
    """repin_boundary writes boundary + cursor + timestamp together and, unlike
    pin_bootstrap_boundary, leaves bootstrap_status == 'complete' (a re-pin on a
    finished store must not trigger a re-bootstrap on the next boot)."""
    clock = [7777]
    store = StateStore(str(tmp_path / "state.db"), now_ms=lambda: clock[0])
    store.set_meta("bootstrap_status", "complete")
    store.set_meta("bootstrap_boundary_history_id", "100")
    store.set_last_processed_history_id("100")

    store.repin_boundary("900")
    assert store.get_meta("bootstrap_boundary_history_id") == "900"
    assert store.get_last_processed_history_id() == "900"
    assert store.get_last_processed_at() == 7777
    # Crucially NOT flipped to in_progress.
    assert store.get_bootstrap_status() == "complete"
    store.close()


def test_backend_satisfies_storage_backend_protocol(tmp_path):
    from gmail_classifier.storage_backend import StorageBackend
    backend = StateBackend(str(tmp_path / "state.db"), excluded=set())
    assert isinstance(backend, StorageBackend)
    backend.close()


def test_loop_persist_cursor_is_durable_across_restart(tmp_path):
    """Integration: wiring StateBackend.set_last_processed_history_id as the
    loop's persist_cursor advances the cursor durably, so a fresh backend over
    the same file resumes from it (the crash-before-ack replay guarantee)."""
    from gmail_classifier.pubsub_loop import LoopState, LoopDeps, run_iteration
    from dataclasses import dataclass

    @dataclass
    class Notif:
        history_id: str

    class Sub:
        def __init__(self):
            self.acked = []
        def pull(self, timeout):
            return [Notif("200")], ["ack-200"]
        def ack(self, ack_ids):
            self.acked.append(list(ack_ids))
        def close(self):
            pass

    path = str(tmp_path / "state.db")
    backend = StateBackend(path, excluded=set())
    sub = Sub()

    deps = LoopDeps(
        make_subscriber=lambda: sub,
        watch=lambda: ("999", 10**18),
        get_history=lambda hid: ([object()], "555"),
        check_inbox=lambda: None,
        process_events=lambda evs: None,
        persist_cursor=backend.set_last_processed_history_id,
        log=lambda *a, **k: None,
        sleep=lambda s: None,
        now_ms=lambda: 0,
    )
    state = LoopState(history_id="100", expiration=10**18, backoff=0, subscriber=sub)
    run_iteration(state, deps)

    assert backend.get_last_processed_history_id() == "555"
    assert sub.acked == [["ack-200"]]
    backend.close()

    # A fresh adapter (a "restart") resumes from the durably-persisted cursor.
    fresh = StateBackend(path, excluded=set())
    assert fresh.get_last_processed_history_id() == "555"
    fresh.close()
