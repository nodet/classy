"""Tests for the state-backend status report (Phase 7 ops).

Drive an in-memory-ish on-disk SQLite state.db (tmp_path) with a fake embedder --
no Gmail, no FastEmbed. Covers the "Status" verification: gcp-state-status reports
active backend, schema/fingerprint, bootstrap status, index size, per-label
counts, skip count, pending count, and the history cursor + last_processed_at.
"""
import numpy as np

from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.state_status import (
    format_report,
    gather_status,
    print_report,
)
from gmail_classifier.state_store import StateStore


def _vec(seed, dim=8):
    return np.full(dim, float(seed), dtype=np.float32)


def _seed_store(path):
    """A populated, WARM-looking store: two labels, a skip row, a pending row,
    an orphan embedding (no label) and an orphan label (no embedding), plus meta."""
    store = StateStore(str(path), now_ms=lambda: 1_000_000)
    # Two "Banque" + one "github" labeled rows, all embedded.
    store.upsert_embedding("b1", _vec(1)); store.upsert_label("b1", "L_BANK", "Banque")
    store.upsert_embedding("b2", _vec(2)); store.upsert_label("b2", "L_BANK", "Banque")
    store.upsert_embedding("g1", _vec(3)); store.upsert_label("g1", "L_GH", "github")
    # One skip row, embedded.
    store.upsert_embedding("s1", _vec(4)); store.upsert_label("s1", SKIP_LABEL, SKIP_LABEL)
    # Orphans: excluded from the join, so they must NOT count.
    store.upsert_embedding("orphan_emb", _vec(5))   # embedding, no label
    store.upsert_label("orphan_lab", "L_BANK", "Banque")  # label, no embedding
    # One parked (immature) message.
    store.add_pending("p1", "500")
    # Meta the report surfaces.
    store.set_meta("state_schema_version", "1")
    store.set_meta("ml_fingerprint", "model=x|dim=8")
    store.set_meta("gmail_account_id", "me@example.com")
    store.set_meta("excluded_labels_hash", "abc123")
    store.set_meta("bootstrap_status", "complete")
    store.set_meta("bootstrap_boundary_history_id", "400")
    store.set_last_processed_history_id("600")
    return store


def test_gather_status_counts_only_join_rows(tmp_path):
    store = _seed_store(tmp_path / "state.db")
    try:
        status = gather_status(store, excluded_config=["XLC", "XLE"])
    finally:
        store.close()

    # Orphans (orphan_emb, orphan_lab) are excluded exactly as iter_index does.
    assert status.per_label_counts == {"Banque": 2, "github": 1}
    assert status.skip_count == 1
    assert status.index_size == 4  # 2 + 1 labeled + 1 skip, no orphans
    assert status.pending_count == 1


def test_gather_status_surfaces_meta_and_cursor(tmp_path):
    store = _seed_store(tmp_path / "state.db")
    try:
        status = gather_status(store, excluded_config=["XLE", "XLC"])
    finally:
        store.close()

    assert status.schema_version == "1"
    assert status.ml_fingerprint == "model=x|dim=8"
    assert status.gmail_account_id == "me@example.com"
    assert status.bootstrap_status == "complete"
    assert status.excluded_labels_hash == "abc123"
    # Config list is normalized (sorted) for stable display.
    assert status.excluded_labels_config == ["XLC", "XLE"]
    assert status.bootstrap_boundary_history_id == "400"
    assert status.last_processed_history_id == "600"
    # set_last_processed_history_id stamps last_processed_at from the store clock.
    assert status.last_processed_at == 1_000_000


def test_format_report_includes_key_fields(tmp_path):
    store = _seed_store(tmp_path / "state.db")
    try:
        status = gather_status(store, excluded_config=["XLC"])
    finally:
        store.close()

    text = format_report(status)
    assert "Storage backend: state" in text
    assert "bootstrap status:   complete" in text
    assert "Banque" in text and "github" in text
    assert "Skip examples:        1" in text
    assert "Pending (immature):   1" in text
    assert "last processed id:  600" in text
    # last_processed_at rendered as a UTC timestamp, not a bare number alone.
    assert "1970-01-01" in text  # 1_000_000 ms after epoch


def test_gather_status_empty_store(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    try:
        status = gather_status(store, excluded_config=[])
    finally:
        store.close()
    assert status.index_size == 0
    assert status.per_label_counts == {}
    assert status.skip_count == 0
    assert status.pending_count == 0
    assert status.bootstrap_status is None
    # Renders without error on a fresh store.
    assert "Index size:           0" in format_report(status)


def test_print_report_missing_db_does_not_create_file(tmp_path, capsys):
    missing = tmp_path / "nope.db"
    print_report(str(missing), excluded_config=[])
    out = capsys.readouterr().out
    assert "No state database" in out
    # Guard: a status check must not materialize a fresh (empty) state.db, which
    # would then read as bootstrap-needed on the next real boot.
    assert not missing.exists()


def test_print_report_existing_db(tmp_path, capsys):
    _seed_store(tmp_path / "state.db").close()
    print_report(str(tmp_path / "state.db"), excluded_config=["XLC"])
    out = capsys.readouterr().out
    assert "Storage backend: state" in out
    assert "Banque" in out
