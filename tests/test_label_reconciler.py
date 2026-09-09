"""Tests for live label reconciliation (deletion + rename detection)."""
import numpy as np
from unittest.mock import MagicMock, patch

from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.label_reconciler import reconcile_labels
from gmail_classifier.label_registry import LabelDiff, LabelRegistry
from gmail_classifier.storage_state import StateStore
from gmail_classifier.training_index import TrainingIndex


def _make_registry(labels, excluded=None):
    client = MagicMock()
    client.list_user_labels.return_value = labels
    return LabelRegistry(client, excluded=excluded or set())


def _make_store(label_rows=None, embeddings=None):
    """Create a mock StateStore with configurable behavior.

    ``label_rows`` is keyed by Gmail label id (not name) -- matching
    ``message_ids_by_label_id``, the id-keyed method both the rename and
    delete branches now use."""
    store = MagicMock()
    label_rows = label_rows or {}
    embeddings = embeddings or {}
    store.message_ids_by_label_id.side_effect = lambda lid: label_rows.get(lid, set())
    store.get_embedding.side_effect = lambda mid: embeddings.get(mid)
    store.rename_label_by_id.return_value = 0
    store.remove_labels_by_name.return_value = 0
    return store


def _make_index(labels=None, ids=None):
    n = len(labels) if labels else 0
    embs = np.random.randn(n, 384).astype(np.float32) if n else np.empty((0, 0))
    return TrainingIndex(embs, labels or [], ids or [])


def test_no_change_is_noop():
    registry = _make_registry([("L1", "Tech"), ("L2", "Travel")])
    store = _make_store()
    index = _make_index(["Tech", "Travel"], ["m1", "m2"])
    client = MagicMock()
    skip_ids = set()
    log = MagicMock()

    reconcile_labels(registry, store, index, client, skip_ids, log)

    store.rename_label_by_id.assert_not_called()
    store.remove_labels_by_name.assert_not_called()
    client.move_to_inbox.assert_not_called()
    log.assert_not_called()


def test_deleted_label_purges_and_reinboxes():
    registry = _make_registry([("L1", "Tech"), ("L2", "Travel")])
    vec = np.random.randn(384).astype(np.float32)
    store = _make_store(
        label_rows={"L2": {"m3", "m4"}},
        embeddings={"m3": vec, "m4": vec},
    )
    index = _make_index(
        ["Tech", "Travel", "Travel"],
        ["m1", "m3", "m4"],
    )
    client = MagicMock()
    skip_ids = set()
    log = MagicMock()

    # Simulate deletion of Travel (L2 disappears on next refresh)
    registry._client.list_user_labels.return_value = [("L1", "Tech")]

    reconcile_labels(registry, store, index, client, skip_ids, log)

    # Messages re-inboxed
    assert client.move_to_inbox.call_count == 2
    moved = {c.args[0] for c in client.move_to_inbox.call_args_list}
    assert moved == {"m3", "m4"}

    # Store: each message's row replaced in place by the skip upsert (its
    # PRIMARY KEY means no separate bulk-delete is needed beforehand).
    store.remove_labels_by_name.assert_not_called()
    assert store.upsert_label.call_count == 2
    for call in store.upsert_label.call_args_list:
        assert call.args[1] == SKIP_LABEL

    # Index: entries converted to skip
    assert "m3" in index
    assert "m4" in index
    assert all(l == SKIP_LABEL for l in index.labels if l != "Tech")

    # skip_ids updated
    assert skip_ids == {"m3", "m4"}


def test_renamed_label_updates_store_and_index():
    registry = _make_registry([("L1", "Tech"), ("L2", "Travel")])
    store = _make_store(label_rows={"L2": {"m2", "m3"}})
    store.rename_label_by_id.return_value = 3
    index = _make_index(
        ["Tech", "Travel", "Travel"],
        ["m1", "m2", "m3"],
    )
    client = MagicMock()
    skip_ids = set()
    log = MagicMock()

    # Simulate rename: L2 "Travel" -> "Voyages"
    registry._client.list_user_labels.return_value = [("L1", "Tech"), ("L2", "Voyages")]

    reconcile_labels(registry, store, index, client, skip_ids, log)

    store.rename_label_by_id.assert_called_once_with("L2", "Voyages")
    client.move_to_inbox.assert_not_called()
    store.remove_labels_by_name.assert_not_called()

    # In-memory index updated
    assert "Travel" not in index.labels
    assert index.labels.count("Voyages") == 2


def test_move_to_inbox_failure_continues(caplog):
    registry = _make_registry([("L1", "Tech"), ("L2", "Travel")])
    vec = np.random.randn(384).astype(np.float32)
    store = _make_store(
        label_rows={"L2": {"m3", "m4"}},
        embeddings={"m3": vec, "m4": vec},
    )
    index = _make_index(["Tech", "Travel", "Travel"], ["m1", "m3", "m4"])
    client = MagicMock()
    client.move_to_inbox.side_effect = [Exception("gone"), None]
    skip_ids = set()
    log = MagicMock()

    registry._client.list_user_labels.return_value = [("L1", "Tech")]

    with caplog.at_level("WARNING"):
        reconcile_labels(registry, store, index, client, skip_ids, log)

    # The failure is surfaced (not silently swallowed at debug level), and
    # the message that failed to move is still marked skip -- see next
    # assertions -- but the summary log line flags it so it's discoverable.
    assert "move_to_inbox failed" in caplog.text
    summary = log.call_args.args[0]
    assert "1 FAILED to move" in summary

    # Both messages still cleaned up in store despite one API failure
    store.remove_labels_by_name.assert_not_called()
    assert store.upsert_label.call_count == 2
    assert skip_ids == {"m3", "m4"}


def test_deleted_label_with_no_stored_messages():
    registry = _make_registry([("L1", "Tech"), ("L2", "Travel")])
    store = _make_store(label_rows={"L2": set()})
    index = _make_index(["Tech"], ["m1"])
    client = MagicMock()
    skip_ids = set()
    log = MagicMock()

    registry._client.list_user_labels.return_value = [("L1", "Tech")]

    reconcile_labels(registry, store, index, client, skip_ids, log)

    client.move_to_inbox.assert_not_called()
    store.remove_labels_by_name.assert_not_called()
    log.assert_called_once()
    assert "no stored messages" in log.call_args.args[0]


def test_mixed_delete_and_rename():
    registry = _make_registry([("L1", "Tech"), ("L2", "Travel"), ("L3", "News")])
    vec = np.random.randn(384).astype(np.float32)
    store = _make_store(
        label_rows={"L2": {"m2", "m3"}, "L3": {"m5"}},
        embeddings={"m5": vec},
    )
    store.rename_label_by_id.return_value = 2
    index = _make_index(
        ["Tech", "Travel", "Travel", "News"],
        ["m1", "m2", "m3", "m5"],
    )
    client = MagicMock()
    skip_ids = set()
    log = MagicMock()

    # L2 renamed to "Voyages", L3 deleted
    registry._client.list_user_labels.return_value = [("L1", "Tech"), ("L2", "Voyages")]

    reconcile_labels(registry, store, index, client, skip_ids, log)

    # Rename happened
    store.rename_label_by_id.assert_called_once_with("L2", "Voyages")
    assert index.labels.count("Voyages") == 2

    # Deletion happened
    client.move_to_inbox.assert_called_once_with("m5")
    store.remove_labels_by_name.assert_not_called()
    assert "m5" in skip_ids


def test_deleted_label_crash_mid_loop_leaves_original_labels_intact(tmp_path):
    """A crash partway through the skip-marking loop must roll back to every
    message's ORIGINAL label -- never a mix of skip/erased/original. Uses a
    real StateStore (not the MagicMock fixture above) because a mocked
    transaction() would silently swallow the injected exception instead of
    letting it propagate, which is exactly the behavior under test."""
    store = StateStore(str(tmp_path / "state.db"))
    vec = np.random.randn(384).astype(np.float32)
    for mid in ("m1", "m2", "m3"):
        store.upsert_label(mid, "L2", "Travel", source="user")
        store.upsert_embedding(mid, vec)

    registry = _make_registry([("L1", "Tech"), ("L2", "Travel")])
    registry._client.list_user_labels.return_value = [("L1", "Tech")]  # Travel deleted
    index = _make_index(["Travel", "Travel", "Travel"], ["m1", "m2", "m3"])
    client = MagicMock()
    skip_ids = set()
    log = MagicMock()

    real_upsert = store.upsert_label
    calls = {"n": 0}

    def flaky_upsert(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash mid-loop")
        return real_upsert(*args, **kwargs)

    with patch.object(store, "upsert_label", side_effect=flaky_upsert):
        try:
            reconcile_labels(registry, store, index, client, skip_ids, log)
            assert False, "expected the simulated crash to propagate"
        except RuntimeError:
            pass

    # Rolled back: every message still shows its ORIGINAL label, none erased
    # and none left half-converted to skip.
    assert store.message_ids_by_label("Travel") == {"m1", "m2", "m3"}
    assert store.known_ids() == {"m1", "m2", "m3"}
    store.close()


def test_rename_onto_a_freed_name_does_not_destroy_the_renamed_messages(tmp_path):
    """Bug D: L2 is renamed Travel->News in the same pass that a *different*,
    unrelated label L4 (already named News) is deleted. Detection is by
    Gmail label id and correctly tells the two apart; before the fix, the
    delete branch's name-keyed message_ids_by_label("News") would have swept
    up L2's just-renamed messages too and destroyed them as if orphaned."""
    store = StateStore(str(tmp_path / "state.db"))
    vec = np.random.randn(384).astype(np.float32)
    for mid in ("m1", "m2"):
        store.upsert_label(mid, "L2", "Travel", source="user")
        store.upsert_embedding(mid, vec)
    store.upsert_label("m3", "L4", "News", source="user")
    store.upsert_embedding("m3", vec)

    registry = _make_registry([("L1", "Tech"), ("L2", "Travel"), ("L4", "News")])
    # L2 renamed Travel -> News; L4 (a different id, same old name) deleted.
    registry._client.list_user_labels.return_value = [("L1", "Tech"), ("L2", "News")]
    index = _make_index(["Travel", "Travel", "News"], ["m1", "m2", "m3"])
    client = MagicMock()
    skip_ids = set()
    log = MagicMock()

    reconcile_labels(registry, store, index, client, skip_ids, log)

    # Only the genuinely deleted label's message was re-inboxed/skip-marked.
    client.move_to_inbox.assert_called_once_with("m3")
    assert skip_ids == {"m3"}

    # m1/m2 (renamed) survive intact under their new name, NOT erased/skipped.
    rows = {mid: (lid, name, source) for mid, lid, name, source in store.iter_labels()}
    assert rows["m1"] == ("L2", "News", "user")
    assert rows["m2"] == ("L2", "News", "user")
    assert rows["m3"] == (SKIP_LABEL, SKIP_LABEL, "auto")

    # In-memory index agrees: renamed entries keep their embedding under the
    # new name, the deleted one becomes skip.
    assert index.labels[index._id_to_idx["m1"]] == "News"
    assert index.labels[index._id_to_idx["m2"]] == "News"
    assert index.labels[index._id_to_idx["m3"]] == SKIP_LABEL

    store.close()
