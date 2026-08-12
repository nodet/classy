"""Tests for live label reconciliation (deletion + rename detection)."""
import numpy as np
from unittest.mock import MagicMock, patch

from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.label_reconciler import reconcile_labels
from gmail_classifier.label_registry import LabelDiff, LabelRegistry
from gmail_classifier.training_index import TrainingIndex


def _make_registry(labels, excluded=None):
    client = MagicMock()
    client.list_user_labels.return_value = labels
    return LabelRegistry(client, excluded=excluded or set())


def _make_store(label_rows=None, embeddings=None):
    """Create a mock StateStore with configurable behavior."""
    store = MagicMock()
    label_rows = label_rows or {}
    embeddings = embeddings or {}
    store.message_ids_by_label.side_effect = lambda name: label_rows.get(name, set())
    store.get_embedding.side_effect = lambda mid: embeddings.get(mid)
    store.rename_label.return_value = 0
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

    store.rename_label.assert_not_called()
    store.remove_labels_by_name.assert_not_called()
    client.move_to_inbox.assert_not_called()
    log.assert_not_called()


def test_deleted_label_purges_and_reinboxes():
    registry = _make_registry([("L1", "Tech"), ("L2", "Travel")])
    vec = np.random.randn(384).astype(np.float32)
    store = _make_store(
        label_rows={"Travel": {"m3", "m4"}},
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

    # Store: label rows removed, then upserted as skip
    store.remove_labels_by_name.assert_called_once_with({"Travel"})
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
    store = _make_store()
    store.rename_label.return_value = 3
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

    store.rename_label.assert_called_once_with("Travel", "Voyages")
    client.move_to_inbox.assert_not_called()
    store.remove_labels_by_name.assert_not_called()

    # In-memory index updated
    assert "Travel" not in index.labels
    assert index.labels.count("Voyages") == 2


def test_move_to_inbox_failure_continues():
    registry = _make_registry([("L1", "Tech"), ("L2", "Travel")])
    vec = np.random.randn(384).astype(np.float32)
    store = _make_store(
        label_rows={"Travel": {"m3", "m4"}},
        embeddings={"m3": vec, "m4": vec},
    )
    index = _make_index(["Tech", "Travel", "Travel"], ["m1", "m3", "m4"])
    client = MagicMock()
    client.move_to_inbox.side_effect = [Exception("gone"), None]
    skip_ids = set()
    log = MagicMock()

    registry._client.list_user_labels.return_value = [("L1", "Tech")]

    reconcile_labels(registry, store, index, client, skip_ids, log)

    # Both messages still cleaned up in store despite one API failure
    store.remove_labels_by_name.assert_called_once_with({"Travel"})
    assert store.upsert_label.call_count == 2
    assert skip_ids == {"m3", "m4"}


def test_deleted_label_with_no_stored_messages():
    registry = _make_registry([("L1", "Tech"), ("L2", "Travel")])
    store = _make_store(label_rows={"Travel": set()})
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
        label_rows={"News": {"m5"}},
        embeddings={"m5": vec},
    )
    store.rename_label.return_value = 2
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
    store.rename_label.assert_called_once_with("Travel", "Voyages")
    assert index.labels.count("Voyages") == 2

    # Deletion happened
    client.move_to_inbox.assert_called_once_with("m5")
    store.remove_labels_by_name.assert_called_once_with({"News"})
    assert "m5" in skip_ids
