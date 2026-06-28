"""Tests for process_inbox (the poll-mode classify step)."""
from unittest.mock import MagicMock, patch

import numpy as np

from gmail_classifier.classifier import Action, ClassificationResult
from gmail_classifier.inbox_check import process_inbox


def _make_raw_message(msg_id, subject="Test", label_ids=None):
    """A raw Gmail API message dict (mirrors test_history_processor)."""
    return {
        "id": msg_id,
        "labelIds": label_ids if label_ids is not None else ["INBOX"],
        "payload": {
            "headers": [
                {"name": "From", "value": "sender@example.com"},
                {"name": "Subject", "value": subject},
            ],
            "body": {"data": ""},
            "parts": [],
        },
    }


def _make_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = np.zeros(384)
    return embedder


def _make_index():
    index = MagicMock()
    index.embeddings = np.zeros((10, 384))
    index.labels = ["Tech"] * 10
    return index


def _make_registry(name_to_id=None, user_label_ids=None):
    registry = MagicMock()
    name_to_id = name_to_id if name_to_id is not None else {"Tech": "Label_1"}
    registry.get_id.side_effect = lambda name: name_to_id.get(name)
    registry.user_label_ids = user_label_ids if user_label_ids is not None else {"Label_1"}
    registry.max_label_width = 10
    return registry


def _result(action, label="Tech", confidence=1.0):
    return ClassificationResult(label=label, confidence=confidence,
                                action=action, neighbors=[])


def _run(client, *, result, dry_run=False, skip_ids=None, self_labeled=None,
         registry=None, inbox_ids=None):
    skip_store = MagicMock()
    with patch("gmail_classifier.inbox_check.classify", return_value=result):
        results = process_inbox(
            client=client,
            embedder=_make_embedder(),
            index=_make_index(),
            registry=registry or _make_registry(),
            skip_ids=skip_ids if skip_ids is not None else set(),
            skip_store=skip_store,
            k=5,
            max_messages=50,
            dry_run=dry_run,
            self_labeled=self_labeled,
            inbox_ids=inbox_ids,
        )
    return results, skip_store


# 1. New unlabeled message -> classified, label applied, ids recorded.
def test_new_message_labeled_and_recorded():
    client = MagicMock()
    client.list_message_ids.return_value = ["m1"]
    client.get_message.return_value = _make_raw_message("m1")

    skip_ids = set()
    self_labeled = set()
    results, _ = _run(client, result=_result(Action.LABEL),
                      skip_ids=skip_ids, self_labeled=self_labeled)

    client.apply_label.assert_called_once_with("m1", "Label_1", archive=True)
    assert results[0]["applied"] is True
    assert "m1" in skip_ids
    assert "m1" in self_labeled


# 2. Message already carrying a user label -> skipped entirely.
def test_already_user_labeled_message_skipped():
    client = MagicMock()
    client.list_message_ids.return_value = ["m1"]
    client.get_message.return_value = _make_raw_message(
        "m1", label_ids=["INBOX", "Label_1"])

    results, skip_store = _run(client, result=_result(Action.LABEL))

    assert results == []
    client.apply_label.assert_not_called()
    skip_store.save_message.assert_not_called()


# 3. Low-confidence / NO_LABEL -> saved as skip example, not labeled.
def test_no_label_message_saved_to_skip_store():
    client = MagicMock()
    client.list_message_ids.return_value = ["m1"]
    client.get_message.return_value = _make_raw_message("m1")

    skip_ids = set()
    results, skip_store = _run(
        client, result=_result(Action.NO_LABEL, label="Tech", confidence=0.1),
        skip_ids=skip_ids)

    client.apply_label.assert_not_called()
    skip_store.save_message.assert_called_once()
    assert results[0]["applied"] is False
    assert "m1" in skip_ids


# 4. dry_run -> no label applied, nothing saved, but result reported.
def test_dry_run_makes_no_changes():
    client = MagicMock()
    client.list_message_ids.return_value = ["m1", "m2"]
    client.get_message.side_effect = lambda mid: _make_raw_message(mid)

    skip_ids = set()
    self_labeled = set()
    with patch("gmail_classifier.inbox_check.classify") as cls:
        cls.side_effect = [_result(Action.LABEL), _result(Action.NO_LABEL)]
        skip_store = MagicMock()
        results = process_inbox(
            client=client, embedder=_make_embedder(), index=_make_index(),
            registry=_make_registry(), skip_ids=skip_ids, skip_store=skip_store,
            dry_run=True, self_labeled=self_labeled,
        )

    client.apply_label.assert_not_called()
    skip_store.save_message.assert_not_called()
    assert self_labeled == set()
    assert results[0]["applied"] is False
    # dry-run still records ids so they aren't re-reported within the session
    assert skip_ids == {"m1", "m2"}


# 5. Predicted label missing in Gmail -> warning, no apply, NOT recorded.
def test_missing_label_warns_and_does_not_record():
    client = MagicMock()
    client.list_message_ids.return_value = ["m1"]
    client.get_message.return_value = _make_raw_message("m1")

    # Registry knows no labels -> get_id returns None.
    registry = _make_registry(name_to_id={}, user_label_ids=set())
    skip_ids = set()
    results, skip_store = _run(
        client, result=_result(Action.LABEL, label="Ghost"),
        skip_ids=skip_ids, registry=registry)

    client.apply_label.assert_not_called()
    skip_store.save_message.assert_not_called()
    assert results[0]["warning"] is True
    # Not recorded: a later label refresh should be able to retry it.
    assert "m1" not in skip_ids


# 6. Ids already in skip_ids are filtered before any fetch.
def test_already_seen_ids_not_fetched():
    client = MagicMock()
    client.list_message_ids.return_value = ["m1", "m2"]
    client.get_message.side_effect = lambda mid: _make_raw_message(mid)

    results, _ = _run(client, result=_result(Action.NO_LABEL),
                      skip_ids={"m1"})

    fetched = [c.args[0] for c in client.get_message.call_args_list]
    assert fetched == ["m2"]  # m1 skipped, never fetched


# 7. Supplied inbox_ids are used instead of calling the API.
def test_supplied_inbox_ids_avoid_relisting():
    client = MagicMock()
    client.get_message.return_value = _make_raw_message("m9")

    results, _ = _run(client, result=_result(Action.NO_LABEL),
                      inbox_ids=["m9"])

    client.list_message_ids.assert_not_called()
    assert results[0]["message_id"] == "m9"
