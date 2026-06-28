"""Tests for LabelRegistry: dynamic label map with lazy refresh."""
from unittest.mock import MagicMock

from gmail_classifier.label_registry import LabelRegistry


def _make_client(labels):
    """Create a mock GmailClient returning given [(id, name), ...] labels."""
    client = MagicMock()
    client.list_user_labels.return_value = labels
    return client


def test_initial_state():
    """Registry populates maps from client on construction."""
    client = _make_client([("L1", "Tech"), ("L2", "Travel")])
    registry = LabelRegistry(client, excluded=set())

    assert registry.get_name("L1") == "Tech"
    assert registry.get_id("Travel") == "L2"
    assert registry.user_label_ids == {"L1", "L2"}


def test_is_known():
    client = _make_client([("L1", "Tech")])
    registry = LabelRegistry(client, excluded=set())

    assert registry.is_known("L1")
    assert not registry.is_known("L99")


def test_is_excluded():
    client = _make_client([("L1", "Tech"), ("L2", "XLC")])
    registry = LabelRegistry(client, excluded={"XLC"})

    assert not registry.is_excluded("L1")
    assert registry.is_excluded("L2")
    assert not registry.is_excluded("L99")  # unknown is not excluded


def test_refresh_picks_up_new_labels():
    """After refresh, newly created labels become available."""
    client = _make_client([("L1", "Tech")])
    registry = LabelRegistry(client, excluded=set())

    assert not registry.is_known("L2")

    # Simulate label creation
    client.list_user_labels.return_value = [("L1", "Tech"), ("L2", "NewLabel")]
    registry.refresh()

    assert registry.is_known("L2")
    assert registry.get_name("L2") == "NewLabel"
    assert registry.get_id("NewLabel") == "L2"
    assert "L2" in registry.user_label_ids


def test_refresh_picks_up_deleted_labels():
    """After refresh, deleted labels are no longer available."""
    client = _make_client([("L1", "Tech"), ("L2", "Travel")])
    registry = LabelRegistry(client, excluded=set())

    assert registry.is_known("L2")

    # Simulate label deletion
    client.list_user_labels.return_value = [("L1", "Tech")]
    registry.refresh()

    assert not registry.is_known("L2")
    assert registry.get_name("L2") is None
    assert "L2" not in registry.user_label_ids


def test_ensure_known_refreshes_on_unknown_id():
    """ensure_known triggers a refresh if the label ID is not known."""
    client = _make_client([("L1", "Tech")])
    registry = LabelRegistry(client, excluded=set())

    # Initial call was in __init__
    assert client.list_user_labels.call_count == 1

    # Simulate label creation
    client.list_user_labels.return_value = [("L1", "Tech"), ("L2", "NewLabel")]

    result = registry.ensure_known("L2")
    assert result is True
    assert registry.get_name("L2") == "NewLabel"
    assert client.list_user_labels.call_count == 2


def test_ensure_known_does_not_refresh_for_known_id():
    """ensure_known does NOT call the API if the label is already known."""
    client = _make_client([("L1", "Tech")])
    registry = LabelRegistry(client, excluded=set())

    result = registry.ensure_known("L1")
    assert result is True
    # No extra API call
    assert client.list_user_labels.call_count == 1


def test_ensure_known_returns_false_if_still_unknown():
    """ensure_known returns False if label not found even after refresh."""
    client = _make_client([("L1", "Tech")])
    registry = LabelRegistry(client, excluded=set())

    # API still returns only L1 after refresh
    result = registry.ensure_known("L99")
    assert result is False
    assert client.list_user_labels.call_count == 2


def test_max_label_width():
    """max_label_width is the length of the longest non-excluded label."""
    client = _make_client([("L1", "Tech"), ("L2", "Technologie"), ("L3", "XLC")])
    registry = LabelRegistry(client, excluded={"XLC"})

    assert registry.max_label_width == len("Technologie")


def test_max_label_width_zero_when_all_excluded():
    """The default=0 guard: an empty non-excluded set must not crash."""
    client = _make_client([("L1", "XLC")])
    registry = LabelRegistry(client, excluded={"XLC"})

    assert registry.max_label_width == 0


def test_max_label_width_updates_on_refresh():
    """max_label_width updates when a longer label is discovered."""
    client = _make_client([("L1", "Tech")])
    registry = LabelRegistry(client, excluded=set())

    assert registry.max_label_width == 4

    client.list_user_labels.return_value = [("L1", "Tech"), ("L2", "Conferences")]
    registry.refresh()

    assert registry.max_label_width == len("Conferences")
