"""Tests for the progressive bootstrap driver (progressive.py).

Guards the plan's "Unit -- progressive interleave": batches obey the
message/time budget, the index grows via add_many, the bootstrap is resumable
and idempotent, and the store is finalized exactly once when the work-list
drains. The "notification serviced between batches" case is proven at the
pubsub-loop wiring layer (test_pubsub_mode_startup) where run_iteration and
run_batch actually interleave; here we prove the driver primitives.
"""
import numpy as np

from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.maturity import MATURITY_EXAMPLES_PER_LABEL
from gmail_classifier.progressive import ProgressiveBootstrap
from gmail_classifier.state_store import STATE_SCHEMA_VERSION, StateStore
from gmail_classifier.training_index import TrainingIndex


class _FakeEmbedder:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    dimension = 8

    def embed(self, text):
        return np.full(self.dimension, 1.0, dtype=np.float32)


class _FakeClient:
    def __init__(self, labels=None, inbox=None, watch_id="1000"):
        self._labels = labels or {}
        self._inbox = inbox or []
        self._watch_id = watch_id
        self.get_calls = []

    def watch(self, topic):
        return (self._watch_id, 10**18)

    def list_user_labels(self):
        return [(lid, name) for lid, (name, _ids) in self._labels.items()]

    def list_message_ids(self, label_id, max_results=0):
        ids = self._inbox if label_id == "INBOX" else self._labels.get(label_id, ("", []))[1]
        return list(ids[:max_results]) if max_results else list(ids)

    def list_unlabeled_inbox_ids(self, max_results=0):
        labeled = set()
        for _name, ids in self._labels.values():
            labeled.update(ids)
        unlabeled = [m for m in self._inbox if m not in labeled]
        return list(unlabeled[:max_results]) if max_results else list(unlabeled)

    def get_message(self, mid):
        self.get_calls.append(mid)
        return {"id": mid, "payload": {"headers": [], "body": {}}, "labelIds": []}


def _empty_index():
    return TrainingIndex(np.empty((0, 0), dtype=np.float32), [], [])


def _driver(tmp_path, client, **over):
    store = StateStore(str(tmp_path / "state.db"))
    kwargs = dict(
        client=client, embedder=_FakeEmbedder(), store=store,
        index=_empty_index(), excluded=set(), max_per_label=200,
        gmail_account_id="me@x.com",
    )
    kwargs.update(over)
    return ProgressiveBootstrap(**kwargs), store


def test_batch_respects_message_budget(tmp_path):
    client = _FakeClient(labels={"L_A": ("A", [f"a{i}" for i in range(100)])}, inbox=[])
    driver, store = _driver(tmp_path, client, batch_max_messages=10)

    built = driver.run_batch()
    assert built == 10                       # stopped at the message budget
    assert len(driver.index) == 10           # index grew by exactly the batch
    assert not driver.done
    store.close()


def test_batch_respects_time_budget(tmp_path):
    client = _FakeClient(labels={"L_A": ("A", [f"a{i}" for i in range(100)])}, inbox=[])
    # A clock that jumps past the deadline after the first embedded message.
    # First tick sets the deadline (5.0); the next check reads 100.0 and trips.
    ticks = iter([0.0, 100.0, 100.0, 100.0, 100.0])
    driver, store = _driver(tmp_path, client, batch_max_messages=50,
                            batch_max_seconds=5.0, now=lambda: next(ticks))
    built = driver.run_batch()
    assert built == 1  # one message, then the time budget tripped
    assert not driver.done
    store.close()


def test_runs_to_completion_over_many_batches(tmp_path):
    client = _FakeClient(
        labels={"L_A": ("A", ["a1", "a2", "a3"])},
        inbox=["i1", "i2"],
    )
    driver, store = _driver(tmp_path, client, batch_max_messages=2)

    total = 0
    guard = 0
    while not driver.done:
        total += driver.run_batch()
        guard += 1
        assert guard < 100
    assert total == 5  # 3 labeled + 2 skip
    by_id = {mid: label for mid, _v, label in store.iter_index()}
    assert by_id == {"a1": "A", "a2": "A", "a3": "A", "i1": SKIP_LABEL, "i2": SKIP_LABEL}
    # In-memory index matches the store join.
    assert len(driver.index) == 5
    store.close()


def test_finalizes_store_exactly_once(tmp_path):
    client = _FakeClient(labels={"L_A": ("A", ["a1"])}, inbox=[])
    driver, store = _driver(tmp_path, client, batch_max_messages=50)

    driver.run_batch()
    assert driver.done
    assert store.get_bootstrap_status() == "complete"
    assert store.get_meta("gmail_account_id") == "me@x.com"
    assert store.get_meta("state_schema_version") == STATE_SCHEMA_VERSION

    # Extra run_batch after completion is a no-op (idempotent finalize).
    assert driver.run_batch() == 0
    store.close()


def test_resumable_skips_already_embedded(tmp_path):
    client = _FakeClient(labels={"L_A": ("A", ["a1", "a2"])}, inbox=[])
    store = StateStore(str(tmp_path / "state.db"))
    # Pre-embed a1 as if a prior attempt (or a live label) already built it.
    store.upsert_label("a1", "L_A", "A", source="user")
    store.upsert_embedding("a1", np.full(8, 1.0, dtype=np.float32))

    driver = ProgressiveBootstrap(
        client=client, embedder=_FakeEmbedder(), store=store,
        index=_empty_index(), excluded=set(), max_per_label=200,
        gmail_account_id="me@x.com", batch_max_messages=50,
    )
    driver.run_batch()
    while not driver.done:
        driver.run_batch()
    # a1 was NOT re-fetched; only a2 hit the network.
    assert client.get_calls == ["a2"]
    store.close()


def test_grows_index_via_add_many_single_batch(tmp_path, monkeypatch):
    """A batch appends with ONE add_many, not per-message add()."""
    client = _FakeClient(labels={"L_A": ("A", ["a1", "a2", "a3"])}, inbox=[])
    driver, store = _driver(tmp_path, client, batch_max_messages=50)

    calls = {"add_many": 0, "add": 0}
    orig_many = driver.index.add_many
    monkeypatch.setattr(driver.index, "add_many",
                        lambda items: (calls.__setitem__("add_many", calls["add_many"] + 1), orig_many(items))[1])
    monkeypatch.setattr(driver.index, "add",
                        lambda *a, **k: calls.__setitem__("add", calls["add"] + 1))

    driver.run_batch()
    assert calls["add_many"] == 1
    assert calls["add"] == 0
    store.close()


def test_maturity_progresses_as_index_grows(tmp_path):
    # 6 examples of A (>= MIN_EXAMPLES_PER_LABEL=5) and a big skip pool, so the
    # gate has a real label target and a skip target.
    client = _FakeClient(
        labels={"L_A": ("A", [f"a{i}" for i in range(6)])},
        inbox=[f"i{i}" for i in range(60)],
    )
    driver, store = _driver(tmp_path, client, batch_max_messages=5)

    # Before the first batch the gate is unplanned -> conservatively immature.
    assert driver.is_mature() is False

    guard = 0
    while not driver.done:
        driver.run_batch()
        guard += 1
        assert guard < 200
    # Corpus fully built: A has 6 (target min(20,6)=6), skip has 60 (target
    # min(50,60)=50) -> gate open.
    assert driver.gate.label_targets == {"A": 6}
    assert driver.gate.skip_target == 50
    assert driver.is_mature() is True
    store.close()
