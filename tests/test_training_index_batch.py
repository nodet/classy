"""Tests that add_many keeps peak memory of a bulk relabel proportional to the
result, instead of growing with the batch size.

The one-by-one path reallocates the whole embedding matrix on every message, so
processing a large batch churns far more memory than the final index occupies.
The behavioural tests below pin down that add_many is equivalent to sequential
add(); the tracemalloc test measures that its peak allocation stays close to the
final matrix size while the one-by-one path peaks materially higher.
"""
import tracemalloc
from unittest.mock import MagicMock

import numpy as np

from gmail_classifier.label_change_handler import process_label_changes
from gmail_classifier.models import HistoryEvent
from gmail_classifier.storage import MessageStore
from gmail_classifier.training_index import TrainingIndex

FLOAT_BYTES = 4
DIM = 384


def _make_batch(n, dim=DIM, label="Transports", rng=None):
    """A list of (id, embedding, label) tuples, deterministic given rng."""
    rng = rng or np.random.default_rng(0)
    return [
        (f"new{i}", rng.standard_normal(dim).astype(np.float32), label)
        for i in range(n)
    ]


def _seed_index(n_start, dim=384, rng=None):
    rng = rng or np.random.default_rng(1)
    emb = rng.standard_normal((n_start, dim)).astype(np.float32)
    labels = ["Politique"] * n_start
    ids = [f"seed{i}" for i in range(n_start)]
    return TrainingIndex(emb, labels, ids)


def test_add_many_equivalent_to_sequential_adds():
    """add_many must produce the exact same index as N sequential add() calls."""
    N, S = 600, 1000
    batch = _make_batch(N)  # same tuples fed to both paths

    seq = _seed_index(S)
    for mid, emb, lab in batch:
        seq.add(mid, emb, lab)

    bulk = _seed_index(S)  # identical seed rng -> identical starting matrix
    bulk.add_many(batch)

    assert bulk.labels == seq.labels
    assert bulk._ids == seq._ids
    assert bulk._id_to_idx == seq._id_to_idx
    assert np.array_equal(bulk.embeddings, seq.embeddings)


def test_add_many_updates_existing_ids_in_place():
    """Ids already in the index are replaced (not appended), same as add()."""
    idx = _seed_index(10)
    new_vec = np.random.default_rng(9).standard_normal(384).astype(np.float32)

    idx.add_many([("seed3", new_vec, "Banque")])

    assert len(idx) == 10  # no growth
    pos = idx._id_to_idx["seed3"]
    assert idx.labels[pos] == "Banque"
    assert np.array_equal(idx.embeddings[pos], new_vec)


def test_add_many_last_write_wins_within_batch():
    """A repeated id within one batch keeps the last value, like sequential add()."""
    idx = _seed_index(5)
    v1 = np.ones(384, dtype=np.float32)
    v2 = np.full(384, 2.0, dtype=np.float32)

    idx.add_many([("dup", v1, "Pub"), ("dup", v2, "LBC")])

    assert len(idx) == 6  # added once, not twice
    pos = idx._id_to_idx["dup"]
    assert idx.labels[pos] == "LBC"
    assert np.array_equal(idx.embeddings[pos], v2)


# --- Peak memory of a bulk relabel, through the public handler ---


def _raw(msg_id, label_id):
    return {
        "id": msg_id,
        "labelIds": [label_id],
        "payload": {
            "headers": [
                {"name": "From", "value": "sender@example.com"},
                {"name": "Subject", "value": f"subject {msg_id}"},
            ],
            "body": {"data": ""},
            "parts": [],
        },
    }


def _relabel_peak_bytes(tmp_path, n_start, n_batch):
    """Peak bytes allocated while a batch of n_batch labelsAdded events is applied
    to an index of n_start rows, via the real process_label_changes path.

    Returns (peak_added_bytes, final_matrix_bytes). Uses tracemalloc, which since
    numpy 1.22 sees numpy's own allocations, so the returned peak includes the
    embedding matrix (re)allocations that dominate this path.
    """
    events = [
        HistoryEvent(type="labelsAdded", message_id=f"m{i}", label_ids=["Label_T"])
        for i in range(n_batch)
    ]
    client = MagicMock()
    client.get_message.side_effect = lambda mid: _raw(mid, "Label_T")

    training_store = MessageStore(str(tmp_path / "training.db"))
    skip_store = MessageStore(str(tmp_path / "skip.db"))

    rng = np.random.default_rng(1)
    index = TrainingIndex(
        rng.standard_normal((n_start, DIM)).astype(np.float32),
        ["Politique"] * n_start,
        [f"seed{i}" for i in range(n_start)],
    )
    embedder = MagicMock()
    embedder.embed.return_value = np.ones(DIM, dtype=np.float32)

    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    process_label_changes(
        events=events,
        client=client,
        training_store=training_store,
        skip_store=skip_store,
        label_id_to_name={"Label_T": "Transports"},
        user_label_ids={"Label_T"},
        excluded_labels=set(),
        index=index,
        embedder=embedder,
    )
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    training_store.close()
    skip_store.close()

    assert len(index) == n_start + n_batch  # the whole batch really was applied
    final_matrix = (n_start + n_batch) * DIM * FLOAT_BYTES
    return peak - base, final_matrix


def test_bulk_relabel_peak_memory_stays_near_final_index_size(tmp_path):
    """A bulk relabel must not churn memory proportional to the batch size.

    Measured through the real handler, the batched path peaks at <2x the final
    matrix size (it holds the old matrix plus one new copy). The one-by-one path
    it replaces peaks at ~2.6-2.8x because every message reallocates the whole
    matrix; the 2.5x bound below sits cleanly between the two bands and fails on
    the unbatched implementation.
    """
    peak, final_matrix = _relabel_peak_bytes(tmp_path, n_start=1000, n_batch=600)
    assert peak < 2.5 * final_matrix
