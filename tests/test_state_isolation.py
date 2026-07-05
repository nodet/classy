"""Backend-isolation guards (coexistence invariant #1): the state backend opens
ONLY its own files and never touches the legacy three-DB set, regardless of
whether those legacy files are present.
"""
import sqlite3

import numpy as np

from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.models import Message
from gmail_classifier.state_store import StateBackend, StateStore

LEGACY_NAMES = ("training.db", "inbox_sample.db", "embeddings.db")


def _vec(seed, dim=8):
    return np.full(dim, float(seed), dtype=np.float32)


def test_state_backend_never_opens_legacy_files(tmp_path, monkeypatch):
    """Spy on sqlite3.connect and assert no legacy DB path is ever opened during
    a full state warm load + live-adaptation cycle, even with legacy files
    sitting right beside state.db."""
    data = tmp_path
    # Create the legacy files so their mere presence can't change behavior.
    for name in LEGACY_NAMES:
        (data / name).write_bytes(b"")

    opened = []
    real_connect = sqlite3.connect

    def spy_connect(path, *a, **kw):
        opened.append(str(path))
        return real_connect(path, *a, **kw)

    monkeypatch.setattr(sqlite3, "connect", spy_connect)

    state_db = str(data / "state.db")
    backend = StateBackend(state_db, excluded=set())
    msg = Message(id="m1", subject="s", from_address="a@x.com", labels=["Tech"])
    backend.upsert_label(msg, "Label_1", _vec(1))
    backend.load_index(embedder=None)
    backend.set_last_processed_history_id("123")
    backend.close()

    # Every opened path is the state db (or its sidecars); none is a legacy file.
    for path in opened:
        assert not any(path.endswith(name) for name in LEGACY_NAMES), (
            f"state backend opened a legacy file: {path}"
        )
    assert any(p.endswith("state.db") for p in opened)


def test_presence_independence(tmp_path):
    """A state boot with legacy files present produces the same index as a boot
    with none present -- the legacy files are ignored entirely (no migration)."""
    def build_index(dirpath, with_legacy):
        dirpath.mkdir(parents=True, exist_ok=True)
        if with_legacy:
            for name in LEGACY_NAMES:
                (dirpath / name).write_bytes(b"junk-that-must-be-ignored")
        state_db = str(dirpath / "state.db")
        store = StateStore(state_db)
        store.upsert_embedding("m1", _vec(1))
        store.upsert_label("m1", "Label_1", "Tech")
        store.upsert_embedding("s1", _vec(2))
        store.upsert_label("s1", SKIP_LABEL, SKIP_LABEL, source="auto")
        store.close()

        backend = StateBackend(state_db, excluded=set())
        loaded = backend.load_index(embedder=None)
        backend.close()
        return loaded

    with_legacy = build_index(tmp_path / "a", with_legacy=True)
    without_legacy = build_index(tmp_path / "b", with_legacy=False)

    assert with_legacy.skip_ids == without_legacy.skip_ids == {"m1", "s1"}
    assert with_legacy.stats.n_train == without_legacy.stats.n_train == 1
    assert with_legacy.stats.n_skip == without_legacy.stats.n_skip == 1
