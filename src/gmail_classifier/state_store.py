"""State storage backend: the single-file ``state.db`` layout (derived-only).

This is the second ``StorageBackend`` implementation (see ``storage_backend.py``).
Where the legacy backend keeps three DBs and persists message bodies, the state
backend keeps **one** SQLite file that stores only *derived* runtime state --
message ids, label ids/names, and embedding vectors. No bodies, subjects, or
senders are ever written here. The runtime ``TrainingIndex`` is the join
``embeddings ⋈ labels on message_id``.

It opens **only** ``state.db`` / ``state.rebuild.db`` (and their SQLite
sidecars). It never touches ``training.db`` / ``inbox_sample.db`` /
``embeddings.db`` -- there is deliberately no migration path between the
backends (coexistence invariant #1).

Phase 3 wires the **warm path** only: a populated ``state.db`` (seeded for
tests, or bootstrapped by a later phase) loads its index from the join and
resumes from its durable history cursor. Cold bootstrap and Gmail-backed
rebuild/reconcile land in Phase 4; the progressive/maturity logic in Phase 5.

Unlike the legacy adapter, the history cursor here is **durable** (persisted in
``meta``), so a restart replays Gmail history from where it left off rather than
re-``watch()``ing fresh. Every durable cursor write also stamps
``last_processed_at`` in the *same* transaction, so a later phase can measure
the downtime gap accurately.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from enum import Enum
from typing import Callable, Iterator, List, Optional, Set, Tuple

import numpy as np

from gmail_classifier.classifier import (
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    MIN_EXAMPLES_PER_LABEL,
    SKIP_LABEL,
)
from gmail_classifier.embeddings import Embedder
from gmail_classifier.models import Message
from gmail_classifier.storage_backend import LoadedIndex
from gmail_classifier.training import AssemblyStats
from gmail_classifier.training_index import TrainingIndex

# Bumped whenever the state.db schema changes shape. Checked independently of
# the ML fingerprint (a schema bump means the file layout itself is stale).
STATE_SCHEMA_VERSION = "1"

# Manual constant bumped whenever build_text_representation / preprocess_email_body
# changes in a way that alters embeddings (see the plan's fingerprint section).
TEXTREPR_VERSION = "textrepr-v1"


# --------------------------------------------------------------------------
# Fingerprints (vector validity vs. membership -- kept orthogonal on purpose)
# --------------------------------------------------------------------------

def _classifier_params_hash() -> str:
    """Stable hash of the classifier tunables that affect a cached prediction's
    validity. Bundled into the ML fingerprint so a threshold change is noticed."""
    parts = [
        f"high={HIGH_CONFIDENCE_THRESHOLD}",
        f"medium={MEDIUM_CONFIDENCE_THRESHOLD}",
        f"min_examples={MIN_EXAMPLES_PER_LABEL}",
        f"skip={SKIP_LABEL}",
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def compute_ml_fingerprint(model_name: str, dimension: int) -> str:
    """Fingerprint governing **vector validity**: does a cached ``id->vector``
    still reflect the current model + text representation + classifier params?

    A mismatch means the cached vectors are stale and the store must be rebuilt
    from Gmail (Phase 4). Deliberately excludes the excluded-label set -- that
    governs *membership*, not validity, and is tracked separately.
    """
    return "|".join([
        f"model={model_name}",
        f"dim={dimension}",
        TEXTREPR_VERSION,
        f"clf={_classifier_params_hash()}",
    ])


def compute_excluded_hash(excluded_names) -> str:
    """Fingerprint governing **membership**: which labels participate in the
    index. Changing it triggers the cheap reconcile path (Phase 4), never a
    full re-embed, because it invalidates no vector.

    Phase 3 hashes the normalized configured *names* only. A later phase folds
    in their resolution to Gmail label ids (per the plan); since ``state.db`` is
    derived and rebuildable, evolving this format just forces a one-time
    reconcile/rebuild, not data loss.
    """
    normalized = sorted(n.strip() for n in excluded_names if n and n.strip())
    return hashlib.sha256("\n".join(normalized).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Startup dispatch (pure decision -- testable without a mailbox)
# --------------------------------------------------------------------------

class StartupDecision(Enum):
    """What the ``state`` startup path should do, given the store's meta.

    Only ``WARM`` is actionable in Phase 3; the others name the Gmail-backed
    paths that land in Phase 4 (cold bootstrap, ML rebuild, exclusion reconcile)
    or a hard reset. The dispatcher fails closed on anything but ``WARM`` until
    those paths exist, so stale state is never loaded.
    """
    WARM = "warm"              # populated + all fingerprints match -> load join
    BOOTSTRAP = "bootstrap"    # empty/fresh store -> cold bootstrap (Phase 4)
    REBUILD = "rebuild"        # ML fingerprint mismatch -> vectors stale (Phase 4)
    RECONCILE = "reconcile"    # excluded-label membership changed (Phase 4)
    INCOMPATIBLE = "incompatible"  # schema/account mismatch -> reset required


def decide_startup(
    store: "StateStore",
    *,
    schema_version: str,
    ml_fingerprint: str,
    excluded_hash: str,
    gmail_account_id: Optional[str],
) -> StartupDecision:
    """Decide the startup path purely from persisted meta vs. current config.

    Order matters: schema/account incompatibility is a hard stop (the file is
    unusable as-is); then an empty store bootstraps; then vector-validity (ML)
    is checked before membership (excluded set), because a stale vector must be
    rebuilt regardless of membership.
    """
    if store.get_bootstrap_status() in (None, "in_progress"):
        # Absent = fresh VM; in_progress = crashed mid-bootstrap (resume is
        # Phase 5, so Phase 3 treats it as "needs the bootstrap path").
        return StartupDecision.BOOTSTRAP

    stored_schema = store.get_meta("state_schema_version")
    if stored_schema is not None and stored_schema != schema_version:
        return StartupDecision.INCOMPATIBLE

    # A warm store must be *bound* to this mailbox: require a non-empty stored
    # account id that equals the live one. A missing/empty stored value is not a
    # free pass -- an old, manually seeded, or partially populated state.db that
    # never recorded its account is not actually tied to this mailbox, so its
    # label ids and history cursor cannot be trusted. Treat missing, empty, or
    # mismatched as INCOMPATIBLE (a hard reset, same as a schema mismatch).
    stored_account = store.get_meta("gmail_account_id")
    if not stored_account or stored_account != gmail_account_id:
        return StartupDecision.INCOMPATIBLE

    if store.get_meta("ml_fingerprint") != ml_fingerprint:
        return StartupDecision.REBUILD

    if store.get_meta("excluded_labels_hash") != excluded_hash:
        return StartupDecision.RECONCILE

    return StartupDecision.WARM


# --------------------------------------------------------------------------
# Low-level store: the single state.db file
# --------------------------------------------------------------------------

def _vec_to_blob(vector: np.ndarray) -> bytes:
    # Same on-disk format as EmbeddingCache (float32 raw bytes), so the two
    # backends' vectors are byte-compatible even though the files are separate.
    return vector.astype(np.float32).tobytes()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


class StateStore:
    """SQLite wrapper over the single derived-only ``state.db``.

    Tables: ``embeddings`` (the vector cache), ``labels`` (id -> label_id, with
    ``__skip__`` for the skip pool), ``pending_new`` (post-boundary mail parked
    before maturity -- populated in Phase 5), and ``meta`` (schema/fingerprint/
    cursor key-value). One connection per instance.
    """

    def __init__(self, db_path: str, now_ms: Optional[Callable[[], int]] = None):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        # Injected clock keeps last_processed_at deterministic in tests.
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))
        self._create_tables()

    def now_ms(self) -> int:
        """The store's clock (injected in tests). Bootstrap stamps its own meta
        timestamps from this so they stay consistent with the cursor's."""
        return self._now_ms()

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                message_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS labels (
                message_id TEXT PRIMARY KEY,
                label_id TEXT NOT NULL,
                label_name TEXT,
                source TEXT
            );
            CREATE TABLE IF NOT EXISTS pending_new (
                message_id TEXT PRIMARY KEY,
                first_seen_history_id TEXT,
                reason TEXT
            );
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        self._conn.commit()

    # --- embeddings ------------------------------------------------------

    def upsert_embedding(self, message_id: str, vector: np.ndarray) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (message_id, vector) VALUES (?, ?)",
            (message_id, _vec_to_blob(vector)),
        )
        self._conn.commit()

    def has_embedding(self, message_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM embeddings WHERE message_id = ?", (message_id,)
        ).fetchone()
        return row is not None

    def embedded_ids(self) -> Set[str]:
        """Every id with a cached vector -- what bootstrap resumability checks."""
        return {r[0] for r in self._conn.execute("SELECT message_id FROM embeddings")}

    # --- labels ----------------------------------------------------------

    def upsert_label(
        self, message_id: str, label_id: str,
        label_name: Optional[str] = None, source: str = "user",
    ) -> None:
        """Record ``message_id`` under ``label_id`` (a Gmail label id, or
        ``__skip__`` for the skip pool). Last-write-wins on the id: the single
        row-per-id structurally prevents the duplicate-vote / orphan-row bug the
        two-store legacy layout had."""
        self._conn.execute(
            """INSERT OR REPLACE INTO labels (message_id, label_id, label_name, source)
               VALUES (?, ?, ?, ?)""",
            (message_id, label_id, label_name, source),
        )
        self._conn.commit()

    def remove_label(self, message_id: str) -> None:
        self._conn.execute("DELETE FROM labels WHERE message_id = ?", (message_id,))
        self._conn.commit()

    def known_ids(self) -> Set[str]:
        """Every id present in ``labels`` -- real labels **and** ``__skip__``.
        Inbox/history processing skips these so already-seen mail isn't
        re-classified."""
        return {r[0] for r in self._conn.execute("SELECT message_id FROM labels")}

    def iter_labels(self) -> Iterator[Tuple[str, str, Optional[str], Optional[str]]]:
        """Yield ``(message_id, label_id, label_name, source)`` for every label
        row. Used by the ML rebuild to re-embed each id under its *existing*
        label -- a fingerprint mismatch invalidates vectors, not membership."""
        return iter(self._conn.execute(
            "SELECT message_id, label_id, label_name, source FROM labels"
        ).fetchall())

    def label_names(self) -> Set[str]:
        """Distinct real (non-skip) label-name snapshots present in the store.
        Drives the exclusion reconcile: which configured labels are already
        represented vs. newly included."""
        return {
            r[0] for r in self._conn.execute(
                "SELECT DISTINCT label_name FROM labels WHERE label_id != ? "
                "AND label_name IS NOT NULL", (SKIP_LABEL,)
            )
        }

    def remove_labels_by_name(self, names: Set[str]) -> int:
        """Delete every label row whose ``label_name`` is in ``names`` (a now-
        excluded set). Embeddings are left cached; the join simply stops
        including them. Returns the number of rows removed."""
        if not names:
            return 0
        placeholders = ",".join("?" for _ in names)
        cur = self._conn.execute(
            f"DELETE FROM labels WHERE label_name IN ({placeholders})",
            tuple(names),
        )
        self._conn.commit()
        return cur.rowcount

    def skip_vote_ids(self) -> Set[str]:
        """Only ids whose ``label_id`` is ``__skip__`` -- the negative examples
        that vote in the KNN."""
        return {
            r[0] for r in self._conn.execute(
                "SELECT message_id FROM labels WHERE label_id = ?", (SKIP_LABEL,)
            )
        }

    def iter_index(self) -> Iterator[Tuple[str, np.ndarray, str]]:
        """Yield ``(message_id, vector, index_label)`` for every id present in
        **both** ``embeddings`` and ``labels`` -- the join that is the runtime
        index. An embedded id with no label (or a label with no embedding) is
        excluded, so orphaned rows can never reach the classifier.

        ``index_label`` is ``SKIP_LABEL`` for skip rows, else the label name
        snapshot (what the classifier and registry compare on)."""
        rows = self._conn.execute(
            """SELECT e.message_id, e.vector, l.label_id, l.label_name
               FROM embeddings e JOIN labels l ON e.message_id = l.message_id"""
        )
        for message_id, blob, label_id, label_name in rows:
            index_label = SKIP_LABEL if label_id == SKIP_LABEL else (label_name or label_id)
            yield message_id, _blob_to_vec(blob), index_label

    # --- pending_new (drained in Phase 5; helpers land now) --------------

    def add_pending(self, message_id: str, history_id: str, reason: str = "immature") -> None:
        # OR IGNORE: parking the same id twice is a no-op (idempotent).
        self._conn.execute(
            """INSERT OR IGNORE INTO pending_new
               (message_id, first_seen_history_id, reason) VALUES (?, ?, ?)""",
            (message_id, history_id, reason),
        )
        self._conn.commit()

    def get_pending(self) -> List[Tuple[str, str, str]]:
        return [
            (r[0], r[1], r[2]) for r in self._conn.execute(
                "SELECT message_id, first_seen_history_id, reason FROM pending_new"
            )
        ]

    def remove_pending(self, message_id: str) -> None:
        self._conn.execute(
            "DELETE FROM pending_new WHERE message_id = ?", (message_id,)
        )
        self._conn.commit()

    # --- meta ------------------------------------------------------------

    def get_meta(self, key: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else row[0]

    def set_meta(self, key: str, value: Optional[str]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def get_bootstrap_status(self) -> Optional[str]:
        return self.get_meta("bootstrap_status")

    def pin_bootstrap_boundary(self, boundary_history_id: str) -> None:
        """Pin the read-only boundary, durable cursor, start timestamp, and
        in-progress status in a **single transaction**.

        Bootstrap calls ``watch()`` once, at start-of-service, to fix the
        historyId boundary; everything at-or-before it is pre-existing (read
        only), only mail after it is classifiable. These four writes must land
        atomically: if the boundary or cursor persisted but ``bootstrap_status``
        did not, a crash in that window would make the next boot think no
        boundary was pinned and call ``watch()`` again -- moving the boundary
        forward and silently skipping mail that arrived in between."""
        now = str(self._now_ms())
        with self._conn:  # one transaction across all four keys
            self._conn.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                [
                    ("bootstrap_boundary_history_id", boundary_history_id),
                    ("last_processed_history_id", boundary_history_id),
                    ("last_processed_at", now),
                    ("bootstrap_started_at", now),
                    ("bootstrap_status", "in_progress"),
                ],
            )

    # --- durable history cursor -----------------------------------------

    def get_last_processed_history_id(self) -> Optional[str]:
        return self.get_meta("last_processed_history_id")

    def set_last_processed_history_id(self, history_id: str) -> None:
        """Advance the durable cursor. Stamps ``last_processed_at`` in the
        **same** transaction so the downtime gap a later phase measures stays
        consistent with the cursor it belongs to."""
        with self._conn:  # one transaction, atomic across both keys
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("last_processed_history_id", history_id),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("last_processed_at", str(self._now_ms())),
            )

    def get_last_processed_at(self) -> Optional[int]:
        raw = self.get_meta("last_processed_at")
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None  # malformed -> caller fails safe to read-only resync

    # --- lifecycle -------------------------------------------------------

    def close(self) -> None:
        self._conn.close()


# --------------------------------------------------------------------------
# The seam adapter
# --------------------------------------------------------------------------

class StateBackend:
    """``StorageBackend`` over the single ``state.db`` (derived-only).

    Live-adaptation writes store **id + label + vector only** and discard the
    body -- the opposite of the legacy adapter's ``save_message``. The history
    cursor is **durable** (persisted in ``state.db``), so a fresh adapter over
    the same file resumes from the last processed history id rather than
    re-watching fresh.
    """

    def __init__(self, state_db: str, excluded: Set[str],
                 now_ms: Optional[Callable[[], int]] = None):
        self._excluded = excluded
        self._store = StateStore(state_db, now_ms=now_ms)

    @property
    def store(self) -> StateStore:
        return self._store

    # --- index load ------------------------------------------------------

    def load_index(self, embedder: Embedder) -> LoadedIndex:
        """Build the runtime index from the ``embeddings ⋈ labels`` join.

        ``embedder`` is unused on the warm path (vectors are cached; re-embedding
        is the Phase-4 rebuild path), but is accepted to satisfy the seam.
        ``skip_ids`` is the full ``known_ids`` set -- every id already in the
        store, so the live loop won't re-classify seen mail.
        """
        ids: List[str] = []
        vectors: List[np.ndarray] = []
        labels: List[str] = []
        n_skip = 0
        for message_id, vec, index_label in self._store.iter_index():
            ids.append(message_id)
            vectors.append(vec)
            labels.append(index_label)
            if index_label == SKIP_LABEL:
                n_skip += 1

        if ids:
            embeddings = np.vstack([v.reshape(1, -1) for v in vectors])
        else:
            embeddings = np.empty((0, 0), dtype=np.float32)
        index = TrainingIndex(embeddings, labels, ids)

        stats = AssemblyStats(
            n_train=len(ids) - n_skip,
            n_skip=n_skip,
            # Labeled-wins-over-skip is enforced at the source (one row per id),
            # so nothing is dropped at load time.
            n_dropped=0,
        )
        return LoadedIndex(index=index, skip_ids=self._store.known_ids(), stats=stats)

    # --- live-adaptation writes (no body persisted) ----------------------

    def upsert_label(
        self, message: Message, label_id: str, vec: Optional[np.ndarray] = None
    ) -> None:
        """Record a labeled example: store the vector + ``id -> label_id`` and
        drop any pending-new row. The body is discarded (only the caller's
        ``message.labels[0]`` is kept, as a display snapshot)."""
        if vec is not None:
            self._store.upsert_embedding(message.id, vec)
        label_name = message.labels[0] if message.labels else label_id
        self._store.upsert_label(message.id, label_id, label_name, source="user")
        self._store.remove_pending(message.id)

    def upsert_skip(self, message: Message, vec: Optional[np.ndarray] = None) -> None:
        """Record a ``__skip__`` example (id + vector, no body)."""
        if vec is not None:
            self._store.upsert_embedding(message.id, vec)
        self._store.upsert_label(message.id, SKIP_LABEL, SKIP_LABEL, source="auto")
        self._store.remove_pending(message.id)

    def remove(self, message_id: str) -> None:
        """Drop a message's label row (its embedding stays cached and is simply
        excluded from the join until a new label row is written)."""
        self._store.remove_label(message_id)

    # --- durable history cursor -----------------------------------------

    def get_last_processed_history_id(self) -> Optional[str]:
        return self._store.get_last_processed_history_id()

    def set_last_processed_history_id(self, history_id: str) -> None:
        self._store.set_last_processed_history_id(history_id)

    # --- lifecycle -------------------------------------------------------

    def close(self) -> None:
        self._store.close()
