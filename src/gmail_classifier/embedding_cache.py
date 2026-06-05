"""SQLite-backed cache for precomputed embeddings."""

import sqlite3
from typing import Dict, List, Optional

import numpy as np


class EmbeddingCache:
    """Stores message_id → embedding vector in SQLite."""

    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                message_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL
            )
        """)
        self._conn.commit()

    def get(self, message_id: str) -> Optional[np.ndarray]:
        row = self._conn.execute(
            "SELECT vector FROM embeddings WHERE message_id = ?",
            (message_id,),
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32).copy()

    def get_batch(self, message_ids: List[str]) -> Dict[str, np.ndarray]:
        if not message_ids:
            return {}
        placeholders = ",".join("?" * len(message_ids))
        rows = self._conn.execute(
            f"SELECT message_id, vector FROM embeddings WHERE message_id IN ({placeholders})",
            message_ids,
        ).fetchall()
        return {
            row[0]: np.frombuffer(row[1], dtype=np.float32).copy()
            for row in rows
        }

    def put(self, message_id: str, vector: np.ndarray):
        blob = vector.astype(np.float32).tobytes()
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (message_id, vector) VALUES (?, ?)",
            (message_id, blob),
        )
        self._conn.commit()

    def put_batch(self, message_ids: List[str], vectors: np.ndarray):
        data = [
            (mid, vectors[i].astype(np.float32).tobytes())
            for i, mid in enumerate(message_ids)
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (message_id, vector) VALUES (?, ?)",
            data,
        )
        self._conn.commit()

    def close(self):
        self._conn.close()
