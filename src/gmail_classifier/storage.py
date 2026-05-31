import json
import sqlite3
from typing import List

from gmail_classifier.models import Message


class MessageStore:
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                subject TEXT,
                from_name TEXT,
                from_address TEXT,
                body_html TEXT,
                labels TEXT,
                list_id TEXT,
                date TEXT
            )
        """)
        self._conn.commit()

    def save_message(self, msg: Message):
        labels_json = json.dumps(msg.labels)
        self._conn.execute(
            """INSERT OR REPLACE INTO messages
               (id, subject, from_name, from_address, body_html, labels, list_id, date)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (msg.id, msg.subject, msg.from_name, msg.from_address,
             msg.body_html, labels_json, msg.list_id, msg.date),
        )
        self._conn.commit()

    def load_all(self) -> List[Message]:
        cursor = self._conn.execute(
            "SELECT id, subject, from_name, from_address, body_html, labels, list_id, date FROM messages"
        )
        return [self._row_to_message(row) for row in cursor.fetchall()]

    def load_by_label(self, label: str) -> List[Message]:
        # Since labels are stored as JSON array, use LIKE for filtering
        # This is adequate for the expected scale (thousands, not millions)
        cursor = self._conn.execute(
            "SELECT id, subject, from_name, from_address, body_html, labels, list_id, date FROM messages WHERE labels LIKE ?",
            (f'%"{label}"%',),
        )
        return [self._row_to_message(row) for row in cursor.fetchall()]

    def has_message(self, message_id: str) -> bool:
        cursor = self._conn.execute(
            "SELECT 1 FROM messages WHERE id = ?", (message_id,)
        )
        return cursor.fetchone() is not None

    def close(self):
        self._conn.close()

    def _row_to_message(self, row) -> Message:
        return Message(
            id=row[0],
            subject=row[1],
            from_name=row[2],
            from_address=row[3],
            body_html=row[4],
            labels=json.loads(row[5]),
            list_id=row[6],
            date=row[7],
        )
