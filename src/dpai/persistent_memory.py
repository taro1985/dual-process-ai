#!/usr/bin/env python3
"""
SQLite-backed Persistent Episodic Memory for DPAI

Provides sub-millisecond, low-memory SQLite storage for multi-turn conversation
history and episodic events. Survives bot/service restarts without RAM overhead.
"""

import sqlite3
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class SQLiteEpisodicMemory:
    """
    Lightweight SQLite persistent storage for conversation turns and agent episodes.
    Uses WAL mode for high concurrency and near-zero RAM consumption (<2MB).
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else Path("data/dpai_memory.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at REAL NOT NULL
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_created
                ON conversation_turns (session_id, created_at);
            """)
            conn.commit()

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        max_turns_per_session: int = 10,
    ) -> None:
        """Append a conversation turn and prune old turns beyond max_turns."""
        now = time.time()
        meta_str = json.dumps(metadata or {}, ensure_ascii=False)
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO conversation_turns (session_id, role, content, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, meta_str, now),
            )
            # Prune old turns beyond max_turns_per_session
            conn.execute("""
                DELETE FROM conversation_turns
                WHERE session_id = ? AND id NOT IN (
                    SELECT id FROM conversation_turns
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
            """, (session_id, session_id, max_turns_per_session))
            conn.commit()

    def get_turns(self, session_id: str, limit: int = 6) -> List[Dict[str, Any]]:
        """Retrieve recent conversation turns for the given session."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT role, content, metadata_json, created_at
                FROM (
                    SELECT id, role, content, metadata_json, created_at
                    FROM conversation_turns
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
            """, (session_id, limit))
            rows = cursor.fetchall()

        results = []
        for role, content, meta_json, ts in rows:
            meta = {}
            if meta_json:
                try:
                    meta = json.loads(meta_json)
                except Exception:
                    pass
            results.append({
                "role": role,
                "text": content,
                "content": content,
                "metadata": meta,
                "created_at": ts,
            })
        return results

    def clear_session(self, session_id: str) -> None:
        """Clear history for a specific session."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM conversation_turns WHERE session_id = ?", (session_id,))
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(DISTINCT session_id), COUNT(*) FROM conversation_turns")
            sessions, total_turns = cursor.fetchone()
        return {
            "total_sessions": sessions or 0,
            "total_turns": total_turns or 0,
        }
