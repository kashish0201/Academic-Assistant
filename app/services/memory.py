import sqlite3
from datetime import datetime, timezone
from typing import Dict, List

from app.config import BASE_DIR

MEMORY_DB_PATH = BASE_DIR / "conversation_memory.db"


class ConversationMemory:
    def __init__(self, db_path=MEMORY_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_session
                ON conversation_turns (session_id, id)
                """
            )

    def get_history(self, session_id: str, limit: int | None = 10) -> List[Dict]:
        with self._connect() as conn:
            if limit is None:
                rows = conn.execute(
                    """
                    SELECT role, content, created_at
                    FROM conversation_turns
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (session_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT role, content, created_at
                    FROM conversation_turns
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()
                rows = list(reversed(rows))

        return [
            {"role": row["role"], "content": row["content"], "created_at": row["created_at"]}
            for row in rows
        ]

    def list_sessions(self, limit: int = 50) -> List[Dict]:
        with self._connect() as conn:
            sessions = conn.execute(
                """
                SELECT session_id, MAX(created_at) AS updated_at, COUNT(*) AS turn_count
                FROM conversation_turns
                GROUP BY session_id
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            result = []
            for session in sessions:
                first_user = conn.execute(
                    """
                    SELECT content
                    FROM conversation_turns
                    WHERE session_id = ? AND role = 'user'
                    ORDER BY id ASC
                    LIMIT 1
                    """,
                    (session["session_id"],),
                ).fetchone()

                title = "New conversation"
                if first_user and first_user["content"]:
                    title = first_user["content"].strip().replace("\n", " ")
                    if len(title) > 48:
                        title = f"{title[:48]}…"

                result.append(
                    {
                        "session_id": session["session_id"],
                        "title": title,
                        "updated_at": session["updated_at"],
                        "turn_count": session["turn_count"],
                    }
                )

            return result

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_turns (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, created_at),
            )

    def clear_session(self, session_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM conversation_turns WHERE session_id = ?",
                (session_id,),
            )
            return cursor.rowcount
