import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class ChatStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self):
        with self._lock:
            connection = sqlite3.connect(self.db_path, timeout=120)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 120000")
            connection.execute("PRAGMA foreign_keys = ON")
            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    focus_entities TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_id ON chat_messages(chat_id)")

    def list_chats(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT chats.id, chats.title, chats.created_at, chats.updated_at,
                       COUNT(chat_messages.id) AS message_count,
                       MAX(chat_messages.content) FILTER (
                           WHERE chat_messages.id = (
                               SELECT MAX(id)
                               FROM chat_messages latest
                               WHERE latest.chat_id = chats.id
                           )
                       ) AS last_message
                FROM chats
                LEFT JOIN chat_messages ON chat_messages.chat_id = chats.id
                GROUP BY chats.id
                ORDER BY chats.updated_at DESC, chats.id DESC
                """
            ).fetchall()
        return [self.chat_payload(row) for row in rows]

    def create_chat(self, title: str = "New chat") -> dict:
        now = time.time()
        clean_title = title.strip() or "New chat"
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO chats (title, created_at, updated_at) VALUES (?, ?, ?)",
                (clean_title, now, now),
            )
            chat_id = int(cursor.lastrowid)
        return self.get_chat(chat_id)

    def rename_chat(self, chat_id: int, title: str) -> dict:
        clean_title = " ".join(str(title or "").split())[:80].strip()
        if not clean_title:
            raise ValueError("Chat title is required.")
        now = time.time()
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                (clean_title, now, int(chat_id)),
            )
        if cursor.rowcount == 0:
            raise ValueError("Chat not found.")
        return self.get_chat(chat_id)

    def delete_chat(self, chat_id: int) -> None:
        with self.connect() as connection:
            cursor = connection.execute("DELETE FROM chats WHERE id = ?", (int(chat_id),))
        if cursor.rowcount == 0:
            raise ValueError("Chat not found.")

    def get_or_create_chat(self, chat_id: int | None) -> dict:
        if chat_id:
            return self.get_chat(chat_id)
        chats = self.list_chats()
        return chats[0] if chats else self.create_chat()

    def get_chat(self, chat_id: int) -> dict:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id, title, created_at, updated_at FROM chats WHERE id = ?",
                (int(chat_id),),
            ).fetchone()
        if row is None:
            raise ValueError("Chat not found.")
        payload = dict(row)
        payload["message_count"] = len(self.messages(chat_id))
        payload["last_message"] = ""
        return payload

    def messages(self, chat_id: int) -> list[dict]:
        self.get_chat_exists(chat_id)
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, chat_id, role, content, sources_json, focus_entities, created_at
                FROM chat_messages
                WHERE chat_id = ?
                ORDER BY id ASC
                """,
                (int(chat_id),),
            ).fetchall()
        return [self.message_payload(row) for row in rows]

    def turns(self, chat_id: int) -> list[dict[str, str]]:
        turns = []
        pending_user = None
        for message in self.messages(chat_id):
            if message["role"] == "user":
                pending_user = message
                continue
            if message["role"] == "assistant" and pending_user:
                turns.append(
                    {
                        "question": pending_user["content"],
                        "answer": message["content"],
                        "focus_entities": message.get("focus_entities", ""),
                    }
                )
                pending_user = None
        return turns

    def append_message(
        self,
        chat_id: int,
        role: str,
        content: str,
        sources: list[dict] | None = None,
        focus_entities: list[str] | None = None,
    ) -> dict:
        if role not in {"user", "assistant"}:
            raise ValueError("Chat message role must be user or assistant.")
        now = time.time()
        focus_text = ", ".join(str(entity) for entity in (focus_entities or []) if entity)
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO chat_messages (
                    chat_id, role, content, sources_json, focus_entities, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(chat_id),
                    role,
                    str(content or ""),
                    json.dumps(sources or [], ensure_ascii=True),
                    focus_text,
                    now,
                ),
            )
            connection.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, int(chat_id)))
            row = connection.execute(
                """
                SELECT id, chat_id, role, content, sources_json, focus_entities, created_at
                FROM chat_messages
                WHERE id = ?
                """,
                (int(cursor.lastrowid),),
            ).fetchone()
        if role == "user":
            self.rename_if_empty(chat_id, content)
        return self.message_payload(row)

    def rename_if_empty(self, chat_id: int, question: str) -> None:
        chat = self.get_chat(chat_id)
        if chat["title"] != "New chat":
            return
        title = " ".join(str(question or "").split())[:54].strip() or "New chat"
        with self.connect() as connection:
            connection.execute("UPDATE chats SET title = ? WHERE id = ?", (title, int(chat_id)))

    def get_chat_exists(self, chat_id: int) -> None:
        with self.connect() as connection:
            exists = connection.execute("SELECT 1 FROM chats WHERE id = ?", (int(chat_id),)).fetchone()
        if exists is None:
            raise ValueError("Chat not found.")

    @staticmethod
    def chat_payload(row: sqlite3.Row) -> dict:
        item = dict(row)
        item["last_message"] = " ".join(str(item.get("last_message") or "").split())[:120]
        item["message_count"] = int(item.get("message_count") or 0)
        return item

    @staticmethod
    def message_payload(row: sqlite3.Row) -> dict:
        item = dict(row)
        try:
            item["sources"] = json.loads(item.pop("sources_json") or "[]")
        except json.JSONDecodeError:
            item["sources"] = []
        return item
