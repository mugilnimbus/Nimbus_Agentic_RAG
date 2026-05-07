import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from nimbus.models import Chunk
from nimbus.retrieval import exact_match_bonus, token_overlap_score, tokens_for_scoring
from nimbus.text_processing import (
    english_only_text,
    normalize_document_text,
    normalize_text,
    semantic_chunks,
    stable_hash,
)


class SourceBase:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
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
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    content_hash TEXT,
                    text TEXT NOT NULL DEFAULT ''
                )
                """
            )
            self.ensure_column(connection, "documents", "content_hash", "TEXT")
            self.ensure_column(connection, "documents", "text", "TEXT NOT NULL DEFAULT ''")
            self.migrate_documents_to_source_schema(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """
            )
            self.migrate_chunks_to_text_only(connection)
            self.backfill_document_text_from_chunks(connection)
            connection.execute("DELETE FROM chunks WHERE document_id NOT IN (SELECT id FROM documents)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
            connection.execute("DROP INDEX IF EXISTS idx_documents_kind")
            connection.execute("DROP INDEX IF EXISTS idx_documents_source")
            connection.execute("DROP INDEX IF EXISTS idx_jobs_status")
            connection.execute("DROP TABLE IF EXISTS jobs")

    def add_document(
        self,
        name: str,
        text: str,
        max_words: int = 420,
        kind: str = "source",
        english_only: bool = True,
        progress_callback=None,
    ) -> int:
        if kind != "source":
            raise ValueError("Source Base stores source documents only. Knowledge Base entries belong in Qdrant.")

        source_text = normalize_document_text(text)
        clean_text = normalize_document_text(english_only_text(text)) if english_only else source_text
        if not clean_text and source_text:
            clean_text = source_text
        if not clean_text:
            raise ValueError("Document text is empty after English filtering.")

        content_hash = stable_hash(f"source\n{name}\n{clean_text}")
        if progress_callback:
            progress_callback(0, 1, "Saving full extracted text to Source Documents")

        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO documents (name, created_at, chunk_count, content_hash, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name.strip() or "Untitled document", time.time(), 0, content_hash, clean_text),
            )
            document_id = int(cursor.lastrowid)

        if progress_callback:
            progress_callback(1, 1, "Stored full Source Document text in SQLite")
        return document_id

    def update_document_chunk_count(self, document_id: int, chunk_count: int) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE documents SET chunk_count = ? WHERE id = ?",
                (max(0, int(chunk_count)), int(document_id)),
            )

    def documents(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, name, created_at, chunk_count, content_hash
                FROM documents
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()
        return [self.document_row_to_dict(row) for row in rows]

    def document(self, document_id: int) -> dict:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, name, created_at, chunk_count, content_hash
                FROM documents
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()
        if row is None:
            raise ValueError("Document not found.")
        return self.document_row_to_dict(row)

    def document_text(self, document_id: int) -> str:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT text FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
        if row is None:
            raise ValueError("Document not found.")
        return str(row["text"] or "")

    def delete_document(self, document_id: int) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def chunks_for_document(self, document_id: int) -> list[Chunk]:
        document = self.document(document_id)
        text = self.document_text(document_id)
        if text:
            return [
                Chunk(
                    id=f"document-{document_id}",
                    document_id=int(document["id"]),
                    document_name=str(document["name"]),
                    document_kind="source-document",
                    chunk_index=0,
                    text=text,
                )
            ]

        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT chunks.id, chunks.document_id, documents.name AS document_name,
                       chunks.chunk_index, chunks.text
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE chunks.document_id = ?
                ORDER BY chunks.chunk_index ASC
                """,
                (document_id,),
            ).fetchall()
        return [
            Chunk(
                id=row["id"],
                document_id=row["document_id"],
                document_name=row["document_name"],
                document_kind="source",
                chunk_index=row["chunk_index"],
                text=row["text"],
            )
            for row in rows
        ]

    def document_ids_oldest_first(self) -> list[int]:
        with self.connect() as connection:
            rows = connection.execute("SELECT id FROM documents ORDER BY created_at ASC, id ASC").fetchall()
        return [int(row["id"]) for row in rows]

    def search_source_chunks(self, query: str, top_k: int = 6) -> list[Chunk]:
        query_tokens = set(tokens_for_scoring(query))
        if not query_tokens:
            return []

        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, name AS document_name, text
                FROM documents
                """
            ).fetchall()

        hits = []
        for row in rows:
            for index, chunk in enumerate(semantic_chunks(str(row["text"] or ""), max_words=650)):
                searchable_text = f"{row['document_name']} raw {chunk['title']} {chunk['text']}"
                score = token_overlap_score(query_tokens, searchable_text) + exact_match_bonus(query, searchable_text)
                if score >= 0.18:
                    hits.append(
                        Chunk(
                            id=f"{row['id']}-{index}",
                            document_id=row["id"],
                            document_name=row["document_name"],
                            document_kind="source",
                            chunk_index=index,
                            text=chunk["text"],
                            score=score,
                        )
                    )
        hits.sort(key=lambda chunk: chunk.score, reverse=True)
        return hits[: max(1, min(top_k, 20))]

    @staticmethod
    def document_row_to_dict(row: sqlite3.Row) -> dict:
        item = dict(row)
        item["kind"] = "source"
        return item

    @staticmethod
    def ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @staticmethod
    def migrate_chunks_to_text_only(connection: sqlite3.Connection) -> None:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(chunks)")}
        if not ({"vector", "vector_model", "vector_dim"} & columns):
            return
        connection.execute("ALTER TABLE chunks RENAME TO chunks_with_vectors_backup")
        connection.execute(
            """
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            INSERT INTO chunks (id, document_id, chunk_index, text)
            SELECT id, document_id, chunk_index, text
            FROM chunks_with_vectors_backup
            ORDER BY id ASC
            """
        )
        connection.execute("DROP TABLE chunks_with_vectors_backup")

    @staticmethod
    def migrate_documents_to_source_schema(connection: sqlite3.Connection) -> None:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(documents)")}
        legacy_columns = {"kind", "source_document_id", "embedding_model", "prompt_version"} & columns
        if not legacy_columns:
            return
        connection.execute("DROP INDEX IF EXISTS idx_documents_kind")
        connection.execute("DROP INDEX IF EXISTS idx_documents_source")
        for column in ("kind", "source_document_id", "embedding_model", "prompt_version"):
            if column in legacy_columns:
                connection.execute(f"ALTER TABLE documents DROP COLUMN {column}")

    @staticmethod
    def backfill_document_text_from_chunks(connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT documents.id
            FROM documents
            WHERE COALESCE(documents.text, '') = ''
            """
        ).fetchall()
        for row in rows:
            chunks = connection.execute(
                """
                SELECT text FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC, id ASC
                """,
                (row["id"],),
            ).fetchall()
            text = "\n\n".join(str(chunk["text"] or "") for chunk in chunks if str(chunk["text"] or "").strip())
            if text:
                connection.execute("UPDATE documents SET text = ? WHERE id = ?", (text, row["id"]))
