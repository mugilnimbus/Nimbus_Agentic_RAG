import json
import math
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid
from array import array
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from nimbus import prompts


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
ENGLISH_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "you", "are", "not",
    "can", "will", "your", "have", "has", "use", "using", "when", "page",
    "what", "where", "which", "into", "only", "must", "should", "first",
    "if", "then", "than", "same", "more", "information", "recommended", "is",
    "my", "me", "i", "a", "an", "of", "in", "on", "to", "there",
}
NON_ENGLISH_HINTS = {
    "und", "der", "die", "das", "von", "mit", "nicht", "bitte", "oder",
    "unter", "unterstützt", "speicher", "anschluss", "steckplatz", "pour",
    "avec", "depuis", "dans", "vous", "veuillez", "connexion", "vitesse",
    "mémoire", "emplacement", "processeur", "sortie", "entrée",
}
@dataclass
class Chunk:
    id: int | str
    document_id: int
    document_name: str
    document_kind: str
    chunk_index: int
    text: str
    score: float = 0.0


class VectorRAG:
    def __init__(
        self,
        db_path: str,
        base_url: str,
        model: str,
        image_model: str,
        embedding_model: str,
        api_key: str,
        vector_backend: str,
        qdrant_url: str,
        qdrant_collection: str,
    ) -> None:
        self.db_path = Path(db_path)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.image_model = image_model or model
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.vector_backend = vector_backend.strip().lower() or "qdrant"
        self.qdrant_url = qdrant_url.rstrip("/")
        self.qdrant_collection = qdrant_collection.strip() or "nimbus_chunks"
        self.notes_group_chunks = int(os.environ.get("AI_NOTES_GROUP_CHUNKS", "30"))
        self.notes_max_tokens = int(os.environ.get("AI_NOTES_MAX_TOKENS", "12000"))
        self.notes_concurrency = max(
            1, min(8, int(os.environ.get("AI_NOTES_CONCURRENCY", "1")))
        )
        self.prompt_version = os.environ.get("RAG_PROMPT_VERSION", "2026-04-27")
        self.rerank_enabled = os.environ.get("RAG_RERANK", "1") != "0"
        self._db_lock = threading.RLock()
        self._qdrant_ready_dims: set[int] = set()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        with self._db_lock:
            conn = sqlite3.connect(self.db_path, timeout=120)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 120000")
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    kind TEXT NOT NULL DEFAULT 'raw',
                    content_hash TEXT
                )
                """
            )
            self._ensure_column(conn, "documents", "kind", "TEXT NOT NULL DEFAULT 'raw'")
            self._ensure_column(conn, "documents", "content_hash", "TEXT")
            self._migrate_documents_to_raw_schema(conn)
            conn.execute(
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
            self._migrate_chunks_to_text_only(conn)
            conn.execute("DELETE FROM documents WHERE COALESCE(kind, 'raw') <> 'raw'")
            conn.execute(
                "DELETE FROM chunks WHERE document_id NOT IN (SELECT id FROM documents)"
            )
            conn.execute(
                """
                UPDATE documents
                SET chunk_count = (
                    SELECT COUNT(*) FROM chunks WHERE chunks.document_id = documents.id
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_kind ON documents(kind)"
            )
            conn.execute("DROP INDEX IF EXISTS idx_documents_source")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    progress_current INTEGER NOT NULL DEFAULT 0,
                    progress_total INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._ensure_column(conn, "jobs", "progress_current", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "jobs", "progress_total", "INTEGER NOT NULL DEFAULT 0")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")

    def _ensure_column(
        self, conn: sqlite3.Connection, table: str, column: str, definition: str
    ) -> None:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _migrate_chunks_to_text_only(self, conn: sqlite3.Connection) -> None:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
        vector_columns = {"vector", "vector_model", "vector_dim"} & columns
        if not vector_columns:
            return
        conn.execute("ALTER TABLE chunks RENAME TO chunks_with_vectors_backup")
        conn.execute(
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
        conn.execute(
            """
            INSERT INTO chunks (id, document_id, chunk_index, text)
            SELECT id, document_id, chunk_index, text
            FROM chunks_with_vectors_backup
            ORDER BY id ASC
            """
        )
        conn.execute("DROP TABLE chunks_with_vectors_backup")

    def _migrate_documents_to_raw_schema(self, conn: sqlite3.Connection) -> None:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(documents)")}
        legacy_columns = {"source_document_id", "embedding_model", "prompt_version"} & columns
        if not legacy_columns:
            return
        conn.execute("DROP INDEX IF EXISTS idx_documents_source")
        for column in ("source_document_id", "embedding_model", "prompt_version"):
            if column in legacy_columns:
                conn.execute(f"ALTER TABLE documents DROP COLUMN {column}")

    def qdrant_enabled(self) -> bool:
        return self.vector_backend == "qdrant"

    def qdrant_status(self) -> dict:
        if not self.qdrant_enabled():
            return {"enabled": False, "status": "disabled"}
        try:
            response = self._qdrant_request("GET", f"/collections/{self.qdrant_collection}", timeout=2)
            result = response.get("result") or {}
            return {
                "enabled": True,
                "status": "ready",
                "url": self.qdrant_url,
                "collection": self.qdrant_collection,
                "points_count": int(result.get("points_count") or 0),
                "vectors_count": int(result.get("vectors_count") or 0),
            }
        except Exception as exc:
            if "HTTP 404" in str(exc):
                return {
                    "enabled": True,
                    "status": "collection_missing",
                    "url": self.qdrant_url,
                    "collection": self.qdrant_collection,
                }
            return {
                "enabled": True,
                "status": "unavailable",
                "url": self.qdrant_url,
                "collection": self.qdrant_collection,
                "error": str(exc),
            }

    def _qdrant_request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        timeout: int = 120,
    ) -> dict:
        if not self.qdrant_enabled():
            raise RuntimeError("Qdrant vector backend is disabled.")
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.qdrant_url}{path}",
            data=data,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Qdrant HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Qdrant is not reachable at {self.qdrant_url}. Start Qdrant or set VECTOR_BACKEND=sqlite."
            ) from exc
        return json.loads(body) if body else {}

    def _ensure_qdrant_collection(self, vector_dim: int) -> None:
        if not self.qdrant_enabled():
            return
        if vector_dim in self._qdrant_ready_dims:
            return
        try:
            response = self._qdrant_request("GET", f"/collections/{self.qdrant_collection}", timeout=5)
            config = response.get("result", {}).get("config", {}).get("params", {})
            vectors = config.get("vectors", {})
            existing_size = vectors.get("size") if isinstance(vectors, dict) else None
            if existing_size and int(existing_size) != int(vector_dim):
                raise RuntimeError(
                    f"Qdrant collection '{self.qdrant_collection}' has vector size "
                    f"{existing_size}, but {self.embedding_model} produced {vector_dim}. "
                    "Use a new QDRANT_COLLECTION or recreate the collection."
                )
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise
            self._qdrant_request(
                "PUT",
                f"/collections/{self.qdrant_collection}",
                {
                    "vectors": {
                        "size": vector_dim,
                        "distance": "Cosine",
                    }
                },
            )
        self._qdrant_ready_dims.add(vector_dim)

    def _qdrant_filter(self, **values) -> dict:
        return {
            "must": [
                {"key": key, "match": {"value": value}}
                for key, value in values.items()
                if value is not None
            ]
        }

    def _record_point_id(self, document_id: int, group_index: int, record_index: int, information: str) -> str:
        digest = stable_hash(f"{document_id}:{group_index}:{record_index}:{information}")[:16]
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"nimbus:record:{document_id}:{group_index}:{record_index}:{digest}"))

    def clear_qdrant_collection(self) -> None:
        if not self.qdrant_enabled():
            return
        try:
            self._qdrant_request("DELETE", f"/collections/{self.qdrant_collection}?timeout=60")
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise
        self._qdrant_ready_dims.clear()

    def qdrant_records(self, limit: int = 200) -> list[dict]:
        if not self.qdrant_enabled():
            return []
        try:
            response = self._qdrant_request(
                "POST",
                f"/collections/{self.qdrant_collection}/points/scroll",
                {
                    "limit": max(1, min(int(limit), 1000)),
                    "with_payload": True,
                    "with_vector": False,
                    "filter": self._qdrant_filter(vector_model=self.embedding_model),
                },
            )
        except RuntimeError as exc:
            if "HTTP 404" in str(exc):
                return []
            raise
        records = []
        for item in (response.get("result") or {}).get("points", []):
            payload = item.get("payload") or {}
            records.append(
                {
                    "id": str(item.get("id") or ""),
                    "document_id": int(payload.get("document_id") or 0),
                    "document_name": str(payload.get("document_name") or "Unknown"),
                    "document_kind": str(payload.get("kind") or "record"),
                    "chunk_index": int(payload.get("chunk_index") or 0),
                    "source_chunk_start": int(payload.get("source_chunk_start") or 0),
                    "source_chunk_end": int(payload.get("source_chunk_end") or 0),
                    "source": str(payload.get("source") or ""),
                    "keywords": payload.get("keywords") or [],
                    "information": str(payload.get("information") or ""),
                    "text": str(payload.get("text") or ""),
                    "vector_model": str(payload.get("vector_model") or ""),
                    "prompt_version": str(payload.get("prompt_version") or ""),
                }
            )
        records.sort(key=lambda item: (item["document_name"], item["source_chunk_start"], item["id"]))
        return records

    def _upsert_qdrant_records(self, records: Sequence[dict]) -> int:
        if not self.qdrant_enabled() or not records:
            return 0
        points = []
        vector_dim = 0
        for record in records:
            keywords = record.get("keywords") or []
            if isinstance(keywords, str):
                keywords = [keywords]
            keywords = [normalize_text(str(keyword)) for keyword in keywords if normalize_text(str(keyword))]
            information = normalize_text(str(record.get("information") or ""))
            if not information:
                continue
            source = normalize_text(str(record.get("source") or ""))
            document_id = int(record["document_id"])
            group_index = int(record.get("group_index") or 0)
            record_index = int(record.get("record_index") or 0)
            source_chunk_start = int(record.get("source_chunk_start") or 0)
            source_chunk_end = int(record.get("source_chunk_end") or source_chunk_start)
            document_name = str(record.get("document_name") or "Unknown")
            vector_text = "Search keywords: " + ", ".join(keywords)
            vector = self.embed_text(vector_text if keywords else information)
            vector_dim = len(vector)
            points.append(
                {
                    "id": self._record_point_id(document_id, group_index, record_index, information),
                    "vector": list(vector),
                    "payload": {
                        "kind": "record",
                        "document_id": document_id,
                        "document_name": document_name,
                        "chunk_index": source_chunk_start,
                        "record_index": record_index,
                        "source_chunk_start": source_chunk_start,
                        "source_chunk_end": source_chunk_end,
                        "source": source,
                        "keywords": keywords,
                        "information": information,
                        "text": information,
                        "vector_model": self.embedding_model,
                        "prompt_version": self.prompt_version,
                    },
                }
            )
        if not points:
            return 0
        self._ensure_qdrant_collection(vector_dim)
        batch_size = 128
        for start in range(0, len(points), batch_size):
            self._qdrant_request(
                "PUT",
                f"/collections/{self.qdrant_collection}/points?wait=true",
                {"points": points[start:start + batch_size]},
            )
        return len(points)

    def _delete_qdrant_document(self, document_id: int) -> None:
        if not self.qdrant_enabled():
            return
        try:
            self._qdrant_request(
                "POST",
                f"/collections/{self.qdrant_collection}/points/delete?wait=true",
                {"filter": self._qdrant_filter(document_id=int(document_id))},
            )
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise

    def add_document(
        self,
        name: str,
        text: str,
        max_words: int = 420,
        kind: str = "raw",
        english_only: bool = True,
        progress_callback=None,
    ) -> int:
        if kind != "raw":
            raise ValueError("SQLite stores raw source documents only. Distilled records belong in Qdrant.")
        source_text = normalize_text(text)
        clean_text = normalize_text(english_only_text(text)) if english_only else source_text
        if not clean_text and source_text:
            # Technical PDFs can be math-heavy enough that the language filter drops
            # every block. Keep the extracted text rather than failing the ingest.
            clean_text = source_text
        if not clean_text:
            raise ValueError("Document text is empty after English filtering.")

        pieces = chunk_text(clean_text, max_words=max_words)
        content_hash = stable_hash(f"{kind}\n{name}\n{clean_text}")
        if progress_callback:
            progress_callback(len(pieces), len(pieces), f"Prepared {len(pieces)} raw chunks; saving to SQLite")

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO documents (
                    name, created_at, chunk_count, kind, content_hash
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    name.strip() or "Untitled document",
                    time.time(),
                    len(pieces),
                    kind,
                    content_hash,
                ),
            )
            document_id = int(cur.lastrowid)
            for idx, piece in enumerate(pieces):
                conn.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, text)
                    VALUES (?, ?, ?)
                    """,
                    (document_id, idx, piece),
                )
            if progress_callback:
                progress_callback(len(pieces), len(pieces), f"Saved {len(pieces)} raw chunks")

        if progress_callback:
            progress_callback(len(pieces), len(pieces), f"Indexed {len(pieces)} raw chunks in SQLite")
        return document_id

    def documents(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, name, created_at, chunk_count, kind, content_hash
                FROM documents
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_document(self, document_id: int) -> None:
        try:
            self._delete_qdrant_document(document_id)
        except Exception:
            pass
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def document(self, document_id: int) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, name, created_at, chunk_count, kind, content_hash
                FROM documents
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()
        if row is None:
            raise ValueError("Document not found.")
        return dict(row)

    def document_chunks(self, document_id: int) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT chunks.id, chunks.document_id, documents.name AS document_name,
                       documents.kind AS document_kind, chunks.chunk_index, chunks.text
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE chunks.document_id = ?
                ORDER BY chunks.chunk_index ASC
                """,
                (document_id,),
            ).fetchall()
        if not rows:
            self.document(document_id)
        return [
            Chunk(
                id=row["id"],
                document_id=row["document_id"],
                document_name=row["document_name"],
                document_kind=row["document_kind"],
                chunk_index=row["chunk_index"],
                text=row["text"],
            )
            for row in rows
        ]

    def distill_document(self, document_id: int, progress_callback=None) -> int:
        source = self.document(document_id)
        source_chunks = self.document_chunks(document_id)
        group_size = max(1, self.notes_group_chunks)
        groups = []

        for start in range(0, len(source_chunks), group_size):
            group = source_chunks[start : start + group_size]
            chunk_label = f"chunks {group[0].chunk_index + 1}-{group[-1].chunk_index + 1}"
            chunk_text = "\n\n".join(
                f"Chunk {chunk.chunk_index + 1}:\n{chunk.text}" for chunk in group
            )
            groups.append((start, chunk_label, chunk_text, group[0].chunk_index, group[-1].chunk_index))

        distilled_by_index: dict[int, list[dict]] = {}
        total_groups = len(groups)
        with ThreadPoolExecutor(max_workers=min(self.notes_concurrency, len(groups))) as pool:
            futures = {
                pool.submit(
                    self.distill_chunk_group,
                    source["name"],
                    chunk_label,
                    chunk_text,
                ): (index, chunk_label, chunk_text, source_chunk_start, source_chunk_end)
                for index, chunk_label, chunk_text, source_chunk_start, source_chunk_end in groups
            }
            completed = 0
            if progress_callback:
                progress_callback(0, total_groups, f"Distilling raw chunk group 0/{total_groups}")
            for future in as_completed(futures):
                index, chunk_label, chunk_text, source_chunk_start, source_chunk_end = futures[future]
                try:
                    records = future.result()
                except Exception as exc:
                    records = [
                        {
                            "keywords": ["distillation failed", source["name"], chunk_label],
                            "information": (
                                f"Distillation failed for {chunk_label}: {exc}. "
                                f"Extracted text needing review: {normalize_text(chunk_text)[:5000]}"
                            ),
                            "source": chunk_label,
                        }
                    ]
                if not records:
                    records = [
                        {
                            "keywords": ["extracted text needing review", source["name"], chunk_label],
                            "information": normalize_text(chunk_text)[:5000],
                            "source": chunk_label,
                        }
                    ]
                for record in records:
                    record["document_id"] = int(document_id)
                    record["document_name"] = source["name"]
                    record["group_index"] = int(index)
                    record["source_chunk_start"] = int(source_chunk_start)
                    record["source_chunk_end"] = int(source_chunk_end)
                    record.setdefault("source", chunk_label)
                distilled_by_index[index] = records
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_groups, f"Distilled raw chunk group {completed}/{total_groups}")

        qdrant_records = []
        for index in sorted(distilled_by_index):
            for record_index, record in enumerate(distilled_by_index[index], start=1):
                record["record_index"] = record_index
                qdrant_records.append(record)

        if not qdrant_records:
            raise ValueError("No distilled records were produced.")

        self._delete_qdrant_document(document_id)
        if progress_callback:
            progress_callback(0, len(qdrant_records), f"Embedding distilled records for {source['name']}")
        written = self._upsert_qdrant_records(qdrant_records)
        if progress_callback:
            progress_callback(written, len(qdrant_records), f"Stored {written} distilled records in Qdrant")
        return written

    def rebuild_records_for_all_raw_documents(self, progress_callback=None) -> list[int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM documents WHERE kind = 'raw' ORDER BY created_at ASC, id ASC"
            ).fetchall()
        rebuilt = []
        total = len(rows)
        self.clear_qdrant_collection()
        for index, row in enumerate(rows, start=1):
            if progress_callback:
                progress_callback(index - 1, total, f"Distilling raw document {index}/{total}")
            rebuilt.append(self.distill_document(int(row["id"])))
            if progress_callback:
                progress_callback(index, total, f"Distilled raw document {index}/{total}")
        return rebuilt

    def create_job(self, job_type: str, detail: str = "") -> int:
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO jobs (type, status, detail, created_at, updated_at)
                VALUES (?, 'queued', ?, ?, ?)
                """,
                (job_type, detail, now, now),
            )
            return int(cur.lastrowid)

    def update_job(
        self,
        job_id: int,
        status: str,
        detail: str | None = None,
        error: str | None = None,
        result: dict | None = None,
        progress_current: int | None = None,
        progress_total: int | None = None,
    ) -> None:
        fields = ["status = ?", "updated_at = ?"]
        values: list[object] = [status, time.time()]
        if detail is not None:
            fields.append("detail = ?")
            values.append(detail)
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        if result is not None:
            fields.append("result_json = ?")
            values.append(json.dumps(result, ensure_ascii=True))
        if progress_current is not None:
            fields.append("progress_current = ?")
            values.append(max(0, int(progress_current)))
        if progress_total is not None:
            fields.append("progress_total = ?")
            values.append(max(0, int(progress_total)))
        values.append(job_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values)

    def jobs(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, type, status, detail, error, result_json,
                       progress_current, progress_total, created_at, updated_at
                FROM jobs
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (max(1, min(limit, 100)),),
            ).fetchall()
        jobs = []
        for row in rows:
            item = dict(row)
            try:
                item["result"] = json.loads(item.pop("result_json") or "{}")
            except json.JSONDecodeError:
                item["result"] = {}
            jobs.append(item)
        return jobs

    def distill_chunk_group(self, document_name: str, chunk_label: str, text: str) -> list[dict]:
        messages = [
            {
                "role": "system",
                "content": prompts.DISTILL_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompts.DISTILL_USER_TEMPLATE.format(
                    document_name=document_name,
                    chunk_label=chunk_label,
                    text=text,
                ),
            },
        ]
        notes = self.chat(messages, max_tokens=self.notes_max_tokens).strip()
        if notes.upper() == "NO_USEFUL_FACTS":
            return []
        return parse_note_records(notes, default_source=chunk_label)

    def extract_image_notes(self, document_name: str, image_data: str, mime_type: str) -> str:
        data_url = f"data:{mime_type};base64,{image_data}"
        messages = [
            {
                "role": "system",
                "content": prompts.IMAGE_NOTES_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompts.IMAGE_NOTES_USER_TEMPLATE.format(
                            document_name=document_name
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ]
        notes = self.chat(
            messages,
            max_tokens=self.notes_max_tokens,
            model=self.image_model,
        ).strip()
        if not notes:
            raise RuntimeError("Vision model returned empty image notes.")
        return notes

    def rewrite_query(self, question: str) -> str:
        messages = [
            {
                "role": "user",
                "content": prompts.QUERY_REWRITE_TEMPLATE.format(question=question),
            },
        ]
        rewritten = normalize_text(self.chat(messages, max_tokens=2000))
        if not rewritten:
            return question
        return rewritten[:1200]

    def search(self, query: str, top_k: int = 6, kind: str | None = None) -> list[Chunk]:
        if self.qdrant_enabled():
            return self.search_qdrant(query, top_k=top_k, kind=kind)
        return self.search_sqlite(query, top_k=top_k, kind=kind)

    def search_qdrant(self, query: str, top_k: int = 6, kind: str | None = None) -> list[Chunk]:
        query_vector = self.embed_text(query)
        self._ensure_qdrant_collection(len(query_vector))
        query_tokens = set(tokens_for_scoring(query))
        limit = max(top_k * 4, 24)
        search_filter = self._qdrant_filter(kind=kind, vector_model=self.embedding_model)
        response = self._qdrant_request(
            "POST",
            f"/collections/{self.qdrant_collection}/points/search",
            {
                "vector": list(query_vector),
                "limit": limit,
                "with_payload": True,
                "filter": search_filter,
            },
        )
        candidates: list[Chunk] = []
        for item in response.get("result", []):
            payload = item.get("payload") or {}
            searchable_text = (
                f"{payload.get('document_name', '')} {payload.get('kind', '')} "
                f"{payload.get('text', '')}"
            )
            vector_score = float(item.get("score") or 0.0)
            lexical_score = token_overlap_score(query_tokens, searchable_text)
            exact_bonus = exact_match_bonus(query, searchable_text)
            score = (0.72 * vector_score) + (0.22 * lexical_score) + exact_bonus
            if payload.get("kind") in {"record", "notes"}:
                score *= 1.08
            if score > 0:
                candidates.append(
                    Chunk(
                        id=str(item.get("id") or payload.get("chunk_id") or ""),
                        document_id=int(payload.get("document_id") or 0),
                        document_name=str(payload.get("document_name") or "Unknown"),
                        document_kind=str(payload.get("kind") or "raw"),
                        chunk_index=int(payload.get("chunk_index") or 0),
                        text=str(payload.get("text") or ""),
                        score=score,
                    )
                )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: max(1, min(top_k, 20))]

    def search_sqlite(self, query: str, top_k: int = 6, kind: str | None = None) -> list[Chunk]:
        raise RuntimeError(
            "SQLite stores raw chunk text only. Use Qdrant for vector search "
            "or set VECTOR_BACKEND=qdrant."
        )

    def search_raw_sqlite(self, query: str, top_k: int = 6) -> list[Chunk]:
        query_tokens = set(tokens_for_scoring(query))
        if not query_tokens:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT chunks.id, chunks.document_id, documents.name AS document_name,
                       documents.kind AS document_kind, chunks.chunk_index, chunks.text
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE documents.kind = 'raw'
                """
            ).fetchall()
        hits = []
        for row in rows:
            searchable_text = f"{row['document_name']} raw {row['text']}"
            score = token_overlap_score(query_tokens, searchable_text) + exact_match_bonus(query, searchable_text)
            if score > 0:
                hits.append(
                    Chunk(
                        id=row["id"],
                        document_id=row["document_id"],
                        document_name=row["document_name"],
                        document_kind=row["document_kind"],
                        chunk_index=row["chunk_index"],
                        text=row["text"],
                        score=score,
                    )
                )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: max(1, min(top_k, 20))]

    def answer(self, question: str, top_k: int = 6) -> dict:
        retrieval_query = self.rewrite_query(question)
        retrieval_k = max(top_k * 3, 12)
        record_hits = merge_hits(
            self.search(retrieval_query, top_k=retrieval_k),
            self.search(question, top_k=retrieval_k),
        )
        raw_hits = self.search_raw_sqlite(retrieval_query, top_k=max(4, top_k))
        use_records = bool(record_hits) and record_hits[0].score >= 0.08
        if use_records:
            hits = merge_hits(record_hits[:retrieval_k], raw_hits[: max(4, top_k)])
            search_mode = "qdrant-records-first"
        else:
            hits = raw_hits[:retrieval_k]
            search_mode = "sqlite-raw-fallback"
        hits = self.rerank_hits(question, hits, top_k) if hits else []
        context = format_context(hits)
        messages = [
            {
                "role": "system",
                "content": prompts.ANSWER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompts.ANSWER_USER_TEMPLATE.format(
                    search_mode=search_mode,
                    retrieval_query=retrieval_query,
                    context=context or "No matching context was found.",
                    question=question,
                ),
            },
        ]
        answer_text = self.chat(messages, max_tokens=4000)
        if not answer_text:
            answer_text = "I found relevant sources, but the model returned an empty answer. Try asking again or increase OPENAI_MODEL output limits in LM Studio."
        return {
            "answer": answer_text,
            "retrieval_query": retrieval_query,
            "search_mode": search_mode,
            "sources": [
                {
                    "id": hit.id,
                    "document_id": hit.document_id,
                    "document_name": hit.document_name,
                    "document_kind": hit.document_kind,
                    "chunk_index": hit.chunk_index,
                    "score": round(hit.score, 4),
                    "text": hit.text,
                }
                for hit in hits
            ],
        }

    def rerank_hits(self, question: str, hits: Sequence[Chunk], top_k: int) -> list[Chunk]:
        if not self.rerank_enabled or len(hits) <= top_k:
            return list(hits[:top_k])
        compact = "\n\n".join(
            f"{idx}. {hit.document_kind.upper()} | {hit.document_name} | chunk {hit.chunk_index + 1} | score {hit.score:.3f}\n"
            f"{hit.text[:900]}"
            for idx, hit in enumerate(hits[:24], start=1)
        )
        messages = [
            {
                "role": "system",
                "content": prompts.RERANK_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompts.RERANK_USER_TEMPLATE.format(
                    question=question,
                    compact=compact,
                    top_k=top_k,
                ),
            },
        ]
        try:
            text = self.chat(messages, max_tokens=800)
            numbers = [int(value) for value in re.findall(r"\d+", text)]
        except Exception:
            return list(hits[:top_k])
        selected = []
        seen = set()
        for number in numbers:
            index = number - 1
            if 0 <= index < len(hits) and index not in seen:
                selected.append(hits[index])
                seen.add(index)
            if len(selected) >= top_k:
                break
        for index, hit in enumerate(hits):
            if len(selected) >= top_k:
                break
            if index not in seen:
                selected.append(hit)
        return selected

    def chat(
        self,
        messages: Sequence[dict],
        max_tokens: int = 900,
        model: str | None = None,
    ) -> str:
        payload = {
            "model": model or self.model,
            "messages": list(messages),
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        body = self.openai_request(req, timeout=600, label="chat completion")

        try:
            return body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected chat completion response: {body}") from exc

    def embed_text(self, text: str) -> array:
        payload = {
            "model": self.embedding_model,
            "input": text,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        body = self.openai_request(req, timeout=120, label="embedding")

        try:
            values = body["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected embedding response: {body}") from exc

        vector = array("f", (float(value) for value in values))
        length = math.sqrt(sum(value * value for value in vector))
        if length:
            for idx, value in enumerate(vector):
                vector[idx] = value / length
        return vector

    def openai_request(
        self, req: urllib.request.Request, timeout: int, label: str, attempts: int = 3
    ) -> dict:
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < attempts - 1:
                    time.sleep(0.75 * (attempt + 1))
                    continue
        raise RuntimeError(f"Could not complete {label} request at {self.base_url}: {last_error}")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()


def english_only_text(text: str) -> str:
    blocks = re.split(r"\n{2,}|(?:Page\s+\d+\s*)", text)
    kept = [normalize_text(block) for block in blocks if looks_english_or_technical(block)]
    return "\n\n".join(block for block in kept if block)


def looks_english_or_technical(text: str) -> bool:
    clean = normalize_text(text)
    if len(clean) < 8:
        return False
    ascii_chars = sum(1 for char in clean if ord(char) < 128)
    ascii_ratio = ascii_chars / max(1, len(clean))
    tokens = TOKEN_RE.findall(clean.lower())
    if not tokens:
        return False
    stop_hits = sum(1 for token in tokens if token in ENGLISH_STOPWORDS)
    non_english_hits = sum(1 for token in tokens if token in NON_ENGLISH_HINTS)
    has_technical = bool(re.search(r"[A-Za-z]+[_/-]?\d+|\d+(\.\d+)?\s?(gb|mb|mhz|ghz|pcie|usb|sata|x\d+)", clean, re.I))
    if non_english_hits > stop_hits and non_english_hits >= 2:
        return False
    if ascii_ratio >= 0.95 and stop_hits >= 1:
        return True
    if ascii_ratio >= 0.9 and has_technical and non_english_hits == 0:
        return True
    return ascii_ratio >= 0.98 and len(tokens) < 40


def chunk_text(text: str, max_words: int = 260, overlap: int = 45) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks = []
    step = max(1, max_words - overlap)
    for start in range(0, len(words), step):
        piece = words[start : start + max_words]
        if piece:
            chunks.append(" ".join(piece))
        if start + max_words >= len(words):
            break
    return chunks


def parse_note_records(text: str, default_source: str = "") -> list[dict]:
    raw = text.strip()
    if not raw:
        return []
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        loose_records = loose_json_note_records(raw, default_source=default_source)
        if loose_records:
            return loose_records
        return markdown_note_records(text, default_source=default_source)
    if isinstance(parsed, dict):
        parsed = parsed.get("records") or parsed.get("items") or [parsed]
    if not isinstance(parsed, list):
        return []
    records = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        keywords = item.get("keywords") or item.get("keyword") or item.get("keys") or []
        information = item.get("information") or item.get("info") or item.get("text") or item.get("value") or ""
        source = item.get("source") or default_source
        document = item.get("document") or item.get("document_name") or ""
        if isinstance(keywords, str):
            keywords = [part.strip() for part in re.split(r"[,;|]", keywords) if part.strip()]
        elif isinstance(keywords, list):
            keywords = [normalize_text(str(part)) for part in keywords if normalize_text(str(part))]
        else:
            keywords = []
        information = normalize_text(str(information))
        source = normalize_text(str(source))
        document = normalize_text(str(document))
        if information:
            records.append(
                {
                    "keywords": keywords[:40],
                    "information": information,
                    "source": source,
                    "document": document,
                }
            )
    return records


def loose_json_note_records(text: str, default_source: str = "") -> list[dict]:
    records = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = text[start : index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        start = None
                        continue
                    if isinstance(parsed, dict):
                        records.extend(
                            parse_note_records(
                                json.dumps([parsed], ensure_ascii=True),
                                default_source=default_source,
                            )
                        )
                    start = None
    return records


def markdown_note_records(text: str, default_source: str = "") -> list[dict]:
    records = []
    blocks = re.split(r"\n(?=###\s+)", text)
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        title = lines[0].strip("# ").strip() if lines[0].startswith("#") else ""
        keywords = []
        information = []
        source = default_source
        for line in lines[1:] if title else lines:
            lower = line.lower()
            value = line.split(":", 1)[1].strip() if ":" in line else line
            if lower.startswith("- keywords:") or lower.startswith("keywords:"):
                keywords.extend(part.strip() for part in re.split(r"[,;|]", value) if part.strip())
            elif lower.startswith("- information:") or lower.startswith("information:"):
                information.append(value)
            elif lower.startswith("- source:") or lower.startswith("source:"):
                source = value
            else:
                information.append(line.lstrip("- "))
        info = normalize_text(" ".join(information))
        if title:
            keywords.insert(0, title)
        if info:
            records.append({"keywords": keywords[:40], "information": info, "source": source})
    return records


def tokens_for_scoring(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in ENGLISH_STOPWORDS
    ]


def token_overlap_score(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = set(tokens_for_scoring(text))
    if not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens) / len(query_tokens)
    phrase_bonus = 0.08 if normalize_text(" ".join(query_tokens)) in text.lower() else 0.0
    return min(1.0, overlap + phrase_bonus)


def exact_match_bonus(query: str, text: str) -> float:
    query_clean = normalize_text(query).lower()
    text_clean = normalize_text(text).lower()
    if not query_clean or not text_clean:
        return 0.0
    bonus = 0.0
    quoted_or_codes = re.findall(r"[A-Za-z]+[-_/]?[A-Za-z0-9.]*\d+[A-Za-z0-9._/-]*|\d+(?:\.\d+)?\s?(?:gb|mb|mhz|ghz|w|eur|€|\$)", query_clean)
    for item in quoted_or_codes:
        if item and item in text_clean:
            bonus += 0.04
    return min(0.16, bonus)


def merge_hits(primary: Sequence[Chunk], secondary: Sequence[Chunk]) -> list[Chunk]:
    seen = set()
    merged = []
    for hit in list(primary) + list(secondary):
        key = (hit.document_kind, hit.document_id, hit.chunk_index, hit.id)
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)
    return merged


def stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def format_context(chunks: Sequence[Chunk]) -> str:
    parts = []
    for idx, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{idx}] {chunk.document_kind.upper()} | {chunk.document_name} | chunk {chunk.chunk_index + 1} | "
            f"score {chunk.score:.3f}\n{chunk.text}"
        )
    return "\n\n".join(parts)
