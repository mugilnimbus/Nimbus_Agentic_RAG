import json
import os
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from nimbus.config import env_int, env_str, load_env_file
from nimbus.extraction import document_text_from_payload
from nimbus.jobs import JobQueue
from nimbus.rag import VectorRAG


ROOT = Path(__file__).resolve().parent.parent
load_env_file(ROOT / ".env")
STATIC_DIR = ROOT / "static"
EXTRACTION_WORKERS = max(1, min(20, env_int("EXTRACTION_WORKERS", 12)))


def make_rag() -> VectorRAG:
    return VectorRAG(
        db_path=env_str("RAG_DB"),
        base_url=env_str("OPENAI_BASE_URL"),
        model=env_str("OPENAI_MODEL"),
        image_model=env_str("IMAGE_MODEL", env_str("OPENAI_MODEL")),
        embedding_model=env_str("EMBEDDING_MODEL"),
        api_key=env_str("OPENAI_API_KEY"),
        vector_backend=env_str("VECTOR_BACKEND"),
        qdrant_url=env_str("QDRANT_URL"),
        qdrant_collection=env_str("QDRANT_COLLECTION"),
    )


RAG = make_rag()
JOBS = JobQueue()
CONFIG_LOCK = threading.Lock()


class RAGHandler(SimpleHTTPRequestHandler):
    server_version = "NimbusVectorRAG/1.0"

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        clean = parsed.path
        if clean == "/":
            clean = "/index.html"
        return str(STATIC_DIR / clean.lstrip("/"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_json(
                {
                    "ok": True,
                    "base_url": RAG.base_url,
                    "model": RAG.model,
                    "image_model": RAG.image_model,
                    "embedding_model": RAG.embedding_model,
                    "notes_concurrency": RAG.notes_concurrency,
                    "notes_group_chunks": RAG.notes_group_chunks,
                    "notes_max_tokens": RAG.notes_max_tokens,
                    "rerank_enabled": RAG.rerank_enabled,
                    "vector_backend": RAG.vector_backend,
                    "qdrant": RAG.qdrant_status(),
                    "documents": len(RAG.documents()),
                }
            )
            return
        if parsed.path == "/api/settings":
            self.send_json(settings_payload())
            return
        if parsed.path == "/api/documents":
            self.send_json({"documents": RAG.documents()})
            return
        if parsed.path == "/api/records":
            params = parse_qs(parsed.query)
            limit = int(params.get("limit", ["200"])[0])
            self.send_json({"records": RAG.qdrant_records(limit=limit)})
            return
        if parsed.path == "/api/jobs":
            params = parse_qs(parsed.query)
            limit = int(params.get("limit", ["20"])[0])
            self.send_json({"jobs": JOBS.jobs(limit=limit)})
            return
        if parsed.path.startswith("/api/documents/") and parsed.path.endswith("/chunks"):
            try:
                document_id = int(parsed.path.split("/")[-2])
            except ValueError:
                self.send_json({"error": "Invalid document id."}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json(
                {
                    "chunks": [
                        {
                            "id": chunk.id,
                            "document_id": chunk.document_id,
                            "document_name": chunk.document_name,
                            "document_kind": chunk.document_kind,
                            "chunk_index": chunk.chunk_index,
                            "text": chunk.text,
                        }
                        for chunk in RAG.document_chunks(document_id)
                    ]
                }
            )
            return
        if parsed.path == "/api/search":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            top_k = int(params.get("top_k", ["6"])[0])
            hits = RAG.search(query, top_k=top_k) if query.strip() else []
            self.send_json(
                {
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
                    ]
                }
            )
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/documents":
                payload = self.read_json()
                job_id = JOBS.queue(
                    "ingest",
                    str(payload.get("name") or "Pasted document"),
                    ingest_payload,
                    payload,
                )
                self.send_json(
                    {
                        "ok": True,
                        "queued": True,
                        "job_id": job_id,
                    },
                    HTTPStatus.ACCEPTED,
                )
                return
            if parsed.path.startswith("/api/documents/") and parsed.path.endswith("/distill"):
                document_id = int(parsed.path.split("/")[-2])
                job_id = JOBS.queue(
                    "distill",
                    f"Build Qdrant records for document {document_id}",
                    distill_payload,
                    document_id,
                )
                self.send_json({"ok": True, "queued": True, "job_id": job_id}, HTTPStatus.ACCEPTED)
                return
            if parsed.path == "/api/ask":
                payload = self.read_json()
                question = str(payload.get("question") or "").strip()
                top_k = int(payload.get("top_k") or 6)
                if not question:
                    self.send_json({"error": "Question is required."}, HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(RAG.answer(question, top_k=top_k))
                return
            if parsed.path == "/api/rebuild-records":
                job_id = JOBS.queue(
                    "rebuild-records",
                    "Rebuild Qdrant records for all raw documents",
                    rebuild_records_payload,
                    None,
                )
                self.send_json({"ok": True, "queued": True, "job_id": job_id}, HTTPStatus.ACCEPTED)
                return
            if parsed.path == "/api/settings":
                payload = self.read_json()
                update_settings(payload)
                self.send_json(settings_payload())
                return
            self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/documents/"):
            try:
                document_id = int(parsed.path.rsplit("/", 1)[-1])
            except ValueError:
                self.send_json({"error": "Invalid document id."}, HTTPStatus.BAD_REQUEST)
                return
            RAG.delete_document(document_id)
            self.send_json({"ok": True})
            return
        self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def ingest_payload(payload: dict, job_id: int) -> dict:
    JOBS.update(job_id, "running", "Extracting text and preparing chunks")
    text, kind, should_distill = document_text_from_payload(payload, RAG, EXTRACTION_WORKERS)
    JOBS.update(job_id, "running", f"Indexing {kind} source")
    def progress(current, total, detail):
        JOBS.update(
            job_id,
            "running",
            detail,
            progress_current=current,
            progress_total=total,
        )
    name = str(payload.get("name") or "Pasted document")
    document_id = RAG.add_document(name, text, kind=kind, english_only=True, progress_callback=progress)
    distilled_id = None
    if should_distill:
        JOBS.update(job_id, "running", "Building Qdrant records")
        distilled_id = RAG.distill_document(document_id, progress_callback=progress)
    return {"document_id": document_id, "distilled_document_id": distilled_id}


def distill_payload(document_id: int, job_id: int) -> dict:
    JOBS.update(job_id, "running", f"Distilling document {document_id} into Qdrant records")
    def progress(current, total, detail):
        JOBS.update(
            job_id,
            "running",
            detail,
            progress_current=current,
            progress_total=total,
        )
    record_count = RAG.distill_document(document_id, progress_callback=progress)
    return {"document_id": document_id, "qdrant_records": record_count}


def rebuild_records_payload(_arg, job_id: int) -> dict:
    JOBS.update(job_id, "running", "Rebuilding Qdrant records from raw SQLite chunks")
    def progress(current, total, detail):
        JOBS.update(
            job_id,
            "running",
            detail,
            progress_current=current,
            progress_total=total,
        )
    rebuilt = RAG.rebuild_records_for_all_raw_documents(progress_callback=progress)
    return {"records_by_document": rebuilt, "qdrant_records": sum(rebuilt)}


def settings_payload() -> dict:
    return {
        "base_url": RAG.base_url,
        "model": RAG.model,
        "image_model": RAG.image_model,
        "embedding_model": RAG.embedding_model,
        "notes_concurrency": RAG.notes_concurrency,
        "notes_group_chunks": RAG.notes_group_chunks,
        "notes_max_tokens": RAG.notes_max_tokens,
        "extraction_workers": EXTRACTION_WORKERS,
        "rerank_enabled": RAG.rerank_enabled,
        "vector_backend": RAG.vector_backend,
        "qdrant_url": RAG.qdrant_url,
        "qdrant_collection": RAG.qdrant_collection,
        "qdrant": RAG.qdrant_status(),
    }


def update_settings(payload: dict) -> None:
    global RAG, EXTRACTION_WORKERS
    with CONFIG_LOCK:
        base_url = str(payload.get("base_url") or RAG.base_url).strip()
        model = str(payload.get("model") or RAG.model).strip()
        image_model = str(payload.get("image_model") or RAG.image_model or model).strip()
        embedding_model = str(payload.get("embedding_model") or RAG.embedding_model).strip()
        vector_backend = str(payload.get("vector_backend") or RAG.vector_backend).strip().lower()
        if vector_backend != "qdrant":
            raise ValueError("Vector backend must be qdrant. SQLite stores raw chunks only.")
        qdrant_url = str(payload.get("qdrant_url") or RAG.qdrant_url).strip()
        qdrant_collection = str(payload.get("qdrant_collection") or RAG.qdrant_collection).strip()
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_MODEL"] = model
        os.environ["IMAGE_MODEL"] = image_model
        os.environ["EMBEDDING_MODEL"] = embedding_model
        os.environ["VECTOR_BACKEND"] = vector_backend
        os.environ["QDRANT_URL"] = qdrant_url
        os.environ["QDRANT_COLLECTION"] = qdrant_collection
        os.environ["AI_NOTES_CONCURRENCY"] = str(
            max(1, min(8, int(payload.get("notes_concurrency") or RAG.notes_concurrency)))
        )
        os.environ["AI_NOTES_GROUP_CHUNKS"] = str(
            max(1, min(100, int(payload.get("notes_group_chunks") or RAG.notes_group_chunks)))
        )
        os.environ["AI_NOTES_MAX_TOKENS"] = str(
            max(512, min(32000, int(payload.get("notes_max_tokens") or RAG.notes_max_tokens)))
        )
        EXTRACTION_WORKERS = max(
            1, min(20, int(payload.get("extraction_workers") or EXTRACTION_WORKERS))
        )
        os.environ["EXTRACTION_WORKERS"] = str(EXTRACTION_WORKERS)
        os.environ["RAG_RERANK"] = "1" if bool(payload.get("rerank_enabled", RAG.rerank_enabled)) else "0"
        RAG = make_rag()


def main() -> None:
    port = env_int("PORT", 8000)
    server = ThreadingHTTPServer(("0.0.0.0", port), RAGHandler)
    print(f"Nimbus UI listening on configured port {port}")
    print("Model, embedding, endpoint, and vector database settings loaded from .env")
    print(f"Vector backend: {RAG.vector_backend}")
    server.serve_forever()


if __name__ == "__main__":
    main()
