import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from nimbus.application import NimbusApplication
from nimbus.config import env_int, env_str, load_env_file
from nimbus.models import Chunk


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"

load_env_file(ROOT / ".env")
APP = NimbusApplication(ROOT)


class RAGHandler(SimpleHTTPRequestHandler):
    server_version = "NimbusVectorRAG/1.0"

    def translate_path(self, path: str) -> str:
        parsed_path = unquote(urlparse(path).path)
        static_path = "/index.html" if parsed_path == "/" else parsed_path
        requested = (STATIC_DIR / static_path.lstrip("/")).resolve()
        static_root = STATIC_DIR.resolve()
        try:
            requested.relative_to(static_root)
        except ValueError:
            return str(static_root / "__not_found__")
        if requested.is_dir():
            return str(requested / "index.html")
        return str(requested)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            handled = self.handle_api_get(parsed.path, parse_qs(parsed.query))
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:
            if parsed.path.startswith("/api/"):
                self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            raise
        if not handled:
            super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            handled = self.handle_api_post(parsed.path)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if not handled:
            self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_PATCH(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/chats/"):
                payload = self.read_json()
                chat_id = self.document_id_from_path(parsed.path)
                self.send_json({"chat": APP.chat_store.rename_chat(chat_id, str(payload.get("title") or ""))})
                return
            self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/chats/"):
                APP.chat_store.delete_chat(self.document_id_from_path(parsed.path))
                self.send_json({"ok": True})
                return
            if parsed.path.startswith("/api/documents/"):
                APP.rag.delete_document(self.document_id_from_path(parsed.path))
                self.send_json({"ok": True})
                return
            self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_api_get(self, path: str, params: dict[str, list[str]]) -> bool:
        if path == "/api/health":
            self.send_json(APP.health())
            return True
        if path == "/api/settings":
            self.send_json(APP.settings())
            return True
        if path == "/api/connections":
            self.send_json(APP.connections())
            return True
        if path == "/api/documents":
            self.send_json({"documents": APP.rag.documents()})
            return True
        if path == "/api/knowledge":
            self.send_json({"entries": APP.rag.knowledge_entries(limit=self.query_int(params, "limit", 200))})
            return True
        if path == "/api/jobs":
            self.send_json({"jobs": APP.jobs.jobs(limit=self.query_int(params, "limit", 20))})
            return True
        if path == "/api/chats":
            self.send_json({"chats": APP.chat_store.list_chats()})
            return True
        if path.startswith("/api/chats/") and path.endswith("/messages"):
            self.send_json({"messages": APP.chat_store.messages(self.document_id_from_path(path, offset=-2))})
            return True
        if path == "/api/search":
            self.send_json(self.search_response(params))
            return True
        if path.startswith("/api/documents/") and path.endswith("/chunks"):
            self.send_json(self.document_chunks_response(path))
            return True
        return False

    def handle_api_post(self, path: str) -> bool:
        if path == "/api/documents":
            job_id = APP.queue_ingest(self.read_json())
            self.send_json(self.queued_response(job_id), HTTPStatus.ACCEPTED)
            return True
        if path == "/api/chats":
            payload = self.read_json()
            self.send_json({"chat": APP.chat_store.create_chat(str(payload.get("title") or "New chat"))}, HTTPStatus.CREATED)
            return True
        if path.startswith("/api/chats/") and path.endswith("/rename"):
            payload = self.read_json()
            chat_id = self.document_id_from_path(path, offset=-2)
            self.send_json({"chat": APP.chat_store.rename_chat(chat_id, str(payload.get("title") or ""))})
            return True
        if path.startswith("/api/chats/") and path.endswith("/delete"):
            APP.chat_store.delete_chat(self.document_id_from_path(path, offset=-2))
            self.send_json({"ok": True})
            return True
        if path.startswith("/api/documents/") and path.endswith("/build-knowledge"):
            job_id = APP.queue_build_knowledge(self.document_id_from_path(path, offset=-2))
            self.send_json(self.queued_response(job_id), HTTPStatus.ACCEPTED)
            return True
        if path == "/api/ask":
            self.send_json(self.ask_response())
            return True
        if path == "/api/agent/ask":
            self.send_json(self.agent_ask_response())
            return True
        if path == "/api/rebuild-knowledge":
            job_id = APP.queue_rebuild_knowledge()
            self.send_json(self.queued_response(job_id), HTTPStatus.ACCEPTED)
            return True
        if path == "/api/settings":
            APP.update_settings(self.read_json())
            self.send_json(APP.settings())
            return True
        return False

    def ask_response(self) -> dict:
        payload = self.read_json()
        question = str(payload.get("question") or "").strip()
        if not question:
            raise ValueError("Question is required.")
        chat_id = payload.get("chat_id")
        return APP.ask(question, int(payload.get("top_k") or 6), int(chat_id) if chat_id else None)

    def agent_ask_response(self) -> dict:
        payload = self.read_json()
        question = str(payload.get("question") or "").strip()
        if not question:
            raise ValueError("Question is required.")
        chat_id = payload.get("chat_id")
        return APP.ask_agent(question, int(payload.get("top_k") or 6), int(chat_id) if chat_id else None)

    def search_response(self, params: dict[str, list[str]]) -> dict:
        query = params.get("q", [""])[0]
        top_k = self.query_int(params, "top_k", 6)
        hits = APP.rag.search(query, top_k=top_k) if query.strip() else []
        return {"sources": [self.chunk_payload(hit, include_score=True) for hit in hits]}

    def document_chunks_response(self, path: str) -> dict:
        document_id = self.document_id_from_path(path, offset=-2)
        chunks = APP.rag.document_chunks(document_id)
        return {"chunks": [self.chunk_payload(chunk, include_score=False) for chunk in chunks]}

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

    @staticmethod
    def document_id_from_path(path: str, offset: int = -1) -> int:
        try:
            return int(path.split("/")[offset])
        except (IndexError, ValueError) as exc:
            raise ValueError("Invalid id.") from exc

    @staticmethod
    def query_int(params: dict[str, list[str]], name: str, default: int) -> int:
        return int(params.get(name, [str(default)])[0])

    @staticmethod
    def queued_response(job_id: int) -> dict:
        return {"ok": True, "queued": True, "job_id": job_id}

    @staticmethod
    def chunk_payload(chunk: Chunk, include_score: bool) -> dict:
        payload = {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "document_kind": chunk.document_kind,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
        }
        if include_score:
            payload["score"] = round(chunk.score, 4)
        return payload


def main() -> None:
    host = env_str("HOST", "127.0.0.1")
    port = env_int("PORT", 8000)
    server = ThreadingHTTPServer((host, port), RAGHandler)
    print(f"Nimbus UI listening on http://{host}:{port}")
    print("Model, embedding, endpoint, and vector database settings loaded from .env")
    print(f"Vector backend: {APP.rag.vector_backend}")
    server.serve_forever()


if __name__ == "__main__":
    main()
