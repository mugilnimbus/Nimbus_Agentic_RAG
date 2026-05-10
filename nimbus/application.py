import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

from nimbus.chat_memory import ChatMemory
from nimbus.chat_store import ChatStore
from nimbus.config import env_int, env_str
from nimbus.extraction import document_text_from_payload
from nimbus.jobs import JobQueue
from nimbus.agent import NimbusAgent
from nimbus.rag import VectorRAG
from nimbus.tools import AgentToolbox


class NimbusApplication:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.jobs = JobQueue()
        self.chat_store = ChatStore(root / "data" / "chats.sqlite")
        self.config_lock = threading.Lock()
        self.extraction_workers = max(1, min(20, env_int("EXTRACTION_WORKERS", 12)))
        self.rag = self.make_rag()
        self.agent = self.make_agent()

    def make_agent(self) -> NimbusAgent:
        return NimbusAgent(
            rag=self.rag,
            toolbox=AgentToolbox(self.rag, self.settings),
        )

    def make_rag(self) -> VectorRAG:
        model = env_str("OPENAI_MODEL")
        db_path = Path(os.environ.get("RAG_DB") or self.root / "data" / "rag.sqlite")
        if not db_path.is_absolute():
            db_path = self.root / db_path
        return VectorRAG(
            db_path=str(db_path),
            base_url=env_str("OPENAI_BASE_URL"),
            model=model,
            image_model=os.environ.get("IMAGE_MODEL") or model,
            embedding_model=env_str("EMBEDDING_MODEL"),
            api_key=os.environ.get("OPENAI_API_KEY") or "local-key",
            vector_backend=os.environ.get("VECTOR_BACKEND") or "qdrant",
            qdrant_url=os.environ.get("QDRANT_URL") or "http://127.0.0.1:6333",
            qdrant_collection=os.environ.get("QDRANT_COLLECTION") or "nimbus_knowledge_base",
            qdrant_source_collection=os.environ.get("QDRANT_SOURCE_COLLECTION") or "nimbus_source_chunks",
        )

    def health(self) -> dict:
        return {
            "ok": True,
            "base_url": self.rag.base_url,
            "model": self.rag.model,
            "image_model": self.rag.image_model,
            "embedding_model": self.rag.embedding_model,
            "knowledge_concurrency": self.rag.knowledge_concurrency,
            "knowledge_group_chunks": self.rag.knowledge_group_chunks,
            "knowledge_max_tokens": self.rag.knowledge_max_tokens,
            "source_chunk_max_words": self.rag.source_chunk_max_words,
            "rerank_enabled": self.rag.rerank_enabled,
            "agent_max_steps": self.agent.max_steps,
            "vector_backend": self.rag.vector_backend,
            "qdrant": self.rag.qdrant_status(),
            "documents": len(self.rag.documents()),
        }

    def settings(self) -> dict:
        return {
            "base_url": self.rag.base_url,
            "model": self.rag.model,
            "image_model": self.rag.image_model,
            "embedding_model": self.rag.embedding_model,
            "knowledge_concurrency": self.rag.knowledge_concurrency,
            "knowledge_group_chunks": self.rag.knowledge_group_chunks,
            "knowledge_max_tokens": self.rag.knowledge_max_tokens,
            "source_chunk_max_words": self.rag.source_chunk_max_words,
            "extraction_workers": self.extraction_workers,
            "rerank_enabled": self.rag.rerank_enabled,
            "agent_max_steps": self.agent.max_steps,
            "vector_backend": self.rag.vector_backend,
            "qdrant_url": self.rag.qdrant_url,
            "qdrant_collection": self.rag.qdrant_collection,
            "qdrant_source_collection": self.rag.qdrant_source_collection,
            "qdrant": self.rag.qdrant_status(),
        }

    def connections(self) -> dict:
        return {
            "llm": self.openai_compatible_status(),
            "qdrant": self.rag.qdrant_status(),
        }

    def openai_compatible_status(self) -> dict:
        request = urllib.request.Request(
            f"{self.rag.base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {self.rag.api_key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.URLError as exc:
            return {"status": "unavailable", "url": self.rag.base_url, "error": str(exc)}
        except (TimeoutError, json.JSONDecodeError) as exc:
            return {"status": "unavailable", "url": self.rag.base_url, "error": str(exc)}

        models = payload.get("data") if isinstance(payload, dict) else []
        model_ids = [
            str(item.get("id"))
            for item in models
            if isinstance(item, dict) and item.get("id")
        ]
        return {
            "status": "ready",
            "url": self.rag.base_url,
            "models": model_ids[:12],
            "model_count": len(model_ids),
        }

    def update_settings(self, payload: dict) -> None:
        with self.config_lock:
            base_url = str(payload.get("base_url") or self.rag.base_url).strip()
            model = str(payload.get("model") or self.rag.model).strip()
            image_model = str(payload.get("image_model") or self.rag.image_model or model).strip()
            embedding_model = str(payload.get("embedding_model") or self.rag.embedding_model).strip()
            vector_backend = str(payload.get("vector_backend") or self.rag.vector_backend).strip().lower()
            if vector_backend != "qdrant":
                raise ValueError("Vector backend must be qdrant. SQLite stores Source Base chunks only.")

            os.environ["OPENAI_BASE_URL"] = base_url
            os.environ["OPENAI_MODEL"] = model
            os.environ["IMAGE_MODEL"] = image_model
            os.environ["EMBEDDING_MODEL"] = embedding_model
            os.environ["VECTOR_BACKEND"] = vector_backend
            os.environ["QDRANT_URL"] = str(payload.get("qdrant_url") or self.rag.qdrant_url).strip()
            os.environ["QDRANT_COLLECTION"] = str(
                payload.get("qdrant_collection") or self.rag.qdrant_collection
            ).strip()
            os.environ["QDRANT_SOURCE_COLLECTION"] = str(
                payload.get("qdrant_source_collection") or self.rag.qdrant_source_collection
            ).strip()
            os.environ["KNOWLEDGE_CONCURRENCY"] = str(
                max(1, min(8, int(payload.get("knowledge_concurrency") or self.rag.knowledge_concurrency)))
            )
            os.environ["KNOWLEDGE_GROUP_CHUNKS"] = str(
                max(1, min(100, int(payload.get("knowledge_group_chunks") or self.rag.knowledge_group_chunks)))
            )
            os.environ["KNOWLEDGE_MAX_TOKENS"] = str(
                max(512, min(32000, int(payload.get("knowledge_max_tokens") or self.rag.knowledge_max_tokens)))
            )
            os.environ["SOURCE_CHUNK_MAX_WORDS"] = str(
                max(120, min(4000, int(payload.get("source_chunk_max_words") or self.rag.source_chunk_max_words)))
            )
            self.extraction_workers = max(
                1,
                min(20, int(payload.get("extraction_workers") or self.extraction_workers)),
            )
            os.environ["EXTRACTION_WORKERS"] = str(self.extraction_workers)
            os.environ["RAG_RERANK"] = "1" if bool(payload.get("rerank_enabled", self.rag.rerank_enabled)) else "0"
            os.environ["AGENT_MAX_STEPS"] = str(
                max(1, min(12, int(payload.get("agent_max_steps") or self.agent.max_steps)))
            )
            self.rag = self.make_rag()
            self.agent = self.make_agent()

    def ask(self, question: str, top_k: int, chat_id: int | None = None) -> dict:
        chat = self.chat_store.get_or_create_chat(chat_id)
        memory = self.chat_memory_for_chat(chat["id"])
        chat_memory = memory.as_prompt_text()
        conversation_messages = memory.as_messages()
        result = self.rag.answer(
            question,
            top_k=top_k,
            chat_memory=chat_memory,
            conversation_messages=conversation_messages,
        )
        self.chat_store.append_message(chat["id"], "user", question)
        self.chat_store.append_message(
            chat["id"],
            "assistant",
            str(result.get("answer") or ""),
            sources=result.get("sources") or [],
            focus_entities=result.get("focus_entities") or [],
        )
        result["chat_id"] = chat["id"]
        return result

    def ask_agent(self, question: str, top_k: int, chat_id: int | None = None) -> dict:
        chat = self.chat_store.get_or_create_chat(chat_id)
        memory = self.chat_memory_for_chat(chat["id"])
        chat_memory = memory.as_prompt_text()
        conversation_messages = memory.as_messages()
        result = self.agent.answer(
            question,
            top_k=top_k,
            chat_memory=chat_memory,
            conversation_messages=conversation_messages,
        )
        self.chat_store.append_message(chat["id"], "user", question)
        self.chat_store.append_message(
            chat["id"],
            "assistant",
            str(result.get("answer") or ""),
            sources=result.get("sources") or [],
            focus_entities=result.get("focus_entities") or [],
        )
        result["chat_id"] = chat["id"]
        return result

    def chat_memory_for_chat(self, chat_id: int) -> ChatMemory:
        return ChatMemory.from_turns(
            self.chat_store.turns(chat_id),
            max_turns=env_int("CHAT_MEMORY_TURNS", 24),
            max_summary_chars=env_int("CHAT_MEMORY_SUMMARY_CHARS", 700),
            max_message_chars=env_int("CHAT_MEMORY_MESSAGE_CHARS", 4000),
        )

    def queue_ingest(self, payload: dict) -> int:
        return self.jobs.queue(
            "ingest",
            str(payload.get("name") or "Pasted document"),
            self.ingest_document,
            payload,
        )

    def queue_build_knowledge(self, document_id: int) -> int:
        return self.jobs.queue(
            "build-knowledge",
            f"Build Knowledge Base for document {document_id}",
            self.build_knowledge_for_document,
            document_id,
        )

    def queue_rebuild_knowledge(self) -> int:
        return self.jobs.queue(
            "rebuild-knowledge",
            "Rebuild Knowledge Base from all Source Base documents",
            self.rebuild_knowledge_base,
            None,
        )

    def ingest_document(self, payload: dict, job_id: int) -> dict:
        self.jobs.update(job_id, "running", "Extracting text")
        text, kind, should_build_knowledge = document_text_from_payload(
            payload,
            self.rag,
            self.extraction_workers,
        )
        self.jobs.update(job_id, "running", f"Indexing {kind} source")
        document_id = self.rag.add_document(
            str(payload.get("name") or "Pasted document"),
            text,
            kind=kind,
            english_only=True,
            progress_callback=self.progress_callback(job_id),
        )
        knowledge_count = None
        if should_build_knowledge:
            self.jobs.update(job_id, "running", "Building Knowledge Base")
            knowledge_count = self.rag.build_knowledge_for_document(
                document_id,
                progress_callback=self.progress_callback(job_id),
            )
        return {"document_id": document_id, "knowledge_entries": knowledge_count}

    def build_knowledge_for_document(self, document_id: int, job_id: int) -> dict:
        self.jobs.update(job_id, "running", f"Building Knowledge Base for document {document_id}")
        entry_count = self.rag.build_knowledge_for_document(
            document_id,
            progress_callback=self.progress_callback(job_id),
        )
        return {"document_id": document_id, "knowledge_entries": entry_count}

    def rebuild_knowledge_base(self, _arg, job_id: int) -> dict:
        self.jobs.update(job_id, "running", "Rebuilding Source and Knowledge Base indexes")
        rebuilt = self.rag.rebuild_knowledge_base_from_source_base(
            progress_callback=self.progress_callback(job_id),
        )
        return {"entries_by_document": rebuilt, "knowledge_entries": sum(rebuilt)}

    def progress_callback(self, job_id: int):
        def update_progress(current, total, detail):
            self.jobs.update(
                job_id,
                "running",
                detail,
                progress_current=current,
                progress_total=total,
            )

        return update_progress
