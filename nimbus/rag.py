import os
from typing import Sequence

from nimbus import prompts
from nimbus.answer_engine import AnswerEngine
from nimbus.knowledge import KnowledgeBuilder
from nimbus.llm import OpenAICompatibleClient
from nimbus.models import Chunk
from nimbus.source_store import SourceBase
from nimbus.source_chunks import SourceChunkIndexer
from nimbus.vector_store import QdrantKnowledgeBase


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
        qdrant_source_collection: str,
    ) -> None:
        self.db_path = db_path
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.image_model = image_model or model
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.vector_backend = vector_backend.strip().lower() or "qdrant"
        self.qdrant_url = qdrant_url.rstrip("/")
        self.qdrant_collection = qdrant_collection.strip() or "nimbus_knowledge_base"
        self.qdrant_source_collection = qdrant_source_collection.strip() or "nimbus_source_chunks"
        self.knowledge_group_chunks = int(os.environ.get("KNOWLEDGE_GROUP_CHUNKS", "30"))
        self.knowledge_max_tokens = int(os.environ.get("KNOWLEDGE_MAX_TOKENS", "12000"))
        self.knowledge_concurrency = max(1, min(8, int(os.environ.get("KNOWLEDGE_CONCURRENCY", "1"))))
        self.source_chunk_max_words = max(120, int(os.environ.get("SOURCE_CHUNK_MAX_WORDS", "650")))
        self.prompt_version = os.environ.get("RAG_PROMPT_VERSION", "2026-04-27")
        self.rerank_enabled = os.environ.get("RAG_RERANK", "1") != "0"

        self.llm = OpenAICompatibleClient(
            base_url=self.base_url,
            api_key=self.api_key,
            chat_model=self.model,
            embedding_model=self.embedding_model,
        )
        self.source_base = SourceBase(self.db_path)
        self.knowledge_base = QdrantKnowledgeBase(
            url=self.qdrant_url,
            collection=self.qdrant_collection,
            embedding_model=self.embedding_model,
            prompt_version=self.prompt_version,
            embed_text=self.llm.embed_text,
            enabled=self.qdrant_enabled(),
            document_kind="knowledge",
            point_namespace="knowledge",
        )
        self.source_chunk_base = QdrantKnowledgeBase(
            url=self.qdrant_url,
            collection=self.qdrant_source_collection,
            embedding_model=self.embedding_model,
            prompt_version=self.prompt_version,
            embed_text=self.llm.embed_text,
            enabled=self.qdrant_enabled(),
            document_kind="source",
            point_namespace="source",
        )
        self.source_chunk_indexer = SourceChunkIndexer(
            source_base=self.source_base,
            source_chunk_base=self.source_chunk_base,
            max_words=self.source_chunk_max_words,
        )
        self.knowledge_builder = KnowledgeBuilder(
            source_base=self.source_base,
            knowledge_base=self.knowledge_base,
            llm=self.llm,
            group_size=self.knowledge_group_chunks,
            max_tokens=self.knowledge_max_tokens,
            concurrency=self.knowledge_concurrency,
        )
        self.answer_engine = AnswerEngine(
            llm=self.llm,
            search_knowledge=self.search,
            search_source=self.search_source_chunks,
            rerank_enabled=self.rerank_enabled,
        )

    def qdrant_enabled(self) -> bool:
        return self.vector_backend == "qdrant"

    def qdrant_status(self) -> dict:
        status = self.knowledge_base.status()
        status["source_chunks"] = self.source_chunk_base.status()
        return status

    def clear_qdrant_collection(self) -> None:
        self.knowledge_base.clear()
        self.source_chunk_base.clear()

    def knowledge_entries(self, limit: int = 200) -> list[dict]:
        return self.knowledge_base.entries(limit)

    def add_document(
        self,
        name: str,
        text: str,
        max_words: int = 420,
        kind: str = "source",
        english_only: bool = True,
        progress_callback=None,
    ) -> int:
        document_id = self.source_base.add_document(
            name=name,
            text=text,
            max_words=max_words,
            kind=kind,
            english_only=english_only,
            progress_callback=progress_callback,
        )
        self.source_chunk_indexer.index_document(document_id, progress_callback=progress_callback)
        return document_id

    def documents(self) -> list[dict]:
        return self.source_base.documents()

    def document(self, document_id: int) -> dict:
        return self.source_base.document(document_id)

    def document_chunks(self, document_id: int) -> list[Chunk]:
        return self.source_base.chunks_for_document(document_id)

    def delete_document(self, document_id: int) -> None:
        try:
            self.knowledge_base.delete_document(document_id)
        except Exception:
            pass
        try:
            self.source_chunk_base.delete_document(document_id)
        except Exception:
            pass
        self.source_base.delete_document(document_id)

    def build_knowledge_for_document(self, document_id: int, progress_callback=None) -> int:
        return self.knowledge_builder.build_for_document(document_id, progress_callback)

    def rebuild_knowledge_base_from_source_base(self, progress_callback=None) -> list[int]:
        self.source_chunk_indexer.rebuild_all(progress_callback)
        return self.knowledge_builder.rebuild_all(progress_callback)

    def build_knowledge_from_chunk_group(self, document_name: str, chunk_label: str, text: str) -> list[dict]:
        return self.knowledge_builder.build_group_entries(document_name, chunk_label, text)

    def extract_image_source_text(self, document_name: str, image_data: str, mime_type: str) -> str:
        data_url = f"data:{mime_type};base64,{image_data}"
        messages = [
            {"role": "system", "content": prompts.IMAGE_SOURCE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompts.IMAGE_SOURCE_USER_TEMPLATE.format(document_name=document_name),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        source_text = self.chat(messages, max_tokens=self.knowledge_max_tokens, model=self.image_model).strip()
        if not source_text:
            raise RuntimeError("Vision model returned empty image source text.")
        return source_text

    def rewrite_query(self, question: str, chat_memory: str = "") -> str:
        return self.answer_engine.rewrite_query(question, chat_memory)

    def search(self, query: str, top_k: int = 6, kind: str | None = None) -> list[Chunk]:
        if not self.qdrant_enabled():
            return self.search_sqlite(query, top_k, kind)
        return self.knowledge_base.search(query, top_k)

    def search_qdrant(self, query: str, top_k: int = 6, kind: str | None = None) -> list[Chunk]:
        return self.knowledge_base.search(query, top_k)

    def search_sqlite(self, query: str, top_k: int = 6, kind: str | None = None) -> list[Chunk]:
        raise RuntimeError(
            "Source Base stores source chunk text only. Use the Knowledge Base for vector search."
        )

    def search_raw_sqlite(self, query: str, top_k: int = 6) -> list[Chunk]:
        return self.source_base.search_source_chunks(query, top_k)

    def search_source_chunks(self, query: str, top_k: int = 6) -> list[Chunk]:
        if not self.qdrant_enabled():
            return self.search_raw_sqlite(query, top_k)
        try:
            return self.source_chunk_base.search(query, top_k)
        except Exception:
            return self.search_raw_sqlite(query, top_k)

    def answer(
        self,
        question: str,
        top_k: int = 6,
        chat_memory: str = "",
        conversation_messages: Sequence[dict] | None = None,
    ) -> dict:
        return self.answer_engine.answer(question, top_k, chat_memory, conversation_messages)

    def rerank_hits(
        self,
        question: str,
        hits: Sequence[Chunk],
        top_k: int,
        chat_memory: str = "",
    ) -> list[Chunk]:
        return self.answer_engine.rerank_hits(question, hits, top_k, chat_memory)

    def chat(
        self,
        messages: Sequence[dict],
        max_tokens: int = 900,
        model: str | None = None,
    ) -> str:
        return self.llm.chat(messages, max_tokens=max_tokens, model=model)

    def embed_text(self, text: str):
        return self.llm.embed_text(text)

    def openai_request(self, req, timeout: int, label: str, attempts: int = 3) -> dict:
        return self.llm.request_json(req, timeout=timeout, label=label, attempts=attempts)
