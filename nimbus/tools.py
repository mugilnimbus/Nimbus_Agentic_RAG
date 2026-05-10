import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nimbus.models import Chunk
from nimbus.text_processing import normalize_text


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    parameters: dict[str, str]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


class AgentToolbox:
    def __init__(self, rag, settings_provider: Callable[[], dict]) -> None:
        self.rag = rag
        self.settings_provider = settings_provider
        self.tools = {
            tool.name.lower(): tool
            for tool in (
                AgentTool(
                    name="search_knowledge",
                    description="Search compact distilled Knowledge Base entries for direct facts, constraints, relationships, and reusable evidence.",
                    parameters={"query": "Search text.", "top_k": "Optional number of results, 1 to 10."},
                    handler=self.search_knowledge,
                ),
                AgentTool(
                    name="search_source",
                    description="Search semantic Source Document chunks for raw fallback evidence and exact source wording.",
                    parameters={"query": "Search text.", "top_k": "Optional number of results, 1 to 10."},
                    handler=self.search_source,
                ),
                AgentTool(
                    name="list_documents",
                    description="List uploaded Source Documents with ids, names, dates, and indexed chunk counts.",
                    parameters={"limit": "Optional number of documents, 1 to 50."},
                    handler=self.list_documents,
                ),
                AgentTool(
                    name="open_source_document",
                    description="Open one full Source Document by id when the agent needs broader context than search snippets.",
                    parameters={"document_id": "Source Document id.", "max_chars": "Optional character budget, 1000 to 20000."},
                    handler=self.open_source_document,
                ),
                AgentTool(
                    name="inspect_settings",
                    description="Inspect runtime model, embedding, retrieval, and processing settings.",
                    parameters={},
                    handler=self.inspect_settings,
                ),
            )
        }

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self.tools.values()
        ]

    def run(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        tool = self.tools.get(normalize_text(name).lower())
        if tool is None:
            return {"ok": False, "error": f"Unknown tool: {name}", "content": ""}
        try:
            result = tool.handler(arguments or {})
        except Exception as exc:
            return {"ok": False, "error": str(exc), "content": ""}
        return {"ok": True, **result}

    def search_knowledge(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = self.required_text(arguments, "query")
        hits = self.rag.search(query, top_k=self.top_k(arguments))
        return {
            "content": self.format_hits(hits),
            "sources": [self.source_payload(hit) for hit in hits],
        }

    def search_source(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = self.required_text(arguments, "query")
        hits = self.rag.search_source_chunks(query, top_k=self.top_k(arguments))
        return {
            "content": self.format_hits(hits),
            "sources": [self.source_payload(hit) for hit in hits],
        }

    def list_documents(self, arguments: dict[str, Any]) -> dict[str, Any]:
        limit = self.clamped_int(arguments.get("limit"), default=20, minimum=1, maximum=50)
        documents = self.rag.documents()[:limit]
        rows = [
            f"- id={document['id']} name={document['name']} chunks={document.get('chunk_count', 0)}"
            for document in documents
        ]
        return {
            "content": "\n".join(rows) if rows else "No Source Documents are indexed.",
            "documents": documents,
        }

    def open_source_document(self, arguments: dict[str, Any]) -> dict[str, Any]:
        document_id = self.clamped_int(arguments.get("document_id"), default=0, minimum=1, maximum=10**12)
        max_chars = self.clamped_int(arguments.get("max_chars"), default=12000, minimum=1000, maximum=20000)
        document = self.rag.document(document_id)
        chunks = self.rag.document_chunks(document_id)
        text = chunks[0].text if chunks else ""
        truncated = len(text) > max_chars
        visible_text = text[:max_chars]
        source = {
            "id": f"document-{document_id}",
            "document_id": int(document["id"]),
            "document_name": str(document["name"]),
            "document_kind": "source-document",
            "chunk_index": 0,
            "score": 1.0,
            "text": visible_text,
        }
        return {
            "content": f"Document id={document_id}, name={document['name']}, characters_returned={len(visible_text)}, truncated={truncated}\n\n{visible_text}",
            "sources": [source],
        }

    def inspect_settings(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        settings = self.settings_provider()
        safe_settings = {
            key: value
            for key, value in settings.items()
            if key not in {"api_key", "openai_api_key"}
        }
        return {"content": json.dumps(safe_settings, ensure_ascii=True, indent=2)}

    @staticmethod
    def required_text(arguments: dict[str, Any], name: str) -> str:
        value = normalize_text(str(arguments.get(name) or ""))
        if not value:
            raise ValueError(f"{name} is required.")
        return value

    @staticmethod
    def top_k(arguments: dict[str, Any]) -> int:
        return AgentToolbox.clamped_int(arguments.get("top_k"), default=6, minimum=1, maximum=10)

    @staticmethod
    def clamped_int(value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        return max(minimum, min(maximum, number))

    @staticmethod
    def format_hits(hits: list[Chunk]) -> str:
        if not hits:
            return "No matching results."
        return "\n\n".join(
            f"[{index}] {hit.document_kind} | {hit.document_name} | chunk {hit.chunk_index + 1} | score {hit.score:.4f}\n{hit.text}"
            for index, hit in enumerate(hits, start=1)
        )

    @staticmethod
    def source_payload(hit: Chunk) -> dict[str, Any]:
        return {
            "id": hit.id,
            "document_id": hit.document_id,
            "document_name": hit.document_name,
            "document_kind": hit.document_kind,
            "chunk_index": hit.chunk_index,
            "score": round(hit.score, 4),
            "text": hit.text,
        }
