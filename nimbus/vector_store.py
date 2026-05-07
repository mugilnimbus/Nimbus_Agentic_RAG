import json
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Sequence

from nimbus.models import Chunk
from nimbus.retrieval import exact_match_bonus, token_overlap_score, tokens_for_scoring
from nimbus.text_processing import normalize_document_text, normalize_text, stable_hash


class QdrantKnowledgeBase:
    def __init__(
        self,
        url: str,
        collection: str,
        embedding_model: str,
        prompt_version: str,
        embed_text: Callable[[str], Sequence[float]],
        enabled: bool = True,
        document_kind: str = "knowledge",
        point_namespace: str = "knowledge",
    ) -> None:
        self.url = url.rstrip("/")
        self.collection = collection.strip() or "nimbus_knowledge_base"
        self.embedding_model = embedding_model
        self.prompt_version = prompt_version
        self.embed_text = embed_text
        self.enabled = enabled
        self.document_kind = document_kind
        self.point_namespace = point_namespace
        self._ready_dimensions: set[int] = set()

    def status(self) -> dict:
        if not self.enabled:
            return {"enabled": False, "status": "disabled"}
        try:
            response = self.request("GET", f"/collections/{self.collection}", timeout=2)
            result = response.get("result") or {}
            return {
                "enabled": True,
                "status": "ready",
                "url": self.url,
                "collection": self.collection,
                "points_count": int(result.get("points_count") or 0),
                "vectors_count": int(result.get("vectors_count") or 0),
            }
        except Exception as exc:
            if "HTTP 404" in str(exc):
                return {
                    "enabled": True,
                    "status": "collection_missing",
                    "url": self.url,
                    "collection": self.collection,
                }
            return {
                "enabled": True,
                "status": "unavailable",
                "url": self.url,
                "collection": self.collection,
                "error": str(exc),
            }

    def clear(self) -> None:
        if not self.enabled:
            return
        try:
            self.request("DELETE", f"/collections/{self.collection}?timeout=60")
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise
        self._ready_dimensions.clear()

    def entries(self, limit: int = 200) -> list[dict]:
        if not self.enabled:
            return []
        try:
            response = self.request(
                "POST",
                f"/collections/{self.collection}/points/scroll",
                {
                    "limit": max(1, min(int(limit), 1000)),
                    "with_payload": True,
                    "with_vector": False,
                    "filter": self.payload_filter(vector_model=self.embedding_model),
                },
            )
        except RuntimeError as exc:
            if "HTTP 404" in str(exc):
                return []
            raise

        entries = [self.entry_from_point(point) for point in (response.get("result") or {}).get("points", [])]
        entries.sort(key=lambda item: (item["document_name"], item["source_chunk_start"], item["id"]))
        return entries

    def search(self, query: str, top_k: int = 6) -> list[Chunk]:
        query_vector = self.embed_text(query)
        self.ensure_collection(len(query_vector))
        query_tokens = set(tokens_for_scoring(query))
        response = self.request(
            "POST",
            f"/collections/{self.collection}/points/search",
            {
                "vector": list(query_vector),
                "limit": max(top_k * 4, 24),
                "with_payload": True,
                "filter": self.payload_filter(vector_model=self.embedding_model),
            },
        )

        candidates = []
        for item in response.get("result", []):
            chunk = self.chunk_from_search_point(item, query, query_tokens)
            if chunk.score > 0:
                candidates.append(chunk)
        candidates.sort(key=lambda chunk: chunk.score, reverse=True)
        return candidates[: max(1, min(top_k, 20))]

    def upsert(self, entries: Sequence[dict]) -> int:
        if not self.enabled or not entries:
            return 0

        points = []
        vector_dimension = 0
        for entry in entries:
            point = self.point_from_entry(entry)
            if point is None:
                continue
            vector_dimension = len(point["vector"])
            points.append(point)

        if not points:
            return 0

        self.ensure_collection(vector_dimension)
        for start in range(0, len(points), 128):
            self.request(
                "PUT",
                f"/collections/{self.collection}/points?wait=true",
                {"points": points[start:start + 128]},
            )
        return len(points)

    def delete_document(self, document_id: int) -> None:
        if not self.enabled:
            return
        try:
            self.request(
                "POST",
                f"/collections/{self.collection}/points/delete?wait=true",
                {"filter": self.payload_filter(document_id=int(document_id))},
            )
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise

    def ensure_collection(self, vector_dimension: int) -> None:
        if not self.enabled or vector_dimension in self._ready_dimensions:
            return
        try:
            response = self.request("GET", f"/collections/{self.collection}", timeout=5)
            config = response.get("result", {}).get("config", {}).get("params", {})
            vectors = config.get("vectors", {})
            existing_size = vectors.get("size") if isinstance(vectors, dict) else None
            if existing_size and int(existing_size) != int(vector_dimension):
                raise RuntimeError(
                    f"Qdrant collection '{self.collection}' has vector size {existing_size}, "
                    f"but {self.embedding_model} produced {vector_dimension}. "
                    "Use a new QDRANT_COLLECTION or recreate the collection."
                )
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise
            self.request(
                "PUT",
                f"/collections/{self.collection}",
                {"vectors": {"size": vector_dimension, "distance": "Cosine"}},
            )
        self._ready_dimensions.add(vector_dimension)

    def request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        timeout: int = 120,
    ) -> dict:
        if not self.enabled:
            raise RuntimeError("Qdrant vector backend is disabled.")

        request = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Qdrant HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Qdrant is not reachable at {self.url}. Start Qdrant and check QDRANT_URL.") from exc
        return json.loads(body) if body else {}

    @staticmethod
    def payload_filter(**values) -> dict:
        return {
            "must": [
                {"key": key, "match": {"value": value}}
                for key, value in values.items()
                if value is not None
            ]
        }

    def point_from_entry(self, entry: dict) -> dict | None:
        keywords = self.clean_keywords(entry.get("keywords") or [])
        information = normalize_document_text(str(entry.get("information") or ""))
        if not information:
            return None

        document_id = int(entry["document_id"])
        group_index = int(entry.get("group_index") or 0)
        entry_index = int(entry.get("entry_index") or 0)
        source_chunk_start = int(entry.get("source_chunk_start") or 0)
        source_chunk_end = int(entry.get("source_chunk_end") or source_chunk_start)
        document_name = str(entry.get("document_name") or "Unknown")
        source = normalize_text(str(entry.get("source") or ""))
        vector_text = "Search keywords: " + ", ".join(keywords)
        vector = self.embed_text(vector_text if keywords else information)

        return {
            "id": self.make_point_id(document_id, group_index, entry_index, information),
            "vector": list(vector),
            "payload": {
                "document_id": document_id,
                "document_name": document_name,
                "document_kind": self.document_kind,
                "entry_index": entry_index,
                "source_chunk_start": source_chunk_start,
                "source_chunk_end": source_chunk_end,
                "source": source,
                "keywords": keywords,
                "information": information,
                "vector_model": self.embedding_model,
                "prompt_version": self.prompt_version,
            },
        }

    def chunk_from_search_point(self, item: dict, query: str, query_tokens: set[str]) -> Chunk:
        payload = item.get("payload") or {}
        searchable_text = f"{payload.get('document_name', '')} {self.document_kind} {payload.get('source', '')} {payload.get('information', '')}"
        vector_score = float(item.get("score") or 0.0)
        lexical_score = token_overlap_score(query_tokens, searchable_text)
        score = ((0.72 * vector_score) + (0.22 * lexical_score) + exact_match_bonus(query, searchable_text)) * 1.08
        return Chunk(
            id=str(item.get("id") or payload.get("chunk_id") or ""),
            document_id=int(payload.get("document_id") or 0),
            document_name=str(payload.get("document_name") or "Unknown"),
            document_kind=self.document_kind,
            chunk_index=int(payload.get("source_chunk_start") or 0),
            text=str(payload.get("information") or ""),
            score=score,
        )

    @staticmethod
    def entry_from_point(point: dict) -> dict:
        payload = point.get("payload") or {}
        return {
            "id": str(point.get("id") or ""),
            "document_id": int(payload.get("document_id") or 0),
            "document_name": str(payload.get("document_name") or "Unknown"),
            "document_kind": str(payload.get("document_kind") or "knowledge"),
            "chunk_index": int(payload.get("source_chunk_start") or 0),
            "source_chunk_start": int(payload.get("source_chunk_start") or 0),
            "source_chunk_end": int(payload.get("source_chunk_end") or 0),
            "source": str(payload.get("source") or ""),
            "keywords": payload.get("keywords") or [],
            "information": str(payload.get("information") or ""),
            "vector_model": str(payload.get("vector_model") or ""),
            "prompt_version": str(payload.get("prompt_version") or ""),
        }

    @staticmethod
    def clean_keywords(raw_keywords) -> list[str]:
        if isinstance(raw_keywords, str):
            raw_keywords = [raw_keywords]
        return [
            normalize_text(str(keyword))
            for keyword in raw_keywords
            if normalize_text(str(keyword))
        ]

    @staticmethod
    def point_id(document_id: int, group_index: int, entry_index: int, information: str) -> str:
        digest = stable_hash(f"{document_id}:{group_index}:{entry_index}:{information}")[:16]
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"nimbus:knowledge:{document_id}:{group_index}:{entry_index}:{digest}"))

    def make_point_id(self, document_id: int, group_index: int, entry_index: int, information: str) -> str:
        digest = stable_hash(f"{document_id}:{group_index}:{entry_index}:{information}")[:16]
        return str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"nimbus:{self.point_namespace}:{document_id}:{group_index}:{entry_index}:{digest}",
        ))
