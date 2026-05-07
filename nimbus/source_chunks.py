from nimbus.source_store import SourceBase
from nimbus.text_processing import keyword_candidates, semantic_chunks
from nimbus.vector_store import QdrantKnowledgeBase


class SourceChunkIndexer:
    def __init__(
        self,
        source_base: SourceBase,
        source_chunk_base: QdrantKnowledgeBase,
        max_words: int,
    ) -> None:
        self.source_base = source_base
        self.source_chunk_base = source_chunk_base
        self.max_words = max(120, max_words)

    def index_document(self, document_id: int, progress_callback=None) -> int:
        document = self.source_base.document(document_id)
        text = self.source_base.document_text(document_id)
        chunks = semantic_chunks(text, max_words=self.max_words)
        entries = []
        total = len(chunks)

        if progress_callback:
            progress_callback(0, total or 1, f"Preparing {total} Source Base retrieval chunks")

        for index, chunk in enumerate(chunks):
            entries.append(
                {
                    "document_id": int(document["id"]),
                    "document_name": document["name"],
                    "group_index": index,
                    "entry_index": 1,
                    "source_chunk_start": index,
                    "source_chunk_end": index,
                    "source": chunk["title"],
                    "keywords": keyword_candidates(chunk["text"], chunk["title"]),
                    "information": chunk["text"],
                }
            )

        self.source_chunk_base.delete_document(document_id)
        written = self.source_chunk_base.upsert(entries)
        self.source_base.update_document_chunk_count(document_id, written)

        if progress_callback:
            progress_callback(written, total or 1, f"Stored {written} Source Base retrieval chunks in Qdrant")
        return written

    def rebuild_all(self, progress_callback=None) -> list[int]:
        document_ids = self.source_base.document_ids_oldest_first()
        self.source_chunk_base.clear()
        counts = []
        total = len(document_ids)
        for index, document_id in enumerate(document_ids, start=1):
            if progress_callback:
                progress_callback(index - 1, total, f"Indexing Source Base retrieval chunks {index}/{total}")
            counts.append(self.index_document(document_id, progress_callback))
        return counts
