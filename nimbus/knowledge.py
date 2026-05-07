from concurrent.futures import ThreadPoolExecutor, as_completed

from nimbus import prompts
from nimbus.knowledge_parser import parse_knowledge_entries
from nimbus.source_store import SourceBase
from nimbus.text_processing import (
    compact_structured_text,
    is_dense_structured_text,
    keyword_candidates,
    normalize_text,
    semantic_chunks,
)
from nimbus.vector_store import QdrantKnowledgeBase


class KnowledgeBuilder:
    def __init__(
        self,
        source_base: SourceBase,
        knowledge_base: QdrantKnowledgeBase,
        llm,
        group_size: int,
        max_tokens: int,
        concurrency: int,
    ) -> None:
        self.source_base = source_base
        self.knowledge_base = knowledge_base
        self.llm = llm
        self.group_size = max(1, group_size)
        self.max_tokens = max_tokens
        self.concurrency = max(1, min(8, concurrency))

    def build_for_document(self, document_id: int, progress_callback=None) -> int:
        document = self.source_base.document(document_id)
        groups = self.chunk_groups(document_id)
        entries = self.entries_from_groups(document, groups, progress_callback)
        if not entries:
            raise ValueError("No Knowledge Base entries were produced.")

        self.knowledge_base.delete_document(document_id)
        if progress_callback:
            progress_callback(0, len(entries), f"Embedding Knowledge Base entries for {document['name']}")
        written = self.knowledge_base.upsert(entries)
        if progress_callback:
            progress_callback(written, len(entries), f"Stored {written} Knowledge Base entries")
        return written

    def rebuild_all(self, progress_callback=None) -> list[int]:
        document_ids = self.source_base.document_ids_oldest_first()
        self.knowledge_base.clear()
        rebuilt = []
        total = len(document_ids)

        for index, document_id in enumerate(document_ids, start=1):
            if progress_callback:
                progress_callback(index - 1, total, f"Building Knowledge Base for Source Base document {index}/{total}")
            rebuilt.append(self.build_for_document(document_id))
            if progress_callback:
                progress_callback(index, total, f"Built Knowledge Base for Source Base document {index}/{total}")
        return rebuilt

    def chunk_groups(self, document_id: int) -> list[dict]:
        text = self.source_base.document_text(document_id)
        max_words = max(600, min(9000, int(self.max_tokens * 0.55)))
        if len(text.split()) <= max_words and is_dense_structured_text(text):
            return [
                {
                    "index": 0,
                    "label": "dense structured document",
                    "text": text,
                    "source_chunk_start": 0,
                    "source_chunk_end": 0,
                }
            ]
        chunks = semantic_chunks(text, max_words=max_words, min_words=180)
        chunks = self.pack_semantic_sections(chunks, max_words=max_words)
        groups = []
        for index, chunk in enumerate(chunks):
            label = chunk["title"] or f"section {index + 1}"
            groups.append(
                {
                    "index": index,
                    "label": label,
                    "text": chunk["text"],
                    "source_chunk_start": index,
                    "source_chunk_end": index,
                }
            )
        return groups

    @staticmethod
    def pack_semantic_sections(chunks: list[dict], max_words: int) -> list[dict]:
        packed = []
        current_titles = []
        current_texts = []
        current_words = 0

        def flush() -> None:
            nonlocal current_titles, current_texts, current_words
            if not current_texts:
                return
            title = " / ".join(current_titles[:4])
            if len(current_titles) > 4:
                title = f"{title} / {len(current_titles) - 4} more sections"
            packed.append({"title": title or "Document section", "text": "\n\n".join(current_texts)})
            current_titles = []
            current_texts = []
            current_words = 0

        for chunk in chunks:
            text = str(chunk.get("text") or "")
            words = len(text.split())
            if current_texts and current_words + words > max_words:
                flush()
            current_titles.append(str(chunk.get("title") or "Section"))
            current_texts.append(text)
            current_words += words
        flush()
        return packed

    def entries_from_groups(self, document: dict, groups: list[dict], progress_callback=None) -> list[dict]:
        entries_by_group: dict[int, list[dict]] = {}
        total = len(groups)
        llm_groups = []

        for group in groups:
            if self.should_preserve_dense_group(group):
                entries = [self.dense_group_entry(document, group)]
                self.attach_source_metadata(entries, document, group)
                entries_by_group[group["index"]] = entries
            else:
                llm_groups.append(group)

        workers = min(self.concurrency, len(llm_groups) or 1)

        if progress_callback:
            progress_callback(0, total, f"Building Knowledge Base group 0/{total}")

        completed = len(entries_by_group)
        if llm_groups:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self.build_group_entries, document["name"], group["label"], group["text"]): group
                    for group in llm_groups
                }
                for future in as_completed(futures):
                    group = futures[future]
                    entries = self.safe_group_result(future, document["name"], group)
                    self.attach_source_metadata(entries, document, group)
                    entries_by_group[group["index"]] = entries
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total, f"Built Knowledge Base group {completed}/{total}")
        elif progress_callback:
            progress_callback(total, total, f"Preserved {total} dense Knowledge Base group{'s' if total != 1 else ''}")

        ordered_entries = []
        for group_index in sorted(entries_by_group):
            for entry_index, entry in enumerate(entries_by_group[group_index], start=1):
                entry["entry_index"] = entry_index
                ordered_entries.append(entry)
        return ordered_entries

    @staticmethod
    def should_preserve_dense_group(group: dict) -> bool:
        text = str(group.get("text") or "")
        word_count = len(text.split())
        return word_count <= 2500 and is_dense_structured_text(text)

    @staticmethod
    def dense_group_entry(document: dict, group: dict) -> dict:
        compact = compact_structured_text(str(group.get("text") or ""))
        keywords = keyword_candidates(compact, f"{document['name']} {group['label']}", limit=60)
        return {
            "keywords": keywords,
            "information": compact,
            "source": group["label"],
        }

    def build_group_entries(self, document_name: str, chunk_label: str, text: str) -> list[dict]:
        messages = [
            {"role": "system", "content": prompts.KNOWLEDGE_BUILD_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompts.KNOWLEDGE_BUILD_USER_TEMPLATE.format(
                    document_name=document_name,
                    chunk_label=chunk_label,
                    text=text,
                ),
            },
        ]
        knowledge_json = self.llm.chat(messages, max_tokens=self.max_tokens).strip()
        if knowledge_json.upper() == "NO_USEFUL_FACTS":
            return []
        return parse_knowledge_entries(knowledge_json, default_source=chunk_label)

    @staticmethod
    def safe_group_result(future, document_name: str, group: dict) -> list[dict]:
        try:
            entries = future.result()
        except Exception as exc:
            return [
                {
                    "keywords": ["knowledge build failed", document_name, group["label"]],
                    "information": (
                        f"Knowledge Base build failed for {group['label']}: {exc}. "
                        f"Extracted text needing review: {normalize_text(group['text'])[:5000]}"
                    ),
                    "source": group["label"],
                }
            ]
        if entries:
            return entries
        return [
            {
                "keywords": ["extracted text needing review", document_name, group["label"]],
                "information": normalize_text(group["text"])[:5000],
                "source": group["label"],
            }
        ]

    @staticmethod
    def attach_source_metadata(entries: list[dict], document: dict, group: dict) -> None:
        for entry in entries:
            entry["document_id"] = int(document["id"])
            entry["document_name"] = document["name"]
            entry["group_index"] = int(group["index"])
            entry["source_chunk_start"] = int(group["source_chunk_start"])
            entry["source_chunk_end"] = int(group["source_chunk_end"])
            entry.setdefault("source", group["label"])
