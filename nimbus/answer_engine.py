import json
from collections.abc import Callable, Sequence

from nimbus import prompts
from nimbus.models import Chunk
from nimbus.retrieval import (
    filter_hits_by_focus_entities,
    focused_followup_queries,
    focus_entities_for_followup,
    format_context,
    is_followup_question,
    merge_hits,
    parse_rerank_numbers,
)
from nimbus.text_processing import normalize_text


class AnswerEngine:
    def __init__(
        self,
        llm,
        search_knowledge: Callable[[str, int], list[Chunk]],
        search_source: Callable[[str, int], list[Chunk]],
        rerank_enabled: bool,
    ) -> None:
        self.llm = llm
        self.search_knowledge = search_knowledge
        self.search_source = search_source
        self.rerank_enabled = rerank_enabled

    def answer(
        self,
        question: str,
        top_k: int = 6,
        chat_memory: str = "",
        conversation_messages: Sequence[dict] | None = None,
    ) -> dict:
        memory_text = chat_memory or "No previous conversation."
        retrieval_query = self.rewrite_query(question, memory_text)
        retrieval_k = max(top_k * 3, 12)
        followup = is_followup_question(question) and bool(chat_memory)
        needs_reasoning = self.is_reasoning_question(question)
        focus_entities = focus_entities_for_followup(retrieval_query, memory_text) if followup else []
        search_queries = self.search_queries(question, retrieval_query, memory_text, focus_entities, followup)

        knowledge_hits = self.knowledge_hits(question, retrieval_query, retrieval_k, search_queries, followup)
        raw_hits = self.raw_hits(retrieval_query, top_k, search_queries, followup, focus_entities)
        hits, search_mode = self.choose_context_hits(knowledge_hits, raw_hits, retrieval_k, top_k)

        if followup and focus_entities:
            hits = filter_hits_by_focus_entities(hits, focus_entities)
        rerank_question = f"{retrieval_query}\nOriginal question: {question}" if followup else question
        candidate_hits = list(hits)
        hits = self.rerank_hits(rerank_question, hits, top_k, chat_memory=memory_text) if hits else []
        if needs_reasoning and not hits and candidate_hits:
            hits = candidate_hits[:top_k]
        answer_text = self.write_answer(
            question,
            retrieval_query,
            search_mode,
            memory_text,
            hits,
            conversation_messages=conversation_messages or [],
        )

        return {
            "answer": answer_text,
            "retrieval_query": retrieval_query,
            "search_mode": search_mode,
            "memory_used": bool(chat_memory or conversation_messages),
            "focus_entities": focus_entities,
            "sources": [self.source_payload(hit) for hit in hits],
        }

    def rewrite_query(self, question: str, chat_memory: str = "") -> str:
        messages = [
            {
                "role": "user",
                "content": prompts.QUERY_REWRITE_TEMPLATE.format(
                    question=question,
                    chat_memory=chat_memory or "No previous conversation.",
                ),
            }
        ]
        rewritten = normalize_text(self.llm.chat(messages, max_tokens=2000))
        return rewritten[:1200] if rewritten else question

    def rerank_hits(
        self,
        question: str,
        hits: Sequence[Chunk],
        top_k: int,
        chat_memory: str = "",
    ) -> list[Chunk]:
        if not self.rerank_enabled:
            return list(hits[:top_k])

        compact = "\n\n".join(
            f"{index}. {hit.document_kind.upper()} | {hit.document_name} | chunk {hit.chunk_index + 1} | score {hit.score:.3f}\n"
            f"{hit.text[:900]}"
            for index, hit in enumerate(hits[:24], start=1)
        )
        messages = [
            {"role": "system", "content": prompts.RERANK_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompts.RERANK_USER_TEMPLATE.format(
                    question=question,
                    chat_memory=chat_memory or "No previous conversation.",
                    compact=compact,
                    top_k=top_k,
                ),
            },
        ]
        try:
            numbers = parse_rerank_numbers(self.llm.chat(messages, max_tokens=800))
        except Exception:
            return list(hits[:top_k])

        selected = []
        seen_indexes = set()
        for number in numbers:
            index = number - 1
            if 0 <= index < len(hits) and index not in seen_indexes:
                selected.append(hits[index])
                seen_indexes.add(index)
            if len(selected) >= top_k:
                break
        return selected

    def write_answer(
        self,
        question: str,
        retrieval_query: str,
        search_mode: str,
        chat_memory: str,
        hits: Sequence[Chunk],
        conversation_messages: Sequence[dict] | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": prompts.ANSWER_SYSTEM_PROMPT},
            *self.safe_conversation_messages(conversation_messages or []),
            {
                "role": "user",
                "content": prompts.ANSWER_USER_TEMPLATE.format(
                    search_mode=search_mode,
                    retrieval_query=retrieval_query,
                    chat_memory=chat_memory,
                    context=format_context(hits) or "No matching context was found.",
                    question=question,
                ),
            },
        ]
        answer = self.llm.chat(messages, max_tokens=4000)
        if answer:
            return answer
        return "I found relevant sources, but the model returned an empty answer. Try asking again or increase the chat model output limit."

    @staticmethod
    def safe_conversation_messages(messages: Sequence[dict]) -> list[dict[str, str]]:
        safe_messages = []
        for message in messages:
            role = message.get("role")
            content = str(message.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            safe_messages.append({"role": role, "content": content})
        return safe_messages

    def knowledge_hits(
        self,
        question: str,
        retrieval_query: str,
        retrieval_k: int,
        search_queries: Sequence[str],
        followup: bool,
    ) -> list[Chunk]:
        searches = [self.search_knowledge(query, retrieval_k) for query in search_queries]
        if not followup:
            searches.append(self.search_knowledge(question, retrieval_k))
        return merge_hits(*searches)

    def raw_hits(
        self,
        retrieval_query: str,
        top_k: int,
        search_queries: Sequence[str],
        followup: bool,
        focus_entities: Sequence[str],
    ) -> list[Chunk]:
        if followup and focus_entities:
            return merge_hits(
                *[self.search_source(query, max(4, top_k)) for query in search_queries]
            )
        return merge_hits(
            *[self.search_source(query, max(4, top_k)) for query in search_queries]
        )

    @staticmethod
    def choose_context_hits(
        knowledge_hits: Sequence[Chunk],
        raw_hits: Sequence[Chunk],
        retrieval_k: int,
        top_k: int,
    ) -> tuple[list[Chunk], str]:
        useful_knowledge = [hit for hit in knowledge_hits if hit.score >= 0.05]
        useful_sources = [hit for hit in raw_hits if hit.score >= 0.05]
        if useful_knowledge and useful_sources:
            source_limit = max(top_k, retrieval_k // 2)
            hits = merge_hits(useful_knowledge[:retrieval_k], useful_sources[:source_limit])
            return hits, "knowledge-and-source"
        if useful_knowledge:
            return list(useful_knowledge[:retrieval_k]), "knowledge-base"
        return list(useful_sources[:retrieval_k]), "source-base"

    def search_queries(
        self,
        question: str,
        retrieval_query: str,
        chat_memory: str,
        focus_entities: Sequence[str],
        followup: bool,
    ) -> list[str]:
        if followup and focus_entities:
            return focused_followup_queries(question, retrieval_query, focus_entities)
        queries = [retrieval_query, question]
        queries.extend(self.expanded_queries(question, retrieval_query, chat_memory))
        return self.unique_queries(queries)

    def expanded_queries(self, question: str, retrieval_query: str, chat_memory: str) -> list[str]:
        messages = [
            {
                "role": "user",
                "content": prompts.QUERY_EXPANSION_TEMPLATE.format(
                    question=question,
                    retrieval_query=retrieval_query,
                    chat_memory=chat_memory or "No previous conversation.",
                ),
            }
        ]
        try:
            raw = self.llm.chat(messages, max_tokens=1200)
        except Exception:
            return []
        return self.parse_query_list(raw)[:5]

    @staticmethod
    def parse_query_list(text: str) -> list[str]:
        raw = normalize_text(text)
        start = raw.find("[")
        end = raw.rfind("]")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        return [
            line.strip(" -\t")
            for line in str(text or "").splitlines()
            if line.strip(" -\t")
        ]

    @staticmethod
    def is_reasoning_question(question: str) -> bool:
        text = normalize_text(question).lower()
        markers = (
            "can i",
            "can we",
            "what can",
            "which",
            "recommend",
            "best",
            "run on",
            "fit",
            "enough",
            "how many",
            "how much",
            "estimate",
            "possible",
            "should",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def unique_queries(queries: Sequence[str]) -> list[str]:
        unique = []
        seen = set()
        for query in queries:
            clean = normalize_text(query)
            key = clean.lower()
            if clean and key not in seen:
                unique.append(clean)
                seen.add(key)
        return unique

    @staticmethod
    def source_payload(hit: Chunk) -> dict:
        return {
            "id": hit.id,
            "document_id": hit.document_id,
            "document_name": hit.document_name,
            "document_kind": hit.document_kind,
            "chunk_index": hit.chunk_index,
            "score": round(hit.score, 4),
            "text": hit.text,
        }
