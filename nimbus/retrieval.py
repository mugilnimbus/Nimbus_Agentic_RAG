import json
import re
from typing import Sequence

from nimbus.models import Chunk
from nimbus.text_processing import RETRIEVAL_STOPWORDS, TOKEN_RE, normalize_text


GENERIC_ENTITY_WORDS = {
    "assistant",
    "base",
    "chunk",
    "context",
    "documents",
    "entries",
    "information",
    "knowledge",
    "question",
    "source",
    "sources",
    "user",
}


def tokens_for_scoring(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in RETRIEVAL_STOPWORDS
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
    pattern = r"[A-Za-z]+[-_/]?[A-Za-z0-9.]*\d+[A-Za-z0-9._/-]*|\d+(?:\.\d+)?\s?(?:gb|mb|mhz|ghz|w|eur|€|\$)"
    for item in re.findall(pattern, query_clean):
        if item and item in text_clean:
            bonus += 0.04
    return min(0.16, bonus)


def merge_hits(*hit_groups: Sequence[Chunk]) -> list[Chunk]:
    seen = set()
    merged = []
    for group in hit_groups:
        for hit in group:
            key = (hit.document_kind, hit.document_id, hit.chunk_index, hit.id)
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)
    return merged


def compatible_confirmation_hits(raw_hits: Sequence[Chunk], knowledge_hits: Sequence[Chunk]) -> list[Chunk]:
    if not raw_hits or not knowledge_hits:
        return []
    knowledge_documents = {hit.document_id for hit in knowledge_hits if hit.document_id}
    knowledge_tokens = set()
    for hit in knowledge_hits:
        knowledge_tokens.update(tokens_for_scoring(f"{hit.document_name} {hit.text}"))
    compatible = []
    for hit in raw_hits:
        if knowledge_documents and hit.document_id not in knowledge_documents:
            continue
        raw_tokens = set(tokens_for_scoring(f"{hit.document_name} {hit.text}"))
        if not raw_tokens or not knowledge_tokens:
            continue
        overlap = len(raw_tokens & knowledge_tokens) / max(1, min(len(raw_tokens), len(knowledge_tokens)))
        if overlap >= 0.08:
            compatible.append(hit)
    return compatible


def is_followup_question(question: str) -> bool:
    text = f" {normalize_text(question).lower()} "
    followup_terms = (
        " these ",
        " those ",
        " them ",
        " they ",
        " it ",
        " that ",
        " this ",
        " above ",
        " previous ",
        " earlier ",
        " same ",
        " one ",
        " ones ",
    )
    return any(term in text for term in followup_terms)


def focus_entities_for_followup(retrieval_query: str, chat_memory: str) -> list[str]:
    entities = explicit_memory_focus_entities(chat_memory)
    if entities:
        return entities[:8]
    entities = extract_named_entities(chat_memory)
    if entities:
        return entities[:8]
    return extract_named_entities(retrieval_query)[:8]


def focused_followup_queries(
    question: str,
    retrieval_query: str,
    focus_entities: Sequence[str],
) -> list[str]:
    property_terms = focused_followup_property_terms(question, retrieval_query, focus_entities)
    queries = []
    seen = set()
    for entity in focus_entities:
        clean_entity = normalize_text(entity)
        if not clean_entity:
            continue
        query = normalize_text(f"{clean_entity} {property_terms}")
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
    return queries or [retrieval_query]


def focused_followup_property_terms(
    question: str,
    retrieval_query: str,
    focus_entities: Sequence[str] | None = None,
) -> str:
    combined = normalize_text(f"{question} {retrieval_query}")
    focus_tokens = {
        token.lower()
        for entity in (focus_entities or [])
        for token in TOKEN_RE.findall(entity)
    }
    tokens = [
        token
        for token in TOKEN_RE.findall(combined)
        if len(token) > 1
        and token.lower() not in RETRIEVAL_STOPWORDS
        and token.lower() not in focus_tokens
    ]
    seen = set()
    kept = []
    for token in tokens:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(token)
    return " ".join(kept[:12])


def explicit_memory_focus_entities(chat_memory: str) -> list[str]:
    matches = re.findall(r"Focus entities:\s*([^\n]+)", chat_memory or "", flags=re.I)
    if not matches:
        return []
    latest = matches[-1]
    entities = []
    seen = set()
    for part in re.split(r"[,;|]", latest):
        entity = normalize_text(part).strip(" .,:;()[]")
        if not entity:
            continue
        words = [word.lower() for word in TOKEN_RE.findall(entity)]
        if not words or all(word in GENERIC_ENTITY_WORDS for word in words):
            continue
        key = entity.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(entity)
    return entities


def extract_named_entities(text: str) -> list[str]:
    patterns = (
        r"['\"]([^'\"]{3,80})['\"]",
        r"\b[A-Z]{2,}[A-Za-z]*-\d+[A-Za-z0-9]*(?:\s+[A-Za-z]+)?\b",
        r"\b[A-Z][A-Za-z]+-\d+(?:\.\d+)?[A-Za-z]*(?:\s+[A-Z][A-Za-z]+)?\b",
        r"\b[A-Z][A-Za-z]+\s+\d+(?:\.\d+)?(?:x\d+)?[A-Za-z]*(?:\s+\d+[A-Za-z]*)?\b",
        r"\b[A-Z]{2,}[A-Za-z]*\d+[A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*(?:\s+[A-Z][A-Za-z]+)?\b",
        r"\b[A-Z][A-Za-z0-9._/-]*(?:\s+[A-Z][A-Za-z0-9._/-]*){1,5}\b",
        r"\b(?:[A-Za-z]:)?[\\/][^\s,;:]{3,80}\b",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    )
    candidates = []
    for pattern in patterns:
        candidates.extend(re.findall(pattern, text or "", flags=re.I))
    entities = []
    seen = set()
    for candidate in candidates:
        entity = normalize_text(candidate).strip(" .,:;()[]")
        if not entity or len(entity) < 4 or len(entity) > 60:
            continue
        words = [word.lower() for word in TOKEN_RE.findall(entity)]
        if not words or all(word in GENERIC_ENTITY_WORDS for word in words):
            continue
        key = entity.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(entity)
    return entities


def filter_hits_by_focus_entities(hits: Sequence[Chunk], entities: Sequence[str]) -> list[Chunk]:
    if not entities:
        return list(hits)
    filtered = []
    for hit in hits:
        haystack = f"{hit.document_name}\n{hit.text}".lower()
        if any(entity_matches_text(entity, haystack) for entity in entities):
            filtered.append(hit)
    return filtered


def entity_matches_text(entity: str, text: str) -> bool:
    entity_text = normalize_text(entity).lower()
    if entity_text and entity_text in text:
        return True
    tokens = [token.lower() for token in TOKEN_RE.findall(entity) if token.lower() not in GENERIC_ENTITY_WORDS]
    if not tokens:
        return False
    return all(token in text for token in tokens)


def parse_rerank_numbers(text: str) -> list[int]:
    raw = normalize_text(text)
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            numbers = []
            for item in parsed:
                try:
                    numbers.append(int(item))
                except (TypeError, ValueError):
                    continue
            return numbers
    if re.search(r"\b(no|none|nothing|irrelevant|unrelated)\b", raw, flags=re.I):
        return []
    return [int(value) for value in re.findall(r"\b\d+\b", raw)]


def format_context(chunks: Sequence[Chunk]) -> str:
    parts = []
    for idx, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{idx}] {chunk.document_kind.upper()} | {chunk.document_name} | chunk {chunk.chunk_index + 1} | "
            f"score {chunk.score:.3f}\n{chunk.text}"
        )
    return "\n\n".join(parts)
