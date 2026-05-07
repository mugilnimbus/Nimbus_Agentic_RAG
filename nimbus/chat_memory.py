import threading

from nimbus.retrieval import extract_named_entities


class ChatMemory:
    def __init__(
        self,
        max_turns: int = 24,
        max_summary_chars: int = 700,
        max_message_chars: int = 4000,
    ) -> None:
        self.max_turns = max_turns
        self.max_summary_chars = max_summary_chars
        self.max_message_chars = max_message_chars
        self._lock = threading.Lock()
        self._turns: list[dict[str, str]] = []

    @classmethod
    def from_turns(
        cls,
        turns: list[dict[str, str]],
        max_turns: int = 24,
        max_summary_chars: int = 700,
        max_message_chars: int = 4000,
    ):
        memory = cls(max_turns, max_summary_chars, max_message_chars)
        with memory._lock:
            memory._turns = list(turns[-max_turns:])
        return memory

    def as_prompt_text(self) -> str:
        with self._lock:
            turns = list(self._turns[-self.max_turns :])
        if not turns:
            return ""

        parts = []
        for index, turn in enumerate(turns, start=1):
            question = self.compact(turn.get("question", ""), self.max_summary_chars)
            answer = self.compact(turn.get("answer", ""), self.max_summary_chars)
            focus = turn.get("focus_entities", "")
            focus_line = f"\n   Focus entities: {focus}" if focus else ""
            parts.append(f"{index}. User: {question}\n   Assistant: {answer}{focus_line}")
        return "\n".join(parts)

    def as_messages(self) -> list[dict[str, str]]:
        with self._lock:
            turns = list(self._turns[-self.max_turns :])

        messages = []
        for turn in turns:
            question = self.compact(turn.get("question", ""), self.max_message_chars)
            answer = self.compact(turn.get("answer", ""), self.max_message_chars)
            if question:
                messages.append({"role": "user", "content": question})
            if answer:
                messages.append({"role": "assistant", "content": answer})
        return messages

    def remember(
        self,
        question: str,
        answer: str,
        focus_entities: list[str] | None = None,
        sources: list[dict] | None = None,
    ) -> None:
        focus = self.focus_entities(question, answer, focus_entities or [], sources or [])
        turn = {
            "question": str(question or ""),
            "answer": str(answer or ""),
            "focus_entities": ", ".join(focus),
        }
        with self._lock:
            self._turns.append(turn)
            del self._turns[:-self.max_turns]

    def focus_entities(
        self,
        question: str,
        answer: str,
        focus_entities: list[str],
        sources: list[dict],
    ) -> list[str]:
        if focus_entities:
            return self.unique_entities([str(entity) for entity in focus_entities if entity])

        question_entities = {entity.lower() for entity in extract_named_entities(question)}
        answer_entities = extract_named_entities(answer)
        answer_lower = answer.lower()
        answer_reports_missing = (
            "could not find" in answer_lower
            or "couldn't find" in answer_lower
            or "not find" in answer_lower
            or "only" in answer_lower
        )
        if answer_entities:
            if answer_reports_missing:
                answer_entities = [
                    entity
                    for entity in answer_entities
                    if entity.lower() not in question_entities
                ]
            return self.unique_entities(answer_entities)

        source_entities: list[str] = []
        for source in sources[:3]:
            source_entities.extend(extract_named_entities(str(source.get("text") or "")))
            source_entities.extend(extract_named_entities(str(source.get("document_name") or "")))
        if source_entities and not answer_reports_missing:
            return self.unique_entities(source_entities)
        return self.unique_entities(extract_named_entities(question))

    def unique_entities(self, candidates: list[str]) -> list[str]:
        seen = set()
        result = []
        for entity in candidates:
            clean = self.compact(entity, self.max_summary_chars).strip(" .,:;()[]")
            key = clean.lower()
            if not clean or key in seen:
                continue
            seen.add(key)
            result.append(clean)
            if len(result) >= 8:
                break
        return result

    @staticmethod
    def compact(value: str, max_chars: int) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars].rstrip()}..."
