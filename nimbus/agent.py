import json
import os
from dataclasses import dataclass
from typing import Any

from nimbus import prompts
from nimbus.tools import AgentToolbox


@dataclass
class AgentStep:
    index: int
    tool: str
    arguments: dict[str, Any]
    ok: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "tool": self.tool,
            "arguments": self.arguments,
            "ok": self.ok,
            "summary": self.summary,
        }


class NimbusAgent:
    def __init__(self, rag, toolbox: AgentToolbox) -> None:
        self.rag = rag
        self.toolbox = toolbox
        self.max_steps = max(1, min(12, int(os.environ.get("AGENT_MAX_STEPS", "6"))))

    def answer(
        self,
        question: str,
        top_k: int = 6,
        chat_memory: str = "",
        conversation_messages: list[dict] | None = None,
    ) -> dict:
        observations: list[dict[str, Any]] = []
        steps: list[AgentStep] = []
        sources: list[dict[str, Any]] = []

        for step_index in range(1, self.max_steps + 1):
            decision = self.next_action(question, chat_memory, observations)
            if decision.get("action") == "final":
                break

            tool_name = str(decision.get("tool") or "").strip()
            arguments = decision.get("arguments") if isinstance(decision.get("arguments"), dict) else {}
            if not tool_name:
                observations.append({"tool": "agent", "ok": False, "content": "Agent returned no tool name."})
                continue

            result = self.toolbox.run(tool_name, arguments)
            observation = {
                "tool": tool_name,
                "arguments": arguments,
                "ok": bool(result.get("ok")),
                "content": str(result.get("content") or result.get("error") or ""),
            }
            observations.append(observation)
            for source in result.get("sources") or []:
                self.add_unique_source(sources, source)
            steps.append(
                AgentStep(
                    index=step_index,
                    tool=tool_name,
                    arguments=arguments,
                    ok=bool(result.get("ok")),
                    summary=self.summarize_observation(observation["content"]),
                )
            )

            if decision.get("action") not in {"tool", "continue"}:
                break

        if not sources:
            self.collect_default_evidence(question, top_k, observations, steps, sources)

        answer = self.final_answer(
            question=question,
            top_k=top_k,
            chat_memory=chat_memory,
            observations=observations,
            sources=sources,
            conversation_messages=conversation_messages or [],
        )
        return {
            "answer": answer,
            "sources": sources[: max(1, min(top_k, 20))],
            "agent_steps": [step.to_dict() for step in steps],
            "agentic": True,
            "memory_used": bool(chat_memory or conversation_messages),
        }

    def collect_default_evidence(
        self,
        question: str,
        top_k: int,
        observations: list[dict[str, Any]],
        steps: list[AgentStep],
        sources: list[dict[str, Any]],
    ) -> None:
        for tool_name in ("search_knowledge", "search_source"):
            result = self.toolbox.run(tool_name, {"query": question, "top_k": top_k})
            observation = {
                "tool": tool_name,
                "arguments": {"query": question, "top_k": top_k},
                "ok": bool(result.get("ok")),
                "content": str(result.get("content") or result.get("error") or ""),
            }
            observations.append(observation)
            for source in result.get("sources") or []:
                self.add_unique_source(sources, source)
            steps.append(
                AgentStep(
                    index=len(steps) + 1,
                    tool=tool_name,
                    arguments={"query": question, "top_k": top_k},
                    ok=bool(result.get("ok")),
                    summary=self.summarize_observation(observation["content"]),
                )
            )

    def next_action(self, question: str, chat_memory: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": prompts.AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompts.AGENT_DECISION_TEMPLATE.format(
                    question=question,
                    chat_memory=chat_memory or "No previous conversation.",
                    tools=json.dumps(self.toolbox.manifest(), ensure_ascii=True, indent=2),
                    observations=json.dumps(observations, ensure_ascii=True, indent=2),
                ),
            },
        ]
        raw = self.rag.chat(messages, max_tokens=1200)
        return self.parse_json_object(raw) or {"action": "final"}

    def final_answer(
        self,
        question: str,
        top_k: int,
        chat_memory: str,
        observations: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        conversation_messages: list[dict],
    ) -> str:
        evidence = self.format_evidence(sources[: max(1, min(top_k, 20))])
        messages = [
            {"role": "system", "content": prompts.AGENT_FINAL_SYSTEM_PROMPT},
            *self.safe_conversation_messages(conversation_messages),
            {
                "role": "user",
                "content": prompts.AGENT_FINAL_USER_TEMPLATE.format(
                    question=question,
                    chat_memory=chat_memory or "No previous conversation.",
                    observations=json.dumps(observations, ensure_ascii=True, indent=2),
                    evidence=evidence or "No tool evidence was collected.",
                ),
            },
        ]
        answer = self.rag.chat(messages, max_tokens=5000)
        if answer:
            return answer
        return "The agent completed its tool pass, but the model returned an empty final answer."

    @staticmethod
    def parse_json_object(text: str) -> dict[str, Any] | None:
        raw = str(text or "").strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def add_unique_source(sources: list[dict[str, Any]], source: dict[str, Any]) -> None:
        source_key = (
            str(source.get("id") or ""),
            int(source.get("document_id") or 0),
            str(source.get("document_kind") or ""),
            int(source.get("chunk_index") or 0),
        )
        for existing in sources:
            existing_key = (
                str(existing.get("id") or ""),
                int(existing.get("document_id") or 0),
                str(existing.get("document_kind") or ""),
                int(existing.get("chunk_index") or 0),
            )
            if existing_key == source_key:
                return
        sources.append(source)

    @staticmethod
    def summarize_observation(content: str) -> str:
        text = " ".join(str(content or "").split())
        return text[:220] + ("..." if len(text) > 220 else "")

    @staticmethod
    def format_evidence(sources: list[dict[str, Any]]) -> str:
        return "\n\n".join(
            f"[{index}] {source.get('document_kind', 'source')} | {source.get('document_name', 'Unknown')} | chunk {int(source.get('chunk_index') or 0) + 1}\n{source.get('text', '')}"
            for index, source in enumerate(sources, start=1)
        )

    @staticmethod
    def safe_conversation_messages(messages: list[dict]) -> list[dict[str, str]]:
        safe_messages = []
        for message in messages:
            role = message.get("role")
            content = str(message.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                safe_messages.append({"role": role, "content": content})
        return safe_messages
