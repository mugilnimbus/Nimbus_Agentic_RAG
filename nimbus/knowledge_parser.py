import json
import re

from nimbus.text_processing import normalize_text


def parse_knowledge_entries(text: str, default_source: str = "") -> list[dict]:
    raw = text.strip()
    if not raw:
        return []
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        loose_entries = loose_json_knowledge_entries(raw, default_source=default_source)
        if loose_entries:
            return loose_entries
        return markdown_knowledge_entries(text, default_source=default_source)
    if isinstance(parsed, dict):
        parsed = parsed.get("entries") or parsed.get("items") or [parsed]
    if not isinstance(parsed, list):
        return []
    return normalized_knowledge_entries(parsed, default_source)


def normalized_knowledge_entries(items: list, default_source: str) -> list[dict]:
    entries = []
    for item in items:
        if not isinstance(item, dict):
            continue
        keywords = item.get("keywords") or item.get("keyword") or item.get("keys") or []
        information = item.get("information") or item.get("info") or item.get("text") or item.get("value") or ""
        source = item.get("source") or default_source
        document = item.get("document") or item.get("document_name") or ""
        if isinstance(keywords, str):
            keywords = [part.strip() for part in re.split(r"[,;|]", keywords) if part.strip()]
        elif isinstance(keywords, list):
            keywords = [normalize_text(str(part)) for part in keywords if normalize_text(str(part))]
        else:
            keywords = []
        information = normalize_text(str(information))
        source = normalize_text(str(source))
        document = normalize_text(str(document))
        if information:
            entries.append(
                {
                    "keywords": keywords[:40],
                    "information": information,
                    "source": source,
                    "document": document,
                }
            )
    return entries


def loose_json_knowledge_entries(text: str, default_source: str = "") -> list[dict]:
    entries = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = text[start : index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        start = None
                        continue
                    if isinstance(parsed, dict):
                        entries.extend(parse_knowledge_entries(json.dumps([parsed]), default_source))
                    start = None
    return entries


def markdown_knowledge_entries(text: str, default_source: str = "") -> list[dict]:
    entries = []
    blocks = re.split(r"\n(?=###\s+)", text)
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        title = lines[0].strip("# ").strip() if lines[0].startswith("#") else ""
        keywords = []
        information = []
        source = default_source
        for line in lines[1:] if title else lines:
            lower = line.lower()
            value = line.split(":", 1)[1].strip() if ":" in line else line
            if lower.startswith("- keywords:") or lower.startswith("keywords:"):
                keywords.extend(part.strip() for part in re.split(r"[,;|]", value) if part.strip())
            elif lower.startswith("- information:") or lower.startswith("information:"):
                information.append(value)
            elif lower.startswith("- source:") or lower.startswith("source:"):
                source = value
            else:
                information.append(line.lstrip("- "))
        info = normalize_text(" ".join(information))
        if title:
            keywords.insert(0, title)
        if info:
            entries.append({"keywords": keywords[:40], "information": info, "source": source})
    return entries
