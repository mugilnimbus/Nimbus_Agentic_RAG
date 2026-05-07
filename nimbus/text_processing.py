import hashlib
import re


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

ENGLISH_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "you", "are", "not",
    "can", "will", "your", "have", "has", "use", "using", "when", "page",
    "what", "where", "which", "into", "only", "must", "should", "first",
    "if", "then", "than", "same", "more", "information", "recommended", "is",
    "my", "me", "i", "a", "an", "of", "in", "on", "to", "there",
}

RETRIEVAL_STOPWORDS = ENGLISH_STOPWORDS | {
    "about", "above", "again", "all", "answer", "available", "based", "chat",
    "could", "data", "detail", "details", "describe", "document", "documents",
    "does", "entry", "entries", "find", "found", "give", "help", "indexed",
    "know", "list", "mention", "mentioned", "need", "okay",
    "please", "previous", "question", "reply", "result", "results", "said",
    "show", "source", "sources", "tell", "these", "those", "topic", "user",
}

NON_ENGLISH_HINTS = {
    "und", "der", "die", "das", "von", "mit", "nicht", "bitte", "oder",
    "unter", "unterstützt", "speicher", "anschluss", "steckplatz", "pour",
    "avec", "depuis", "dans", "vous", "veuillez", "connexion", "vitesse",
    "mémoire", "emplacement", "processeur", "sortie", "entrée",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", " ").replace("\ufeff", " ")).strip()


def normalize_document_text(text: str) -> str:
    clean = text.replace("\x00", " ").replace("\ufeff", " ")
    clean = clean.replace("\r\n", "\n").replace("\r", "\n")
    clean = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in clean.split("\n"))
    clean = re.sub(r"\n{4,}", "\n\n\n", clean)
    return clean.strip()


def english_only_text(text: str) -> str:
    blocks = re.split(r"\n{2,}|(?:Page\s+\d+\s*)", text)
    kept = [normalize_document_text(block) for block in blocks if looks_english_or_technical(block)]
    return "\n\n".join(block for block in kept if block)


def looks_english_or_technical(text: str) -> bool:
    clean = normalize_text(text)
    if len(clean) < 8:
        return False
    ascii_chars = sum(1 for char in clean if ord(char) < 128)
    ascii_ratio = ascii_chars / max(1, len(clean))
    tokens = TOKEN_RE.findall(clean.lower())
    if not tokens:
        return False
    stop_hits = sum(1 for token in tokens if token in ENGLISH_STOPWORDS)
    non_english_hits = sum(1 for token in tokens if token in NON_ENGLISH_HINTS)
    has_technical = bool(
        re.search(
            r"[A-Za-z]+[_/-]?\d+|\d+(\.\d+)?\s?(gb|mb|mhz|ghz|pcie|usb|sata|x\d+)",
            clean,
            re.I,
        )
    )
    if non_english_hits > stop_hits and non_english_hits >= 2:
        return False
    if ascii_ratio >= 0.95 and stop_hits >= 1:
        return True
    if ascii_ratio >= 0.9 and has_technical and non_english_hits == 0:
        return True
    return ascii_ratio >= 0.98 and len(tokens) < 40


def chunk_text(text: str, max_words: int = 260, overlap: int = 45) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks = []
    step = max(1, max_words - overlap)
    for start in range(0, len(words), step):
        piece = words[start : start + max_words]
        if piece:
            chunks.append(" ".join(piece))
        if start + max_words >= len(words):
            break
    return chunks


def semantic_chunks(text: str, max_words: int = 520, min_words: int = 80) -> list[dict]:
    clean = text.replace("\x00", " ").strip()
    if not clean:
        return []

    blocks = structural_blocks(clean)
    chunks = []
    current_title = ""
    current_blocks = []
    current_words = 0

    def flush() -> None:
        nonlocal current_blocks, current_words, current_title
        if not current_blocks:
            return
        chunk_text_value = normalize_text("\n\n".join(current_blocks))
        if chunk_text_value:
            chunks.append({"title": current_title or "Document section", "text": chunk_text_value})
        current_blocks = []
        current_words = 0

    for block in blocks:
        block_text = block["text"]
        block_words = len(block_text.split())
        if block["kind"] == "heading":
            if current_words >= min_words:
                flush()
            current_title = block_text[:160]
            current_blocks.append(block_text)
            current_words += block_words
            continue
        if block["kind"] in {"table", "code"} and current_blocks and current_words + block_words > max_words:
            flush()
        elif current_blocks and current_words >= min_words and current_words + block_words > max_words:
            flush()
        current_blocks.append(block_text)
        current_words += block_words

    flush()
    return split_large_semantic_chunks(chunks, max_words=max_words)


def structural_blocks(text: str) -> list[dict]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    blocks = []
    in_code = False
    code_lines = []

    for paragraph in paragraphs:
        if paragraph.strip().startswith("```"):
            in_code = not in_code
            code_lines.append(paragraph)
            if not in_code:
                blocks.append({"kind": "code", "text": "\n\n".join(code_lines)})
                code_lines = []
            continue
        if in_code:
            code_lines.append(paragraph)
            continue

        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        kind = block_kind(lines, paragraph)
        blocks.append({"kind": kind, "text": paragraph})

    if code_lines:
        blocks.append({"kind": "code", "text": "\n\n".join(code_lines)})
    return blocks


def block_kind(lines: list[str], paragraph: str) -> str:
    first = lines[0]
    if looks_like_heading(first, len(lines)):
        return "heading"
    table_rows = sum(1 for line in lines if "|" in line or re.search(r"\S\s{2,}\S", line))
    if len(lines) >= 2 and table_rows >= max(2, len(lines) // 2):
        return "table"
    code_rows = sum(1 for line in lines if re.match(r"\s{2,}|\t|[{}[\]();]|(?:def|class|function|import|SELECT|CREATE)\b", line))
    if len(lines) >= 2 and code_rows >= max(2, len(lines) // 2):
        return "code"
    if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", first) and len(lines) > 1:
        return "list"
    return "paragraph"


def looks_like_heading(line: str, line_count: int) -> bool:
    clean = line.strip().strip("#").strip()
    if not clean or len(clean) > 140:
        return False
    if re.match(r"^(?:#{1,6}\s+|chapter\s+\d+|section\s+\d+|\d+(?:\.\d+)*[.)]?\s+\S)", line, flags=re.I):
        return True
    words = clean.split()
    if line_count == 1 and 1 <= len(words) <= 12:
        if not re.search(r"[,|:;\t]", clean) and not clean.endswith(".") and clean[0].isupper():
            return True
        letters = [char for char in clean if char.isalpha()]
        if letters and sum(1 for char in letters if char.isupper()) / len(letters) >= 0.45:
            return True
        if clean.istitle():
            return True
    return False


def split_large_semantic_chunks(chunks: list[dict], max_words: int) -> list[dict]:
    result = []
    for chunk in chunks:
        words = chunk["text"].split()
        if len(words) <= max_words:
            result.append(chunk)
            continue
        for index, piece in enumerate(chunk_text(chunk["text"], max_words=max_words, overlap=max(30, max_words // 10))):
            result.append({"title": f"{chunk['title']} part {index + 1}", "text": piece})
    return result


def keyword_candidates(text: str, title: str = "", limit: int = 28) -> list[str]:
    candidates = []
    for value in (title, text[:3000]):
        candidates.extend(re.findall(r"['\"]([^'\"]{3,80})['\"]", value))
        candidates.extend(re.findall(r"\b[A-Z][A-Za-z0-9._/-]*(?:\s+[A-Z][A-Za-z0-9._/-]*){0,5}\b", value))
        candidates.extend(re.findall(r"\b[A-Za-z]+[-_/]?[A-Za-z0-9.]*\d+[A-Za-z0-9._/-]*\b", value))
        candidates.extend(re.findall(r"\b\d+(?:\.\d+)?\s?(?:gb|mb|tb|mhz|ghz|w|v|a|mm|cm|kg|tokens?)\b", value, flags=re.I))
    candidates.extend(tokens_for_keywords(text))

    keywords = []
    seen = set()
    for candidate in candidates:
        clean = normalize_text(str(candidate)).strip(" .,:;()[]")
        if len(clean) < 2:
            continue
        key = clean.lower()
        if key in seen or key in RETRIEVAL_STOPWORDS:
            continue
        seen.add(key)
        keywords.append(clean)
        if len(keywords) >= limit:
            break
    return keywords


def is_dense_structured_text(text: str) -> bool:
    lines = meaningful_lines(text)
    if len(lines) < 8:
        return False

    key_value_rows = sum(1 for line in lines if key_value_parts(line) is not None)
    table_rows = sum(1 for line in lines if "|" in line or "\t" in line or re.search(r"\S\s{2,}\S", line))
    short_rows = sum(1 for line in lines if 2 <= len(line.split()) <= 14)
    numeric_or_unit_rows = sum(1 for line in lines if re.search(r"\d|gb|mb|mhz|ghz|w|hz|mt/s|yes|no", line, flags=re.I))

    structured_ratio = (key_value_rows + table_rows) / max(1, len(lines))
    short_ratio = short_rows / max(1, len(lines))
    evidence_ratio = numeric_or_unit_rows / max(1, len(lines))
    return structured_ratio >= 0.35 and short_ratio >= 0.45 and evidence_ratio >= 0.25


def compact_structured_text(text: str) -> str:
    lines = meaningful_lines(text)
    sections = []
    current_title = "Details"
    current_items = []

    def flush() -> None:
        nonlocal current_items
        if not current_items:
            return
        sections.append(f"{current_title}: " + " | ".join(current_items))
        current_items = []

    for line in lines:
        parts = key_value_parts(line)
        if parts is None:
            if looks_like_heading(line, 1):
                flush()
                current_title = normalize_text(line.strip("#: "))
            elif line:
                current_items.append(normalize_text(line))
            continue
        key, value = parts
        if not value and looks_like_heading(key, 1):
            flush()
            current_title = key
            continue
        current_items.append(f"{key}: {value}" if value else key)

    flush()
    return "\n".join(sections) if sections else normalize_text(text)


def meaningful_lines(text: str) -> list[str]:
    return [
        normalize_text(line)
        for line in text.replace("\x00", " ").splitlines()
        if normalize_text(line)
    ]


def key_value_parts(line: str) -> tuple[str, str] | None:
    clean = normalize_text(line)
    if not clean:
        return None
    separators = [",", "\t", "|", ":"]
    for separator in separators:
        if separator not in clean:
            continue
        left, right = clean.split(separator, 1)
        key = normalize_text(left.strip(" -|,:"))
        value = normalize_text(right.strip(" -|,:"))
        if separator == ":":
            if not re.search(r"[A-Za-z]", key) or re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", key):
                continue
        if 1 <= len(key) <= 120 and (value or separator in {",", "\t", "|"}):
            return key, value
    return None


def tokens_for_keywords(text: str) -> list[str]:
    counts = {}
    for token in TOKEN_RE.findall(text.lower()):
        if len(token) <= 2 or token in RETRIEVAL_STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
    return [
        token
        for token, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:24]
    ]


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
