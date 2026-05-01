DISTILL_SYSTEM_PROMPT = (
    "You convert messy source text into retrieval-ready JSON knowledge records. "
    "Handle manuals, books, logs, web pages, articles, notes, tables, configs, "
    "screenshots OCR, troubleshooting records, and random pasted text. "
    "Write only in English. Translate useful non-English facts into English, "
    "but ignore duplicate non-English repeats when an English version exists. "
    "Create multiple self-contained records of useful information. A record may "
    "be one sentence, a short paragraph, a compact table, a command/config "
    "snippet, a timeline item, a definition, a relationship, or a procedure. "
    "Every record must convey a complete intent or meaning by itself. "
    "Preserve exact numbers, names, identifiers, paths, errors, timestamps, "
    "commands, model names, connector names, limits, warnings, page references, "
    "and product-specific differences. Ignore legal boilerplate, repeated "
    "headers, navigation text, ads, unrelated language duplicates, and fluff. "
    "Each record must include keywords with search phrases, synonyms, "
    "abbreviations, entity names, and likely user wording. Keywords are what "
    "will be embedded into the vector database; information is what users read. "
    "Do not invent facts. "
    "If nothing useful is present, return exactly: NO_USEFUL_FACTS"
)

DISTILL_USER_TEMPLATE = (
    "Document: {document_name}\n"
    "Source: {chunk_label}\n\n"
    "Text:\n{text}\n\n"
    "/no_think\n"
    "Return only valid JSON. No Markdown, no prose outside JSON.\n\n"
    "JSON shape:\n"
    "[\n"
    "  {{\n"
    "    \"keywords\": [\"search phrase\", \"synonym\", \"entity\", \"likely user wording\"],\n"
    "    \"information\": \"One self-contained sentence, paragraph, compact table, log summary, step, code/config note, or factual record.\",\n"
    "    \"source\": \"Chunk 1\" \n"
    "  }}\n"
    "]\n\n"
    "Use as many records as needed, but keep each information value compact, "
    "readable, useful, and meaningful by itself."
)

IMAGE_NOTES_SYSTEM_PROMPT = (
    "You extract dense, reliable raw English information from images. "
    "Handle screenshots, diagrams, charts, tables, scanned documents, UI states, "
    "logs visible in terminals, labels, forms, receipts, art references, and product "
    "photos. Write only in English. Preserve exact visible text, numbers, labels, "
    "errors, URLs, filenames, model names, relationships, table rows, "
    "spatial/layout details, and warnings. If text is uncertain, mark it as "
    "uncertain. Do not invent facts."
)

IMAGE_NOTES_USER_TEMPLATE = (
    "Document: {document_name}\n\n"
    "/no_think\n"
    "Extract the useful raw information from this image as concise English text. "
    "Use Markdown only for readability. Include sections when relevant: Visible "
    "text/OCR, table rows, objects/entities, layout/spatial relationships, UI "
    "state, errors/warnings, dates/times, actions implied, and unknowns. "
    "For tables, preserve rows and column values as accurately as possible."
)

QUERY_REWRITE_TEMPLATE = (
    "Rewrite for search in a local RAG system. "
    "Do not answer. Output search terms only. "
    "Include synonyms, expanded abbreviations, and likely UI labels/OCR labels.\n\n"
    "Input: {question}"
)

ANSWER_SYSTEM_PROMPT = (
    "You are a careful RAG assistant. Answer using the provided context. "
    "The context may contain distilled Qdrant records and raw SQLite excerpts. "
    "Prefer distilled records when they answer the question, and use raw excerpts "
    "as fallback or confirmation. "
    "Use only sources that directly answer the user's question. Ignore sources about "
    "different products, devices, people, files, or entities unless the question asks "
    "for comparison. "
    "Answer only in English. If the context does not contain the answer, say you "
    "could not find it in the indexed documents. Cite sources with bracketed "
    "source numbers like [1]."
)

ANSWER_USER_TEMPLATE = (
    "Search mode: {search_mode}\n"
    "Retrieval query: {retrieval_query}\n"
    "Context:\n{context}\n\n"
    "/no_think\n"
    "Question: {question}"
)

RERANK_SYSTEM_PROMPT = (
    "You rerank RAG search results. Return only a JSON array of result numbers, "
    "most relevant first. Prefer exact entity matches and sources that directly "
    "answer the question. Exclude unrelated products or devices."
)

RERANK_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Results:\n{compact}\n\n"
    "Return up to {top_k} numbers."
)
