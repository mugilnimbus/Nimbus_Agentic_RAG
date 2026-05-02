# Nimbus Vector RAG

A local two-tier RAG system with a dark web UI. The Source Base stores only
source documents and readable source chunks in SQLite. The Knowledge Base stores
LLM-built retrieval entries in Qdrant: keyword vectors plus readable
information payloads. The app retrieves Knowledge Base entries first, falls
back to Source Base chunks when needed, and sends grounded prompts to an
OpenAI-compatible LM Studio endpoint.

## Architecture

```text
Browser UI
  |
  |  upload files / ask questions / inspect DBs
  v
Python HTTP Server: app.py -> nimbus/server.py
  |
  |-- Static UI: static/index.html, app.js, styles.css
  |
  |-- API
      |-- GET    /api/health
      |-- GET    /api/documents
      |-- GET    /api/documents/{id}/chunks
      |-- GET    /api/search
      |-- GET    /api/knowledge
      |-- POST   /api/documents
      |-- POST   /api/documents/{id}/build-knowledge
      |-- POST   /api/ask
      |-- POST   /api/rebuild-knowledge
      |-- DELETE /api/documents/{id}
  |
  v
Nimbus package: nimbus/
  |
  |-- server.py: API routes and app settings
  |-- extraction.py: file parsing, PDF text extraction, base64 validation
  |-- jobs.py: in-memory background job queue and progress tracking
  |-- prompts.py: LLM prompts for Knowledge Base building, image extraction, rewriting, answers
  |-- rag.py: RAG core, Source Base store, Knowledge Base vector store, LM Studio calls
  |
  v
SQLite DB: data/rag.sqlite
  |
  |-- documents table
  |-- chunks table with readable source chunk text only
  |
  v
Qdrant Vector DB
  configured by QDRANT_URL
  collection configured by QDRANT_COLLECTION
  |
  |-- Knowledge Base entry vectors built from keywords
  |-- payload: document id, file name, keywords, information, source range
  |
  v
LM Studio OpenAI-compatible endpoint
configured by OPENAI_BASE_URL
model configured by OPENAI_MODEL
image extraction model configured by IMAGE_MODEL
```

## Data Stores

There are two logical databases in the app. The Source Base stores source
documents and readable source chunk text only. The Knowledge Base stores
LLM-built entries only. Source Base chunks are not vectorized into Qdrant.

```text
Knowledge Base
  Stores LLM-generated English entries from Source Base chunks and images.
  Each point vector is built from keywords.
  Each payload stores keywords, information, document name, source range,
  embedding model, and prompt version.

Source Base
  Stores cleaned English source text in SQLite.
  Used as fallback and evidence confirmation.
```

## Ingestion Flow

```text
User uploads/pastes content
  |
  v
POST /api/documents
  |
  |-- text/log/csv/md/json
  |     -> English filter
  |     -> chunk
  |     -> store as raw
  |     -> optional Knowledge Base entries
  |
  |-- PDF
  |     -> pypdf text extraction
  |     -> English filter
  |     -> chunk
  |     -> store as raw
  |     -> optional Knowledge Base entries
  |
  |-- image
      -> send image to the configured IMAGE_MODEL
      -> extract Source Base observations
      -> store as Source Base chunks
      -> build Knowledge Base entries
```

Supported uploads include images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`,
`.bmp`), PDFs, and plain text-style files such as `.txt`, `.md`, `.csv`, `.json`,
and `.log`.

## Knowledge Base Build Flow

```text
Raw document
  |
  v
group chunks in large batches
  KNOWLEDGE_GROUP_CHUNKS=30
  |
  v
send batches to LM Studio
  KNOWLEDGE_CONCURRENCY=1 by default
  |
  v
LLM produces JSON Knowledge Base entries
  |
  v
embed each Knowledge Base entry's keywords
  |
  v
store Knowledge Base points with readable information payloads
```

## Question Answering Flow

```text
User question
  |
  v
LLM query rewrite
  Example:
  "what is my laptops cpu?"
  ->
  "laptop cpu central processing unit processor model system information..."
  |
  v
Search both:
  - rewritten query
  - original question
  |
  v
Search Knowledge Base first
  |
  | if useful hits found
  v
Use Knowledge Base entries + some Source Base confirmation
  |
  | otherwise
  v
Use Source Base fallback
  |
  v
Build grounded context
  |
  v
Send context + question to the configured chat model
  |
  v
Return answer + source dropdowns
```

## Retrieval

The system uses semantic embeddings from LM Studio and stores them in Qdrant.

```text
Knowledge Base entry keywords
  -> /v1/embeddings
  -> configured embedding model
  -> normalize
  -> Qdrant point vector
  -> payload stores readable information and source metadata
```

At search time:

```text
Qdrant semantic vector similarity
+ small lexical overlap score
+ Knowledge Base preference
= ranked Knowledge Base entries
```

SQLite is still used because it is excellent for local document metadata and
chunk inspection. It does not own embeddings anymore. Qdrant is the true vector
database used for similarity search. Existing Source Base chunks can be rebuilt into
Qdrant through `POST /api/rebuild-knowledge`.

## Qdrant Setup

Start Qdrant before indexing or searching with the default vector backend.

Use `scripts/start_all.ps1` with Docker Qdrant configured in `.env`, or start
Qdrant yourself and put the Qdrant URL and collection name in `.env`.

Qdrant must be running for retrieval. SQLite no longer stores vectors, so it is
not a semantic-search fallback.

## UI

```text
Dark web interface
  |
  |-- left sidebar
  |     |-- upload / paste / index
  |     |-- Knowledge Base card
  |     |-- Source Base card
  |
  |-- database drawer
  |     |-- documents inside selected DB
  |     |-- chunk viewer
  |
  |-- chat area
        |-- user/assistant bubbles
        |-- source dropdown per answer
        |-- rendered Markdown for Knowledge Base entries and Source Base chunks
```

## Main Files

```text
app.py
  small entrypoint that starts Nimbus

nimbus/server.py
  HTTP server, API routes, settings, background task wiring

nimbus/extraction.py
  file parsing, image/PDF dispatch, concurrent PDF page extraction

nimbus/jobs.py
  in-memory job queue, serialized execution, progress state

nimbus/rag.py
  RAG logic, Source Base chunks, Knowledge Base entries, retrieval, LLM calls

nimbus/prompts.py
  Prompt templates for JSON Knowledge Base building, image extraction, query rewrite,
  answer generation, and reranking

static/index.html
  UI structure

static/app.js
  browser logic, uploads, chat, DB drawer, Markdown rendering

static/styles.css
  dark professional UI styling

data/rag.sqlite
  persistent document metadata and source chunk text

storage/
  legacy local Qdrant storage path; Docker Qdrant normally uses its named volume
```

## Run

Create `.env` from `.env.example`, install Python dependencies, start your
configured LM Studio chat and embedding models, then run both services:

```powershell
python -m pip install -r requirements.txt
```

```powershell
.\scripts\start_all.ps1
```

Open:

```text
http://localhost:8000
```

## Configuration

Private runtime settings live in `.env`, which is ignored by git. Use
`.env.example` as the template.

```text
OPENAI_BASE_URL=<openai-compatible-endpoint>
OPENAI_MODEL=<chat-model-name>
IMAGE_MODEL=<image-capable-chat-model-name>
EMBEDDING_MODEL=<embedding-model-name>
OPENAI_API_KEY=<api-key-or-local-placeholder>
PORT=8000
RAG_DB=./data/rag.sqlite
KNOWLEDGE_GROUP_CHUNKS=30
KNOWLEDGE_MAX_TOKENS=12000
KNOWLEDGE_CONCURRENCY=1
EXTRACTION_WORKERS=12
VECTOR_BACKEND=qdrant
QDRANT_URL=<qdrant-url>
QDRANT_PORT=<qdrant-port>
QDRANT_COLLECTION=nimbus_knowledge_base
QDRANT_RUNTIME=docker
QDRANT_CONTAINER_NAME=nimbus-qdrant
QDRANT_DOCKER_IMAGE=qdrant/qdrant:latest
QDRANT_DOCKER_VOLUME=nimbus_qdrant_storage
```

Environment variables override `.env` values if they are already set before
starting the app.

## Strengths

- Runs locally.
- Uses the LM Studio OpenAI-compatible endpoint.
- Supports text, PDF, CSV/log/article-style text, and images.
- Separates raw evidence from LLM-built Knowledge Base entries.
- Uses the configured semantic embedding model.
- Uses LLM query rewriting before retrieval.
- Uses concurrent CPU-side PDF extraction.
- Uses serialized LLM jobs by default for local-model stability.
- Lets you inspect databases, documents, chunks, answers, and sources.

## Limitations

- Qdrant must be running for semantic retrieval.
- PDF image-only/scanned pages are not yet rendered page-by-page into vision text.
- Markdown rendering is intentionally minimal and safe.
- Jobs are in-memory, so progress history resets when Nimbus restarts.
