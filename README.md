# Nimbus Agentic RAG

Nimbus is a local-first agentic RAG application for building a private knowledge
workspace from your own documents. It stores full source documents for reading,
builds dense knowledge entries for retrieval, keeps persistent chat history, and
can answer through either a fast RAG pipeline or a controlled agentic workflow.

The system is designed to run with:

- A local OpenAI-compatible model server such as LM Studio
- Qdrant for vector search
- SQLite for local document and chat storage
- A browser UI served by the Python backend

## What Nimbus Does

- Upload or paste source material: text, Markdown, CSV, JSON, logs, PDFs, and images.
- Store full extracted source text in SQLite so documents remain readable in the UI.
- Create semantic source chunks and store them in Qdrant for fallback evidence.
- Build compact Knowledge Base entries with an LLM and store them in a separate Qdrant collection.
- Keep multiple persistent chats with rename and delete support.
- Answer from the current chat context, Knowledge Base evidence, and Source Base evidence.
- Let you choose between Fast RAG and Agentic mode per question.
- Show answer sources and expandable agent steps in the UI.
- Manage Nimbus and Qdrant with cross-platform controller scripts.

## Architecture

![Nimbus system architecture](docs/system-architecture.svg)

Nimbus has four major layers:

| Layer | Responsibility |
| --- | --- |
| Browser UI | Uploads, chats, settings, source document reading, Knowledge Base inspection, answer mode selection |
| HTTP server | Static file serving and JSON API routes |
| Application services | Chat orchestration, RAG, agent loop, job queue, settings, ingestion |
| Storage and models | SQLite, Qdrant, local OpenAI-compatible chat, vision, and embedding models |

Main runtime path:

```text
Browser UI
  -> nimbus/server.py
  -> NimbusApplication
  -> Fast RAG path or Agentic path
  -> Qdrant + SQLite + local model endpoint
  -> answer with sources
```

## Data And Retrieval Flow

![Nimbus data and retrieval flow](docs/data-retrieval-flow.svg)

Nimbus separates full source storage from semantic retrieval storage.

| Store | Technology | Contents | Purpose |
| --- | --- | --- | --- |
| Source Documents | SQLite `data/rag.sqlite` | Full cleaned extracted text and document metadata | Frontend document reader and full-document access |
| Chats | SQLite `data/chats.sqlite` | Chat sessions, messages, stored sources | Persistent conversations |
| Knowledge Base | Qdrant `QDRANT_COLLECTION` | Dense LLM-distilled entries embedded from keywords | Primary compact retrieval |
| Source chunks | Qdrant `QDRANT_SOURCE_COLLECTION` | Semantic source chunks embedded from keywords | Raw fallback evidence |

## Agentic Workflow

![Nimbus agentic flow](docs/agentic-flow.svg)

Agentic mode adds a bounded tool loop before final answer generation. The LLM
does not directly run code or change the system. It returns structured JSON
tool decisions, and Nimbus validates and executes those tools in the backend.

Current agent tools are read-only:

| Tool | Purpose |
| --- | --- |
| `search_knowledge` | Search compact Knowledge Base entries |
| `search_source` | Search semantic Source Base chunks |
| `list_documents` | List uploaded Source Documents |
| `open_source_document` | Open full extracted text for one Source Document |
| `inspect_settings` | Inspect runtime model, embedding, Qdrant, and processing settings |

Agentic mode stops when:

- The model returns `{"action":"final"}`
- `AGENT_MAX_STEPS` is reached
- The model returns an invalid decision
- No more useful read-only tool calls are needed

If the agent reaches final answer generation without collected sources, Nimbus
performs a default evidence pass against both Knowledge Base and Source chunks.

## Fast RAG Versus Agentic Mode

| Mode | Endpoint | Best for | Flow |
| --- | --- | --- | --- |
| Fast RAG | `POST /api/ask` | Quick grounded answers | Rewrite query, expand query, search both Qdrant collections, rerank, answer |
| Agentic | `POST /api/agent/ask` | Multi-step investigation, broad document questions, uncertain retrieval | Plan tool, execute tool, observe, repeat, answer |

Both modes use persistent chat context. Previous chat messages clarify follow-up
questions, but source grounding still comes from retrieved Knowledge Base or
Source Base evidence.

## Requirements

- Python 3.10 or newer
- Docker Desktop or another Docker runtime for Qdrant
- A local OpenAI-compatible API server
- A chat model
- An embedding model
- Optional: an image-capable model for image extraction

The Python dependency list is intentionally small:

```text
pypdf>=5.0.0
```

## Quick Start

Clone the project:

```powershell
git clone https://github.com/mugilnimbus/Nimbus_Agentic_RAG.git
cd Nimbus_Agentic_RAG
```

Create your runtime config:

```powershell
Copy-Item .env.example .env
```

Edit `.env` for your local model server. The most important values are:

```env
OPENAI_BASE_URL=http://127.0.0.1:1234/v1
OPENAI_MODEL=<chat-model-name>
IMAGE_MODEL=<image-capable-chat-model-name>
EMBEDDING_MODEL=<embedding-model-name>
OPENAI_API_KEY=local-key
QDRANT_URL=http://127.0.0.1:6333
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start Docker Desktop, start your local model server, then start Nimbus:

```powershell
.\scripts\nimbusctl.ps1 start
```

Open the app:

```text
http://localhost:8000
```

Open Qdrant dashboard:

```text
http://localhost:6333/dashboard
```

## Running On Different Platforms

Windows PowerShell:

```powershell
.\scripts\nimbusctl.ps1 status
.\scripts\nimbusctl.ps1 start
.\scripts\nimbusctl.ps1 stop
.\scripts\nimbusctl.ps1 restart
```

Windows Command Prompt:

```cmd
nimbus status
nimbus start
nimbus stop
nimbus restart
```

Cross-platform Python controller:

```bash
python scripts/nimbusctl.py status
python scripts/nimbusctl.py start
python scripts/nimbusctl.py stop
python scripts/nimbusctl.py restart
```

Run Nimbus in the foreground for debugging:

```powershell
.\scripts\nimbusctl.ps1 start -Foreground
```

Stop Nimbus but keep Qdrant running:

```powershell
.\scripts\nimbusctl.ps1 stop -KeepQdrant
```

Start or stop only Qdrant:

```powershell
.\scripts\nimbusctl.ps1 start-qdrant
.\scripts\nimbusctl.ps1 stop-qdrant
```

## Manual Qdrant Setup

The controller can create and start the Qdrant container automatically. To do it
manually instead:

```powershell
docker volume create nimbus_qdrant_storage
```

```powershell
docker run -d `
  --name nimbus-qdrant `
  -p 6333:6333 `
  -p 6334:6334 `
  -v nimbus_qdrant_storage:/qdrant/storage `
  qdrant/qdrant:latest
```

Check Qdrant:

```powershell
Invoke-WebRequest http://127.0.0.1:6333/collections -UseBasicParsing
```

## Using The UI

1. Start Nimbus and open `http://localhost:8000`.
2. Add knowledge from the left sidebar.
3. Upload a file or paste text.
4. Choose whether to build the Knowledge Base immediately.
5. Use the `Chat`, `Knowledge Base`, and `Source Documents` views from the top bar.
6. In the chat composer, choose `Fast RAG` or `Agentic`.
7. Ask a question.
8. Expand `Sources` to inspect the evidence used.
9. In Agentic mode, expand `Agent steps` to inspect tool calls.

### Source Documents View

The Source Documents view reads full extracted document text from SQLite. It is
for human inspection and document-level operations.

### Knowledge Base View

The Knowledge Base view reads Qdrant Knowledge Base entries. It shows compact
LLM-distilled information, source metadata, and keywords.

### Operations

Background jobs appear in the Operations panel. Ingestion and Knowledge Base
builds run there instead of writing job-status messages into chat.

## Ingestion Pipeline

```text
Upload or paste content
  -> extract text
  -> clean and normalize text
  -> store full Source Document in SQLite
  -> create semantic source chunks
  -> embed source chunk keywords into Qdrant Source collection
  -> optionally build dense Knowledge Base entries
  -> embed Knowledge Base entry keywords into Qdrant Knowledge collection
```

Supported source types:

- Plain text
- Markdown
- CSV
- JSON
- Logs
- PDF
- Images: PNG, JPG, JPEG, WebP, GIF, BMP

PDF text extraction uses `pypdf`. Image extraction is sent to the configured
`IMAGE_MODEL`.

## Knowledge Base Build Pipeline

Nimbus builds the Knowledge Base from the full source text stored in SQLite.

```text
Full Source Document
  -> semantic sections
  -> packed model-budget groups
  -> LLM dense distillation
  -> compact JSON entries
  -> keyword embeddings
  -> Qdrant Knowledge Base points
```

The Knowledge Base prompt is designed to keep information dense. Tables,
specifications, procedures, warnings, logs, code snippets, and related details
stay together whenever possible.

## Retrieval Pipeline

Fast RAG uses this flow:

```text
Question + chat memory
  -> query rewrite
  -> query expansion
  -> search Knowledge Base
  -> search Source chunks
  -> merge candidates
  -> optional LLM rerank
  -> final grounded answer
```

Retrieval scoring combines:

- Qdrant vector similarity
- Lexical overlap scoring
- Exact match bonus
- Optional LLM reranking

The final answer prompt can reason from solid evidence. If a result can be
derived by calculation, comparison, estimation, or conservative assumptions, the
model is instructed to answer clearly and label inference separately from direct
source facts.

## Agentic Pipeline

Agentic mode uses this flow:

```text
Question + chat memory
  -> LLM chooses next JSON tool action
  -> backend validates tool name and arguments
  -> AgentToolbox executes read-only tool
  -> observation is stored
  -> sources are collected
  -> repeat until final or step limit
  -> final grounded answer from observations and evidence
```

The agent is modular. New tools can be added by registering an `AgentTool` in
`nimbus/tools.py` without rewriting the agent loop.

## Configuration Reference

Private runtime settings live in `.env`, which is ignored by git. Public default
keys live in `.env.example`.

```env
OPENAI_BASE_URL=http://127.0.0.1:1234/v1
OPENAI_MODEL=<chat-model-name>
IMAGE_MODEL=<image-capable-chat-model-name>
EMBEDDING_MODEL=<embedding-model-name>
OPENAI_API_KEY=local-key

VECTOR_BACKEND=qdrant
QDRANT_URL=http://127.0.0.1:6333
QDRANT_PORT=6333
QDRANT_COLLECTION=nimbus_knowledge_base
QDRANT_SOURCE_COLLECTION=nimbus_source_chunks
QDRANT_RUNTIME=docker
QDRANT_CONTAINER_NAME=nimbus-qdrant
QDRANT_DOCKER_IMAGE=qdrant/qdrant:latest
QDRANT_DOCKER_VOLUME=nimbus_qdrant_storage

PORT=8000
HOST=127.0.0.1
RAG_DB=./data/rag.sqlite

KNOWLEDGE_GROUP_CHUNKS=30
KNOWLEDGE_MAX_TOKENS=12000
KNOWLEDGE_CONCURRENCY=1
SOURCE_CHUNK_MAX_WORDS=650
EXTRACTION_WORKERS=12

CHAT_MEMORY_TURNS=24
CHAT_MEMORY_SUMMARY_CHARS=700
CHAT_MEMORY_MESSAGE_CHARS=4000
RAG_RERANK=1
AGENT_MAX_STEPS=6
```

Environment variables already set in the shell override values loaded from
`.env`.

## API Reference

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Runtime health and model/vector status |
| `GET` | `/api/settings` | Current runtime settings |
| `POST` | `/api/settings` | Update runtime settings for the current process |
| `GET` | `/api/connections` | Check local LLM and Qdrant connectivity |
| `GET` | `/api/chats` | List chats |
| `POST` | `/api/chats` | Create chat |
| `GET` | `/api/chats/{id}/messages` | Load chat messages |
| `PATCH` | `/api/chats/{id}` | Rename chat |
| `DELETE` | `/api/chats/{id}` | Delete chat |
| `POST` | `/api/chats/{id}/rename` | Rename chat fallback route |
| `POST` | `/api/chats/{id}/delete` | Delete chat fallback route |
| `GET` | `/api/documents` | List Source Documents |
| `POST` | `/api/documents` | Queue document ingestion |
| `GET` | `/api/documents/{id}/chunks` | Return full Source Document text for the reader |
| `DELETE` | `/api/documents/{id}` | Delete Source Document and related vector entries |
| `POST` | `/api/documents/{id}/build-knowledge` | Queue Knowledge Base build for one document |
| `GET` | `/api/knowledge` | List Knowledge Base entries |
| `GET` | `/api/search` | Search Knowledge Base entries |
| `GET` | `/api/jobs` | List background jobs |
| `POST` | `/api/ask` | Fast RAG chat answer |
| `POST` | `/api/agent/ask` | Agentic chat answer |
| `POST` | `/api/rebuild-knowledge` | Rebuild Source chunk and Knowledge Base indexes |

## Project Structure

```text
app.py
nimbus/
  agent.py              Agentic planning and answer loop
  answer_engine.py      Fast RAG answer engine
  application.py        App orchestration, settings, jobs, chats
  chat_memory.py        Rolling chat memory and focus context
  chat_store.py         Persistent chat SQLite store
  config.py             .env loading helpers
  extraction.py         Text, PDF, image input extraction
  jobs.py               In-memory background job queue
  knowledge.py          Knowledge Base builder
  knowledge_parser.py   LLM JSON parsing and fallback parsing
  llm.py                OpenAI-compatible chat and embedding client
  models.py             Shared data models
  prompts.py            Prompt policy and all prompt templates
  rag.py                RAG facade wiring stores, LLM, builder, answer engine
  retrieval.py          Retrieval scoring, focus, rerank helpers
  source_chunks.py      Semantic source chunk indexing
  source_store.py       SQLite Source Document store
  text_processing.py    Normalization and semantic chunking
  tools.py              Agent tool registry
  vector_store.py       Qdrant storage and search
static/
  index.html
  app.js
  styles.css
scripts/
  nimbusctl.ps1
  nimbusctl.py
  start_all.ps1
  stop_all.ps1
  restart_all.ps1
  status.ps1
docs/
  system-architecture.svg
  data-retrieval-flow.svg
  agentic-flow.svg
```

## Prompting Policy

Nimbus keeps all prompt templates in `nimbus/prompts.py`. The project policy is:

- Keep prompts general-purpose.
- Do not add hard-coded fixes for specific examples, products, hardware,
  numbers, files, or user questions.
- Improve the technical path when behavior needs improvement: retrieval, query
  rewrite, reranking, context selection, memory, tool use, or prompt structure.
- Separate direct evidence from inference in final answers.
- Cite only sources that are actually used.

## Runtime Data And Git Safety

The repository ignores private and generated runtime data:

```text
.env
.env.local
.conda/
data/
*.sqlite
*.sqlite-shm
*.sqlite-wal
storage/
snapshots/
qdrant/
```

Do not commit `.env`, SQLite databases, Qdrant volumes, model files, or local
runtime logs.

## Troubleshooting

### Nimbus port 8000 is already in use

Use the controller restart command. It is designed to stop stale Nimbus
processes and clean up the managed runtime PID:

```powershell
.\scripts\nimbusctl.ps1 restart
```

### Qdrant did not become ready

Make sure Docker Desktop is running, then start Qdrant:

```powershell
.\scripts\nimbusctl.ps1 start-qdrant
```

Check the dashboard:

```text
http://localhost:6333/dashboard
```

### The model endpoint is unavailable

Start your local OpenAI-compatible server and verify `.env`:

```env
OPENAI_BASE_URL=http://127.0.0.1:1234/v1
OPENAI_MODEL=<chat-model-name>
EMBEDDING_MODEL=<embedding-model-name>
```

Then use the Settings panel in the UI or:

```powershell
.\scripts\nimbusctl.ps1 restart
```

### Existing vectors do not match the embedding model

Qdrant collections are tied to vector dimensions. If you change
`EMBEDDING_MODEL`, use new collection names or recreate the collections:

```env
QDRANT_COLLECTION=nimbus_knowledge_base_new
QDRANT_SOURCE_COLLECTION=nimbus_source_chunks_new
```

Then rebuild the Knowledge Base from the UI or with:

```text
POST /api/rebuild-knowledge
```

## Current Limitations

- Qdrant must be running for semantic retrieval.
- Image-only PDF pages are not rendered page-by-page into vision extraction.
- Background job history is in memory and resets when Nimbus restarts.
- Agent tools are currently read-only. Write, delete, rebuild, or system-control
  agent tools should require explicit confirmation before being added.

## License

No license file is included yet. Add one before distributing or accepting
external contributions.
