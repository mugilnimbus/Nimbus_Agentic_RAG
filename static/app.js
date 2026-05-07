const statusEl = document.querySelector("#status");
const modelPill = document.querySelector("#modelPill");
const chatList = document.querySelector("#chatList");
const newChatButton = document.querySelector("#newChat");
const ingestForm = document.querySelector("#ingestForm");
const askForm = document.querySelector("#askForm");
const messages = document.querySelector("#messages");
const chatPage = document.querySelector("#chatPage");
const sourcePage = document.querySelector("#sourcePage");
const knowledgePage = document.querySelector("#knowledgePage");
const chatViewButton = document.querySelector("#chatViewButton");
const knowledgeViewButton = document.querySelector("#knowledgeViewButton");
const sourceViewButton = document.querySelector("#sourceViewButton");
const sourceSearchInput = document.querySelector("#sourceSearch");
const sourceRefreshButton = document.querySelector("#sourceRefresh");
const sourceStats = document.querySelector("#sourceStats");
const sourceDocumentList = document.querySelector("#sourceDocumentList");
const sourceReader = document.querySelector("#sourceReader");
const knowledgeSearchInput = document.querySelector("#knowledgeSearch");
const knowledgeRefreshButton = document.querySelector("#knowledgeRefresh");
const knowledgeStats = document.querySelector("#knowledgeStats");
const knowledgeListModeButton = document.querySelector("#knowledgeListMode");
const knowledgeGraphModeButton = document.querySelector("#knowledgeGraphMode");
const knowledgeListView = document.querySelector("#knowledgeListView");
const knowledgeGraphView = document.querySelector("#knowledgeGraphView");
const knowledgeEntryList = document.querySelector("#knowledgeEntryList");
const knowledgeReader = document.querySelector("#knowledgeReader");
const knowledgeGraph = document.querySelector("#knowledgeGraph");
const knowledgeGraphDetail = document.querySelector("#knowledgeGraphDetail");
const fileInput = document.querySelector("#docFile");
const textInput = document.querySelector("#docText");
const nameInput = document.querySelector("#docName");
const questionInput = document.querySelector("#question");
const topKInput = document.querySelector("#topK");
const buildKnowledgeUploadInput = document.querySelector("#buildKnowledgeUpload");
const jobList = document.querySelector("#jobList");
const refreshJobsButton = document.querySelector("#refreshJobs");
const rebuildKnowledgeButton = document.querySelector("#rebuildKnowledge");
const settingsToggle = document.querySelector("#settingsToggle");
const settingsPanel = document.querySelector("#settingsPanel");
const settingsForm = document.querySelector("#settingsForm");
const settingBaseUrl = document.querySelector("#settingBaseUrl");
const settingModel = document.querySelector("#settingModel");
const settingImageModel = document.querySelector("#settingImageModel");
const settingEmbeddingModel = document.querySelector("#settingEmbeddingModel");
const settingVectorBackend = document.querySelector("#settingVectorBackend");
const settingQdrantUrl = document.querySelector("#settingQdrantUrl");
const settingQdrantCollection = document.querySelector("#settingQdrantCollection");
const settingQdrantSourceCollection = document.querySelector("#settingQdrantSourceCollection");
const settingConcurrency = document.querySelector("#settingConcurrency");
const settingGroupChunks = document.querySelector("#settingGroupChunks");
const settingMaxTokens = document.querySelector("#settingMaxTokens");
const settingSourceChunkWords = document.querySelector("#settingSourceChunkWords");
const settingExtractionWorkers = document.querySelector("#settingExtractionWorkers");
const settingRerank = document.querySelector("#settingRerank");
const reloadSettingsButton = document.querySelector("#reloadSettings");
const checkLlmButton = document.querySelector("#checkLlm");
const checkQdrantButton = document.querySelector("#checkQdrant");
const llmStatus = document.querySelector("#llmStatus");
const qdrantStatus = document.querySelector("#qdrantStatus");
const drawer = document.querySelector("#dbDrawer");
const overlay = document.querySelector("#dbOverlay");
const drawerKind = document.querySelector("#drawerKind");
const drawerTitle = document.querySelector("#drawerTitle");
const drawerDocs = document.querySelector("#drawerDocs");
const closeDrawerButton = document.querySelector("#closeDrawer");
const chatDialog = document.querySelector("#chatDialog");
const chatDialogForm = document.querySelector("#chatDialogForm");
const chatDialogTitle = document.querySelector("#chatDialogTitle");
const chatDialogMessage = document.querySelector("#chatDialogMessage");
const chatDialogInputWrap = document.querySelector("#chatDialogInputWrap");
const chatDialogInput = document.querySelector("#chatDialogInput");
const chatDialogCancel = document.querySelector("#chatDialogCancel");
const chatDialogConfirm = document.querySelector("#chatDialogConfirm");

let selectedFilePayload = null;
let documentsCache = [];
let chatsCache = [];
let activeChatId = null;
let jobsCache = [];
let jobPollTimer = null;
let qdrantInfo = {};
let selectedSourceDocumentId = null;
let knowledgeEntriesCache = [];
let selectedKnowledgeEntryId = null;
let knowledgeMode = "list";

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[char]);
}

function renderMarkdown(value) {
  const lines = escapeHtml(value).split(/\r?\n/);
  const html = [];
  let inList = false;

  const closeList = () => {
    if (inList) {
      html.push("</ul>");
      inList = false;
    }
  };

  const inline = (text) => text
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      closeList();
      continue;
    }
    const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      closeList();
      html.push(`<h${heading[1].length + 2}>${inline(heading[2])}</h${heading[1].length + 2}>`);
      continue;
    }
    const bullet = trimmed.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      if (!inList) {
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${inline(bullet[1])}</li>`);
      continue;
    }
    closeList();
    html.push(`<p>${inline(trimmed)}</p>`);
  }
  closeList();
  return html.join("");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Request failed");
  }
  return payload;
}

function setStatusChip(element, label, state = "muted") {
  if (!element) return;
  element.textContent = label;
  element.className = `status-chip ${state}`;
}

function updateConnectionBadges(health) {
  if (!health) return;
  if (health.llm) {
    const llm = health.llm;
    setStatusChip(
      llmStatus,
      llm.status === "ready" ? `${llm.model_count || 0} models` : "LM Studio offline",
      llm.status === "ready" ? "ready" : "error",
    );
  } else {
    setStatusChip(llmStatus, "Nimbus online", "ready");
  }
  const qdrant = health.qdrant || {};
  if (qdrant.status === "ready") {
    setStatusChip(qdrantStatus, `${qdrant.points_count || 0} entries`, "ready");
  } else if (qdrant.status === "collection_missing") {
    setStatusChip(qdrantStatus, "Collection missing", "warn");
  } else if (qdrant.status) {
    setStatusChip(qdrantStatus, qdrant.status.replaceAll("_", " "), "error");
  } else {
    setStatusChip(qdrantStatus, "Not checked", "muted");
  }
}

function setEmptyState() {
  if (!messages.children.length) {
    messages.innerHTML = `
      <div class="empty">
        <div class="empty-mark">N</div>
        <h3>Ready when you are</h3>
        <p>Upload files, build the Knowledge Base, and ask from your local sources.</p>
      </div>
    `;
  }
}

function clearEmptyState() {
  const empty = messages.querySelector(".empty");
  if (empty) empty.remove();
}

function addMessage(role, text, sources = []) {
  clearEmptyState();
  const row = document.createElement("article");
  row.className = `message-row ${role}`;
  const avatar = role === "user" ? "You" : "N";
  row.innerHTML = `
    <div class="avatar">${avatar}</div>
    <div class="bubble">
      <div class="message-text rendered">${renderMarkdown(text)}</div>
    </div>
  `;

  if (sources.length) {
    const sourceWrap = document.createElement("details");
    sourceWrap.className = "source-pack";
    sourceWrap.innerHTML = `
      <summary>Sources (${sources.length})</summary>
      <div class="source-list">
        ${sources.map((source, index) => `
          <details class="source-item">
            <summary>
              <span class="kind ${escapeHtml(sourceKindLabel(source.document_kind))}">${escapeHtml(sourceKindLabel(source.document_kind))}</span>
              <span>${index + 1}. ${escapeHtml(source.document_name)}</span>
              <small>chunk ${source.chunk_index + 1} · ${source.score}</small>
            </summary>
            <div class="source-body rendered">${renderMarkdown(source.text)}</div>
          </details>
        `).join("")}
      </div>
    `;
    row.querySelector(".bubble").appendChild(sourceWrap);
  }

  messages.appendChild(row);
  messages.scrollTop = messages.scrollHeight;
}

function renderStoredMessages(storedMessages) {
  messages.innerHTML = "";
  for (const message of storedMessages) {
    addMessage(message.role, message.content, message.sources || []);
  }
  setEmptyState();
}

function openChatDialog({ title, message, confirmLabel, danger = false, inputValue = null }) {
  return new Promise((resolve) => {
    chatDialogTitle.textContent = title;
    chatDialogMessage.textContent = message || "";
    chatDialogConfirm.textContent = confirmLabel;
    chatDialogConfirm.classList.toggle("danger-action", danger);
    chatDialogInputWrap.hidden = inputValue === null;
    chatDialogInput.value = inputValue || "";

    const close = (value) => {
      chatDialogForm.removeEventListener("submit", onSubmit);
      chatDialog.removeEventListener("cancel", onCancel);
      if (chatDialog.open) chatDialog.close();
      resolve(value);
    };

    const onCancel = (event) => {
      event.preventDefault();
      close(null);
    };

    const onSubmit = (event) => {
      event.preventDefault();
      const action = event.submitter?.value || "cancel";
      close(action === "confirm" ? (inputValue === null ? true : chatDialogInput.value.trim()) : null);
    };

    chatDialogForm.addEventListener("submit", onSubmit);
    chatDialog.addEventListener("cancel", onCancel);
    chatDialog.showModal();
    if (inputValue !== null) {
      chatDialogInput.focus();
      chatDialogInput.select();
    } else {
      chatDialogCancel.focus();
    }
  });
}

async function refreshHealth() {
  try {
    const health = await api("/api/health");
    qdrantInfo = health.qdrant || {};
    statusEl.textContent = `${health.documents} documents indexed`;
    const backend = health.vector_backend === "qdrant" ? "Qdrant" : "SQLite";
    const imageLabel = health.image_model && health.image_model !== health.model
      ? ` · image ${health.image_model}`
      : "";
    modelPill.textContent = `${health.model}${imageLabel} · ${health.embedding_model} · ${backend}`;
    updateConnectionBadges(health);
    renderKnowledgePage();
  } catch (error) {
    statusEl.textContent = "Server unavailable";
    statusEl.classList.add("error");
    setStatusChip(llmStatus, "Server offline", "error");
    setStatusChip(qdrantStatus, "Unknown", "muted");
  }
}

function jobLabel(job) {
  const detail = job.error || job.detail || job.type;
  return detail || "Working";
}

function renderJobs() {
  if (!jobsCache.length) {
    jobList.innerHTML = `<div class="job-empty">No background jobs yet.</div>`;
    return;
  }
  jobList.innerHTML = jobsCache.slice(0, 6).map((job) => {
    const current = Number(job.progress_current || 0);
    const total = Number(job.progress_total || 0);
    const percent = total > 0 ? Math.max(0, Math.min(100, Math.round((current / total) * 100))) : 0;
    return `
      <article class="job-item ${escapeHtml(job.status)}">
        <strong>${escapeHtml(job.type.replaceAll("-", " "))}</strong>
        <span>${escapeHtml(jobLabel(job))}</span>
        <small>${escapeHtml(job.status)}</small>
        <div class="job-progress" aria-label="Progress">
          <span style="width: ${percent}%"></span>
        </div>
        <div class="job-meta">
          <span>${total > 0 ? `${current}/${total}` : "waiting"}</span>
          <span>${percent}%</span>
        </div>
      </article>
    `;
  }).join("");
}

async function refreshJobs() {
  const { jobs } = await api("/api/jobs?limit=10");
  const hadActive = jobsCache.some((job) => ["queued", "running"].includes(job.status));
  jobsCache = jobs;
  renderJobs();
  const hasActive = jobs.some((job) => ["queued", "running"].includes(job.status));
  if (hadActive && !hasActive) {
    await refreshHealth();
    await refreshDocuments();
  }
  if (hasActive && !jobPollTimer) {
    jobPollTimer = setInterval(refreshJobs, 2000);
  }
  if (!hasActive && jobPollTimer) {
    clearInterval(jobPollTimer);
    jobPollTimer = null;
  }
}

async function queueMaintenance(path, label) {
  try {
    await api(path, { method: "POST", body: "{}" });
    await refreshJobs();
  } catch (error) {
    window.alert(`${label} failed to queue: ${error.message}`);
  }
}

function fillSettings(settings) {
  settingBaseUrl.value = settings.base_url || "";
  settingModel.value = settings.model || "";
  settingImageModel.value = settings.image_model || settings.model || "";
  settingEmbeddingModel.value = settings.embedding_model || "";
  settingVectorBackend.value = settings.vector_backend || "qdrant";
  settingQdrantUrl.value = settings.qdrant_url || "";
  settingQdrantCollection.value = settings.qdrant_collection || "";
  settingQdrantSourceCollection.value = settings.qdrant_source_collection || "";
  settingConcurrency.value = settings.knowledge_concurrency || 1;
  settingGroupChunks.value = settings.knowledge_group_chunks || 30;
  settingMaxTokens.value = settings.knowledge_max_tokens || 12000;
  settingSourceChunkWords.value = settings.source_chunk_max_words || 650;
  settingExtractionWorkers.value = settings.extraction_workers || 12;
  settingRerank.checked = Boolean(settings.rerank_enabled);
}

async function refreshSettings() {
  const settings = await api("/api/settings");
  fillSettings(settings);
}

function databaseSummary(kind) {
  const docs = documentsCache.filter((doc) => doc.kind === kind);
  const chunks = docs.reduce((sum, doc) => sum + Number(doc.chunk_count || 0), 0);
  return { docs, chunks };
}

function sourceKindLabel(kind) {
  return kind === "knowledge" ? "knowledge" : (kind || "source");
}

function renderDatabases() {
  renderKnowledgePage();
}

async function refreshDocuments() {
  const { documents } = await api("/api/documents");
  documentsCache = documents;
  renderSourcePage();
}

async function refreshKnowledgeEntries() {
  const { entries } = await api("/api/knowledge?limit=1000");
  knowledgeEntriesCache = entries;
  renderKnowledgePage();
}

async function refreshDatabases() {
  await refreshHealth();
  await refreshDocuments();
  await refreshKnowledgeEntries();
}

function renderChats() {
  if (!chatsCache.length) {
    chatList.innerHTML = `<div class="job-empty">No chats yet.</div>`;
    return;
  }
  chatList.innerHTML = chatsCache.map((chat) => {
    const active = Number(chat.id) === Number(activeChatId) ? " active" : "";
    const preview = chat.last_message || "Empty chat";
    return `
      <article class="chat-card${active}">
        <button class="chat-main" type="button" data-chat-id="${chat.id}">
          <strong>${escapeHtml(chat.title || "New chat")}</strong>
          <span>${escapeHtml(preview)}</span>
          <small>${Number(chat.message_count || 0)} messages</small>
        </button>
        <div class="chat-actions" aria-label="Chat actions">
          <button type="button" data-chat-rename="${chat.id}">Rename</button>
          <button type="button" data-chat-delete="${chat.id}">Delete</button>
        </div>
      </article>
    `;
  }).join("");
}

async function refreshChats() {
  const { chats } = await api("/api/chats");
  chatsCache = chats;
  if (!activeChatId && chats.length) {
    await openChat(chats[0].id);
    return;
  }
  renderChats();
}

async function createChat() {
  const { chat } = await api("/api/chats", {
    method: "POST",
    body: JSON.stringify({ title: "New chat" }),
  });
  chatsCache = [chat, ...chatsCache];
  await openChat(chat.id);
}

async function openChat(chatId) {
  activeChatId = Number(chatId);
  const { messages: storedMessages } = await api(`/api/chats/${activeChatId}/messages`);
  renderStoredMessages(storedMessages);
  renderChats();
  setWorkspaceView("chat");
  questionInput.focus();
}

async function renameChat(chatId) {
  const chat = chatsCache.find((item) => Number(item.id) === Number(chatId));
  const cleanTitle = await openChatDialog({
    title: "Rename Chat",
    message: "Choose a clearer name for this chat.",
    confirmLabel: "Rename",
    inputValue: chat?.title || "New chat",
  });
  if (!cleanTitle) return;
  const { chat: updatedChat } = await api(`/api/chats/${chatId}/rename`, {
    method: "POST",
    body: JSON.stringify({ title: cleanTitle }),
  });
  chatsCache = chatsCache.map((item) => (Number(item.id) === Number(chatId) ? updatedChat : item));
  renderChats();
}

async function deleteChat(chatId) {
  const chat = chatsCache.find((item) => Number(item.id) === Number(chatId));
  const title = chat?.title || "this chat";
  const shouldDelete = await openChatDialog({
    title: "Delete Chat",
    message: `Delete "${title}"? This removes the chat history from Nimbus.`,
    confirmLabel: "Delete",
    danger: true,
  });
  if (!shouldDelete) return;
  await api(`/api/chats/${chatId}/delete`, { method: "POST", body: "{}" });
  const deletedActiveChat = Number(activeChatId) === Number(chatId);
  chatsCache = chatsCache.filter((item) => Number(item.id) !== Number(chatId));
  if (deletedActiveChat) {
    activeChatId = null;
    messages.innerHTML = "";
  }
  await refreshChats();
  setEmptyState();
}

function setWorkspaceView(view) {
  const showSource = view === "source";
  const showKnowledge = view === "knowledge";
  sourcePage.hidden = !showSource;
  knowledgePage.hidden = !showKnowledge;
  chatPage.hidden = showSource || showKnowledge;
  askForm.hidden = showSource || showKnowledge;
  sourceViewButton.classList.toggle("active", showSource);
  knowledgeViewButton.classList.toggle("active", showKnowledge);
  chatViewButton.classList.toggle("active", !showSource && !showKnowledge);
  if (showSource) {
    renderSourcePage();
  }
  if (showKnowledge) {
    refreshKnowledgeEntries();
  }
}

function filteredSourceDocuments() {
  const query = (sourceSearchInput.value || "").trim().toLowerCase();
  const docs = documentsCache.filter((doc) => doc.kind === "source");
  if (!query) return docs;
  return docs.filter((doc) => {
    const haystack = `${doc.name} ${doc.id} ${doc.content_hash || ""}`.toLowerCase();
    return haystack.includes(query);
  });
}

function renderSourcePage() {
  if (!sourceStats || !sourceDocumentList) return;
  const allDocs = documentsCache.filter((doc) => doc.kind === "source");
  const docs = filteredSourceDocuments();
  const totalChunks = allDocs.reduce((sum, doc) => sum + Number(doc.chunk_count || 0), 0);
  const shownChunks = docs.reduce((sum, doc) => sum + Number(doc.chunk_count || 0), 0);
  sourceStats.innerHTML = `
    <article>
      <strong>${allDocs.length}</strong>
      <span>Total documents</span>
    </article>
    <article>
      <strong>${totalChunks}</strong>
      <span>Retrieval chunks</span>
    </article>
    <article>
      <strong>${docs.length}</strong>
      <span>Shown documents</span>
    </article>
    <article>
      <strong>${shownChunks}</strong>
      <span>Shown retrieval chunks</span>
    </article>
  `;

  if (!docs.length) {
    sourceDocumentList.innerHTML = `<div class="drawer-empty">No Source Base documents match this search.</div>`;
    return;
  }

  sourceDocumentList.innerHTML = docs.map((doc) => {
    const date = new Date(doc.created_at * 1000).toLocaleString();
    const active = Number(doc.id) === Number(selectedSourceDocumentId) ? " active" : "";
    return `
      <button class="source-document-row${active}" type="button" data-source-doc="${doc.id}">
        <span>
          <strong>${escapeHtml(doc.name)}</strong>
          <small>ID ${doc.id} · ${doc.chunk_count} retrieval chunks · ${date}</small>
        </span>
        <em>${escapeHtml((doc.content_hash || "").slice(0, 10))}</em>
      </button>
    `;
  }).join("");
}

async function openSourceDocument(documentId) {
  selectedSourceDocumentId = Number(documentId);
  renderSourcePage();
  const doc = documentsCache.find((item) => Number(item.id) === Number(documentId));
  sourceReader.innerHTML = `<p class="loading">Loading document...</p>`;
  const { chunks } = await api(`/api/documents/${documentId}/chunks`);
  const fullText = chunks.map((chunk) => chunk.text).join("\n\n");
  const wordCount = fullText.split(/\s+/).filter(Boolean).length;
  const date = doc ? new Date(doc.created_at * 1000).toLocaleString() : "";
  sourceReader.innerHTML = `
    <header class="source-reader-head">
      <div>
        <p class="eyebrow">Document ${escapeHtml(documentId)}</p>
        <h3>${escapeHtml(doc?.name || "Source document")}</h3>
        <span>${wordCount} words · ${doc?.chunk_count || 0} retrieval chunks${date ? ` · ${date}` : ""}</span>
      </div>
      <div class="drawer-actions">
        <button type="button" data-build-knowledge="${documentId}">Build Knowledge Base</button>
        <button type="button" data-delete-source="${documentId}">Delete</button>
      </div>
    </header>
    <div class="source-document-body rendered">
      ${renderMarkdown(fullText)}
    </div>
  `;
}

function filteredKnowledgeEntries() {
  const query = (knowledgeSearchInput.value || "").trim().toLowerCase();
  if (!query) return knowledgeEntriesCache;
  return knowledgeEntriesCache.filter((entry) => {
    const haystack = [
      entry.document_name,
      entry.source,
      entry.information,
      ...(entry.keywords || []),
    ].join(" ").toLowerCase();
    return haystack.includes(query);
  });
}

function knowledgeDocumentGroups(entries) {
  const groups = new Map();
  for (const entry of entries) {
    const key = `${entry.document_id}:${entry.document_name}`;
    if (!groups.has(key)) {
      groups.set(key, { id: entry.document_id, name: entry.document_name, entries: [] });
    }
    groups.get(key).entries.push(entry);
  }
  return [...groups.values()];
}

function renderKnowledgePage() {
  if (!knowledgeStats || !knowledgeEntryList) return;
  const entries = filteredKnowledgeEntries();
  const groups = knowledgeDocumentGroups(entries);
  const keywordCount = new Set(entries.flatMap((entry) => entry.keywords || [])).size;
  knowledgeStats.innerHTML = `
    <article>
      <strong>${Number(qdrantInfo.points_count || knowledgeEntriesCache.length || 0)}</strong>
      <span>Qdrant points</span>
    </article>
    <article>
      <strong>${entries.length}</strong>
      <span>Shown entries</span>
    </article>
    <article>
      <strong>${groups.length}</strong>
      <span>Source documents</span>
    </article>
    <article>
      <strong>${keywordCount}</strong>
      <span>Shown keywords</span>
    </article>
  `;

  if (knowledgeMode === "graph") {
    renderKnowledgeGraph(entries);
  } else {
    renderKnowledgeList(entries);
  }
}

function renderKnowledgeList(entries) {
  knowledgeListView.hidden = false;
  knowledgeGraphView.hidden = true;
  if (!entries.length) {
    knowledgeEntryList.innerHTML = `<div class="drawer-empty">No Knowledge Base entries match this search.</div>`;
    return;
  }
  knowledgeEntryList.innerHTML = entries.map((entry) => {
    const active = String(entry.id) === String(selectedKnowledgeEntryId) ? " active" : "";
    const keywords = (entry.keywords || []).slice(0, 6).map((keyword) => `<span>${escapeHtml(keyword)}</span>`).join("");
    return `
      <button class="knowledge-entry-row${active}" type="button" data-knowledge-entry="${escapeHtml(entry.id)}">
        <strong>${escapeHtml(entry.document_name)}</strong>
        <small>${escapeHtml(entry.source || `chunk ${entry.chunk_index + 1}`)}</small>
        <span>${escapeHtml(entry.information)}</span>
        <div class="knowledge-keywords compact">${keywords}</div>
      </button>
    `;
  }).join("");
}

function renderKnowledgeEntry(entry) {
  if (!entry) {
    return `<div class="source-reader-empty">Select a Knowledge Base entry to inspect it.</div>`;
  }
  const keywords = (entry.keywords || []).map((keyword) => `<span>${escapeHtml(keyword)}</span>`).join("");
  return `
    <header class="source-reader-head">
      <div>
        <p class="eyebrow">Knowledge entry</p>
        <h3>${escapeHtml(entry.document_name)}</h3>
        <span>${escapeHtml(entry.source || `chunk ${entry.chunk_index + 1}`)} · source chunks ${entry.source_chunk_start + 1}-${entry.source_chunk_end + 1}</span>
      </div>
    </header>
    <div class="knowledge-keywords">
      <small>Keywords</small>
      ${keywords}
    </div>
    <div class="knowledge-information rendered">${renderMarkdown(entry.information)}</div>
    <details class="payload-preview">
      <summary>Payload</summary>
      <pre>${escapeHtml(JSON.stringify(entry, null, 2))}</pre>
    </details>
  `;
}

function openKnowledgeEntry(entryId) {
  selectedKnowledgeEntryId = String(entryId);
  const entry = knowledgeEntriesCache.find((item) => String(item.id) === String(entryId));
  knowledgeReader.innerHTML = renderKnowledgeEntry(entry);
  knowledgeGraphDetail.innerHTML = renderKnowledgeEntry(entry);
  renderKnowledgePage();
}

function keywordFrequency(entries) {
  const counts = new Map();
  for (const entry of entries) {
    for (const keyword of entry.keywords || []) {
      const clean = String(keyword).trim();
      if (!clean) continue;
      counts.set(clean, (counts.get(clean) || 0) + 1);
    }
  }
  return counts;
}

function renderKnowledgeGraph(entries) {
  knowledgeListView.hidden = true;
  knowledgeGraphView.hidden = false;
  const visibleEntries = entries.slice(0, 80);
  if (!visibleEntries.length) {
    knowledgeGraph.innerHTML = "";
    knowledgeGraphDetail.innerHTML = `<div class="source-reader-empty">No graph nodes match this search.</div>`;
    return;
  }

  const groups = knowledgeDocumentGroups(visibleEntries);
  const keywordCounts = keywordFrequency(visibleEntries);
  const keywords = [...keywordCounts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, 42)
    .map(([keyword]) => keyword);
  const keywordSet = new Set(keywords);
  const width = 1160;
  const rowHeight = 64;
  const height = Math.max(520, Math.max(groups.length, visibleEntries.length, keywords.length) * rowHeight + 80);
  const docX = 130;
  const entryX = 560;
  const keywordX = 1000;
  const docY = new Map();
  const entryY = new Map();
  const keywordY = new Map();

  groups.forEach((group, index) => docY.set(group.id, 60 + index * rowHeight));
  visibleEntries.forEach((entry, index) => entryY.set(String(entry.id), 60 + index * rowHeight));
  keywords.forEach((keyword, index) => keywordY.set(keyword, 60 + index * rowHeight));

  const docLinks = visibleEntries.map((entry) => {
    const y1 = docY.get(entry.document_id) || 60;
    const y2 = entryY.get(String(entry.id)) || 60;
    return `<path class="graph-link" d="M ${docX + 110} ${y1} C ${docX + 250} ${y1}, ${entryX - 180} ${y2}, ${entryX - 46} ${y2}" />`;
  }).join("");

  const keywordLinks = visibleEntries.flatMap((entry) => (entry.keywords || [])
    .filter((keyword) => keywordSet.has(keyword))
    .slice(0, 4)
    .map((keyword) => {
      const y1 = entryY.get(String(entry.id)) || 60;
      const y2 = keywordY.get(keyword) || 60;
      return `<path class="graph-link keyword-link" d="M ${entryX + 46} ${y1} C ${entryX + 190} ${y1}, ${keywordX - 210} ${y2}, ${keywordX - 86} ${y2}" />`;
    })).join("");

  const docNodes = groups.map((group) => {
    const y = docY.get(group.id);
    return `
      <g class="graph-node document-node" data-node-type="document" data-document-id="${group.id}">
        <rect x="${docX - 110}" y="${y - 22}" width="220" height="44" rx="8"></rect>
        <text x="${docX - 96}" y="${y - 3}">${escapeHtml(truncateText(group.name, 26))}</text>
        <text class="graph-subtext" x="${docX - 96}" y="${y + 14}">${group.entries.length} entries</text>
      </g>
    `;
  }).join("");

  const entryNodes = visibleEntries.map((entry) => {
    const y = entryY.get(String(entry.id));
    const active = String(entry.id) === String(selectedKnowledgeEntryId) ? " active" : "";
    return `
      <g class="graph-node entry-node${active}" data-node-type="entry" data-entry-id="${escapeHtml(entry.id)}">
        <circle cx="${entryX}" cy="${y}" r="20"></circle>
        <text x="${entryX + 30}" y="${y - 3}">${escapeHtml(truncateText(entry.information, 42))}</text>
        <text class="graph-subtext" x="${entryX + 30}" y="${y + 14}">${escapeHtml(entry.source || "source")}</text>
      </g>
    `;
  }).join("");

  const keywordNodes = keywords.map((keyword) => {
    const y = keywordY.get(keyword);
    return `
      <g class="graph-node keyword-node" data-node-type="keyword" data-keyword="${escapeHtml(keyword)}">
        <rect x="${keywordX - 86}" y="${y - 18}" width="172" height="36" rx="18"></rect>
        <text x="${keywordX - 72}" y="${y + 5}">${escapeHtml(truncateText(keyword, 22))}</text>
      </g>
    `;
  }).join("");

  knowledgeGraph.setAttribute("viewBox", `0 0 ${width} ${height}`);
  knowledgeGraph.innerHTML = `
    <text class="graph-label" x="${docX - 110}" y="28">Documents</text>
    <text class="graph-label" x="${entryX - 20}" y="28">Entries</text>
    <text class="graph-label" x="${keywordX - 86}" y="28">Keywords</text>
    ${docLinks}
    ${keywordLinks}
    ${docNodes}
    ${entryNodes}
    ${keywordNodes}
  `;
}

function truncateText(value, maxLength) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  return text.length > maxLength ? `${text.slice(0, maxLength - 1)}...` : text;
}

async function openDrawer(kind) {
  if (kind === "knowledge") {
    drawerKind.textContent = "Knowledge Base";
    drawerTitle.textContent = "Knowledge Base";
    drawerDocs.innerHTML = `<div class="drawer-empty">Loading Knowledge Base entries...</div>`;
    drawer.hidden = false;
    overlay.hidden = false;
    const { entries } = await api("/api/knowledge?limit=300");
    drawerDocs.innerHTML = entries.length ? entries.map((entry) => `
      <article class="drawer-doc knowledge-entry-doc">
        <header>
          <div>
            <strong>${escapeHtml(entry.document_name)}</strong>
            <span>${escapeHtml(entry.source || `chunk ${entry.chunk_index + 1}`)}</span>
          </div>
        </header>
        <div class="knowledge-keywords" aria-label="Keywords">
          <small>Keywords</small>
          ${(entry.keywords || []).map((keyword) => `<span>${escapeHtml(keyword)}</span>`).join("")}
        </div>
        <div class="chunk-body rendered">${renderMarkdown(entry.information)}</div>
      </article>
    `).join("") : `<div class="drawer-empty">No Knowledge Base entries yet.</div>`;
    return;
  }
  const summary = databaseSummary(kind);
  drawerKind.textContent = "Source Base";
  drawerTitle.textContent = "Source Base";
  drawerDocs.innerHTML = summary.docs.length ? summary.docs.map((doc) => {
    const date = new Date(doc.created_at * 1000).toLocaleString();
    const canBuildKnowledge = doc.kind === "source";
    return `
      <article class="drawer-doc">
        <header>
          <div>
            <strong>${escapeHtml(doc.name)}</strong>
            <span>${doc.chunk_count} chunks · ${date}</span>
          </div>
          <div class="drawer-actions">
            ${canBuildKnowledge ? `<button type="button" data-build-knowledge="${doc.id}">Build Knowledge Base</button>` : ""}
            <button type="button" data-delete="${doc.id}">Delete</button>
          </div>
        </header>
        <button class="chunk-toggle" type="button" data-chunks="${doc.id}">View chunks</button>
        <div class="chunk-list" id="chunks-${doc.id}" hidden></div>
      </article>
    `;
  }).join("") : `<div class="drawer-empty">No documents in this database yet.</div>`;
  drawer.hidden = false;
  overlay.hidden = false;
}

function closeDrawer() {
  drawer.hidden = true;
  overlay.hidden = true;
}

async function loadChunks(documentId) {
  const container = document.querySelector(`#chunks-${documentId}`);
  if (!container) return;
  if (!container.hidden) {
    container.hidden = true;
    return;
  }
  container.hidden = false;
  if (container.dataset.loaded) return;
  container.innerHTML = `<p class="loading">Loading chunks...</p>`;
  const { chunks } = await api(`/api/documents/${documentId}/chunks`);
  container.dataset.loaded = "true";
  container.innerHTML = chunks.map((chunk) => `
    <details class="chunk-item">
      <summary>Chunk ${chunk.chunk_index + 1}</summary>
      <div class="chunk-body rendered">${renderMarkdown(chunk.text)}</div>
    </details>
  `).join("");
}

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || "");
      resolve(result.includes(",") ? result.split(",", 2)[1] : result);
    };
    reader.onerror = () => reject(reader.error || new Error("Could not read file"));
    reader.readAsDataURL(file);
  });
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  selectedFilePayload = null;
  if (!nameInput.value.trim()) {
    nameInput.value = file.name;
  }

  if (file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")) {
    selectedFilePayload = { file_type: "pdf", file_data: await readFileAsBase64(file) };
    textInput.value = "";
    textInput.placeholder = "PDF selected. Text will be extracted when you index it.";
    return;
  }

  if (file.type.startsWith("image/")) {
    selectedFilePayload = {
      file_type: "image",
      mime_type: file.type,
      file_data: await readFileAsBase64(file),
    };
    textInput.value = "";
    textInput.placeholder = "Image selected. Vision text will be extracted, stored in Source Base, then used to build Knowledge Base entries.";
    return;
  }

  textInput.value = await file.text();
  textInput.placeholder = "Paste source text, logs, specs, or article text";
});

ingestForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = ingestForm.querySelector("button[type='submit']");
  button.disabled = true;
  button.textContent = "Indexing...";
  try {
    await api("/api/documents", {
      method: "POST",
      body: JSON.stringify({
        name: nameInput.value.trim() || "Pasted document",
        text: textInput.value,
        build_knowledge: buildKnowledgeUploadInput.checked,
        ...(selectedFilePayload || {}),
      }),
    });
    ingestForm.reset();
    selectedFilePayload = null;
    textInput.placeholder = "Paste source text, logs, specs, or article text";
    await refreshJobs();
  } catch (error) {
    window.alert(`Indexing failed: ${error.message}`);
  } finally {
    button.disabled = false;
    button.textContent = "Index";
  }
});

chatList.addEventListener("click", async (event) => {
  const renameButton = event.target.closest("[data-chat-rename]");
  if (renameButton) {
    await renameChat(renameButton.dataset.chatRename);
    return;
  }

  const deleteButton = event.target.closest("[data-chat-delete]");
  if (deleteButton) {
    await deleteChat(deleteButton.dataset.chatDelete);
    return;
  }

  const card = event.target.closest("[data-chat-id]");
  if (card) {
    await openChat(card.dataset.chatId);
  }
});

newChatButton.addEventListener("click", createChat);

drawerDocs.addEventListener("click", async (event) => {
  const chunkButton = event.target.closest("[data-chunks]");
  if (chunkButton) {
    await loadChunks(chunkButton.dataset.chunks);
    return;
  }

  const deleteButton = event.target.closest("[data-delete]");
  if (deleteButton) {
    const shouldDelete = await openChatDialog({
      title: "Delete Source Document",
      message: "Delete this Source Base document and its chunks? This also removes related Knowledge Base entries.",
      confirmLabel: "Delete",
      danger: true,
    });
    if (!shouldDelete) return;
    await api(`/api/documents/${deleteButton.dataset.delete}`, { method: "DELETE" });
    await refreshDocuments();
    await refreshHealth();
    openDrawer("source");
    await refreshHealth();
    return;
  }

  const buildKnowledgeButton = event.target.closest("[data-build-knowledge]");
  if (buildKnowledgeButton) {
    buildKnowledgeButton.disabled = true;
    buildKnowledgeButton.textContent = "Working...";
    try {
      await api(`/api/documents/${buildKnowledgeButton.dataset.buildKnowledge}/build-knowledge`, { method: "POST" });
      await refreshJobs();
      openDrawer("source");
    } catch (error) {
      window.alert(`Knowledge Base build failed: ${error.message}`);
      buildKnowledgeButton.disabled = false;
      buildKnowledgeButton.textContent = "Build Knowledge Base";
    }
  }
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("user", question);
  questionInput.value = "";
  const button = askForm.querySelector("button[type='submit']");
  button.disabled = true;
  button.textContent = "Thinking...";

  try {
    const result = await api("/api/ask", {
      method: "POST",
      body: JSON.stringify({
        question,
        top_k: Number(topKInput.value || 6),
        chat_id: activeChatId,
      }),
    });
    activeChatId = Number(result.chat_id || activeChatId);
    addMessage("assistant", result.answer, result.sources);
    await refreshChats();
  } catch (error) {
    addMessage("assistant", `Answer failed: ${error.message}`);
  } finally {
    button.disabled = false;
    button.textContent = "Ask";
    questionInput.focus();
  }
});

closeDrawerButton.addEventListener("click", closeDrawer);
overlay.addEventListener("click", closeDrawer);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !drawer.hidden) {
    closeDrawer();
  }
});
chatViewButton.addEventListener("click", () => setWorkspaceView("chat"));
knowledgeViewButton.addEventListener("click", () => setWorkspaceView("knowledge"));
sourceViewButton.addEventListener("click", () => setWorkspaceView("source"));
knowledgeRefreshButton.addEventListener("click", refreshKnowledgeEntries);
knowledgeSearchInput.addEventListener("input", renderKnowledgePage);
knowledgeListModeButton.addEventListener("click", () => {
  knowledgeMode = "list";
  knowledgeListModeButton.classList.add("active");
  knowledgeGraphModeButton.classList.remove("active");
  renderKnowledgePage();
});
knowledgeGraphModeButton.addEventListener("click", () => {
  knowledgeMode = "graph";
  knowledgeGraphModeButton.classList.add("active");
  knowledgeListModeButton.classList.remove("active");
  renderKnowledgePage();
});
knowledgeEntryList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-knowledge-entry]");
  if (button) {
    openKnowledgeEntry(button.dataset.knowledgeEntry);
  }
});
knowledgeGraph.addEventListener("click", (event) => {
  const node = event.target.closest(".graph-node");
  if (!node) return;
  if (node.dataset.nodeType === "entry") {
    openKnowledgeEntry(node.dataset.entryId);
    return;
  }
  if (node.dataset.nodeType === "document") {
    const entries = knowledgeEntriesCache.filter((entry) => String(entry.document_id) === String(node.dataset.documentId));
    knowledgeGraphDetail.innerHTML = `
      <header class="source-reader-head">
        <div>
          <p class="eyebrow">Document node</p>
          <h3>${escapeHtml(entries[0]?.document_name || "Document")}</h3>
          <span>${entries.length} Knowledge Base entries</span>
        </div>
      </header>
      <div class="graph-detail-list">
        ${entries.slice(0, 20).map((entry) => `
          <button type="button" data-knowledge-entry="${escapeHtml(entry.id)}">
            ${escapeHtml(truncateText(entry.information, 110))}
          </button>
        `).join("")}
      </div>
    `;
    return;
  }
  if (node.dataset.nodeType === "keyword") {
    const keyword = node.dataset.keyword;
    const entries = knowledgeEntriesCache.filter((entry) => (entry.keywords || []).includes(keyword));
    knowledgeGraphDetail.innerHTML = `
      <header class="source-reader-head">
        <div>
          <p class="eyebrow">Keyword node</p>
          <h3>${escapeHtml(keyword)}</h3>
          <span>${entries.length} connected entries</span>
        </div>
      </header>
      <div class="graph-detail-list">
        ${entries.slice(0, 20).map((entry) => `
          <button type="button" data-knowledge-entry="${escapeHtml(entry.id)}">
            ${escapeHtml(truncateText(entry.information, 110))}
          </button>
        `).join("")}
      </div>
    `;
  }
});
knowledgeGraphDetail.addEventListener("click", (event) => {
  const button = event.target.closest("[data-knowledge-entry]");
  if (button) {
    openKnowledgeEntry(button.dataset.knowledgeEntry);
  }
});
sourceRefreshButton.addEventListener("click", refreshDatabases);
sourceSearchInput.addEventListener("input", renderSourcePage);
sourceDocumentList.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-source-doc]");
  if (button) {
    await openSourceDocument(button.dataset.sourceDoc);
  }
});
sourceReader.addEventListener("click", async (event) => {
  const deleteButton = event.target.closest("[data-delete-source]");
  if (deleteButton) {
    const shouldDelete = await openChatDialog({
      title: "Delete Source Document",
      message: "Delete this Source Base document and its chunks? This also removes related Knowledge Base entries.",
      confirmLabel: "Delete",
      danger: true,
    });
    if (!shouldDelete) return;
    await api(`/api/documents/${deleteButton.dataset.deleteSource}`, { method: "DELETE" });
    selectedSourceDocumentId = null;
    sourceReader.innerHTML = `<div class="source-reader-empty">Select a document to read its chunks.</div>`;
    await refreshDatabases();
    return;
  }
  const buildButton = event.target.closest("[data-build-knowledge]");
  if (buildButton) {
    buildButton.disabled = true;
    buildButton.textContent = "Working...";
    try {
      await api(`/api/documents/${buildButton.dataset.buildKnowledge}/build-knowledge`, { method: "POST" });
      await refreshJobs();
    } catch (error) {
      buildButton.disabled = false;
      buildButton.textContent = "Build Knowledge Base";
      sourceReader.insertAdjacentHTML("afterbegin", `<div class="inline-error">${escapeHtml(error.message)}</div>`);
    }
  }
});
refreshJobsButton.addEventListener("click", refreshJobs);
rebuildKnowledgeButton.addEventListener("click", () => queueMaintenance("/api/rebuild-knowledge", "Knowledge Base rebuild"));
settingsToggle.addEventListener("click", async () => {
  settingsPanel.hidden = !settingsPanel.hidden;
  if (!settingsPanel.hidden) {
    await refreshSettings();
    await refreshHealth();
  }
});
reloadSettingsButton.addEventListener("click", async () => {
  reloadSettingsButton.disabled = true;
  reloadSettingsButton.textContent = "Reloading...";
  try {
    await refreshSettings();
    await refreshHealth();
  } finally {
    reloadSettingsButton.disabled = false;
    reloadSettingsButton.textContent = "Reload saved";
  }
});
checkLlmButton.addEventListener("click", async () => {
  checkLlmButton.disabled = true;
  checkLlmButton.textContent = "Checking...";
  setStatusChip(llmStatus, "Checking", "muted");
  try {
    const status = await api("/api/connections");
    updateConnectionBadges(status);
  } catch (error) {
    setStatusChip(llmStatus, "Check failed", "error");
  } finally {
    checkLlmButton.disabled = false;
    checkLlmButton.textContent = "Check local app";
  }
});
checkQdrantButton.addEventListener("click", async () => {
  checkQdrantButton.disabled = true;
  checkQdrantButton.textContent = "Checking...";
  setStatusChip(qdrantStatus, "Checking", "muted");
  try {
    const status = await api("/api/connections");
    qdrantInfo = status.qdrant || {};
    updateConnectionBadges(status);
    renderDatabases();
    renderKnowledgePage();
  } catch (error) {
    setStatusChip(qdrantStatus, "Check failed", "error");
  } finally {
    checkQdrantButton.disabled = false;
    checkQdrantButton.textContent = "Check Qdrant";
  }
});
settingsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = settingsForm.querySelector("button[type='submit']");
  button.disabled = true;
  button.textContent = "Applying...";
  try {
    const settings = await api("/api/settings", {
      method: "POST",
      body: JSON.stringify({
        base_url: settingBaseUrl.value.trim(),
        model: settingModel.value.trim(),
        image_model: settingImageModel.value.trim(),
        embedding_model: settingEmbeddingModel.value.trim(),
        vector_backend: settingVectorBackend.value,
        qdrant_url: settingQdrantUrl.value.trim(),
        qdrant_collection: settingQdrantCollection.value.trim(),
        qdrant_source_collection: settingQdrantSourceCollection.value.trim(),
        knowledge_concurrency: Number(settingConcurrency.value || 1),
        knowledge_group_chunks: Number(settingGroupChunks.value || 30),
        knowledge_max_tokens: Number(settingMaxTokens.value || 12000),
        source_chunk_max_words: Number(settingSourceChunkWords.value || 650),
        extraction_workers: Number(settingExtractionWorkers.value || 12),
        rerank_enabled: settingRerank.checked,
      }),
    });
    fillSettings(settings);
    await refreshHealth();
  } catch (error) {
    window.alert(`Settings failed: ${error.message}`);
  } finally {
    button.disabled = false;
    button.textContent = "Apply changes";
  }
});

async function init() {
  setEmptyState();
  await refreshHealth();
  await refreshDocuments();
  await refreshChats();
  refreshJobs();
  refreshSettings();
}

init();
