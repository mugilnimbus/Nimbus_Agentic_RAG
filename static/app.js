const statusEl = document.querySelector("#status");
const modelPill = document.querySelector("#modelPill");
const dbList = document.querySelector("#dbList");
const ingestForm = document.querySelector("#ingestForm");
const askForm = document.querySelector("#askForm");
const messages = document.querySelector("#messages");
const fileInput = document.querySelector("#docFile");
const textInput = document.querySelector("#docText");
const nameInput = document.querySelector("#docName");
const questionInput = document.querySelector("#question");
const topKInput = document.querySelector("#topK");
const distillUploadInput = document.querySelector("#distillUpload");
const jobList = document.querySelector("#jobList");
const refreshJobsButton = document.querySelector("#refreshJobs");
const rebuildNotesButton = document.querySelector("#rebuildNotes");
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
const settingConcurrency = document.querySelector("#settingConcurrency");
const settingGroupChunks = document.querySelector("#settingGroupChunks");
const settingMaxTokens = document.querySelector("#settingMaxTokens");
const settingExtractionWorkers = document.querySelector("#settingExtractionWorkers");
const settingRerank = document.querySelector("#settingRerank");
const drawer = document.querySelector("#dbDrawer");
const overlay = document.querySelector("#dbOverlay");
const drawerKind = document.querySelector("#drawerKind");
const drawerTitle = document.querySelector("#drawerTitle");
const drawerDocs = document.querySelector("#drawerDocs");
const closeDrawerButton = document.querySelector("#closeDrawer");

let selectedFilePayload = null;
let documentsCache = [];
let jobsCache = [];
let jobPollTimer = null;
let qdrantInfo = {};

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

function setEmptyState() {
  if (!messages.children.length) {
    messages.innerHTML = `
      <div class="empty">
        <div class="empty-mark">N</div>
        <h3>Ready when you are</h3>
        <p>Upload files, build Qdrant records, and ask from your local knowledge base.</p>
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
    if (documentsCache.length || dbList.children.length) {
      renderDatabases();
    }
  } catch (error) {
    statusEl.textContent = "Server unavailable";
    statusEl.classList.add("error");
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
    const result = await api(path, { method: "POST", body: "{}" });
    addMessage("assistant", `${label} queued as job ${result.job_id}.`);
    await refreshJobs();
  } catch (error) {
    addMessage("assistant", `${label} failed to queue: ${error.message}`);
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
  settingConcurrency.value = settings.notes_concurrency || 1;
  settingGroupChunks.value = settings.notes_group_chunks || 30;
  settingMaxTokens.value = settings.notes_max_tokens || 12000;
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
  return kind === "notes" || kind === "record" ? "record" : (kind || "source");
}

function renderDatabases() {
  const raw = databaseSummary("raw");
  const recordCount = Number(qdrantInfo.points_count || 0);
  dbList.innerHTML = [
    {
      kind: "records",
      title: "Qdrant Records",
      body: "Keyword vectors and readable distilled facts stored in Qdrant.",
      docs: 0,
      chunks: recordCount,
    },
    {
      kind: "raw",
      title: "Raw SQLite",
      body: "Cleaned source text used as fallback and confirmation evidence.",
      docs: raw.docs.length,
      chunks: raw.chunks,
    },
  ].map((db) => `
    <button class="db-card" type="button" data-db="${db.kind}">
      <span class="db-icon">${db.kind === "records" ? "Q" : "R"}</span>
      <span>
        <strong>${db.title}</strong>
        <small>${db.body}</small>
      </span>
      <em>${db.kind === "records" ? `${db.chunks} records` : `${db.docs} docs · ${db.chunks} chunks`}</em>
    </button>
  `).join("");
}

async function refreshDocuments() {
  const { documents } = await api("/api/documents");
  documentsCache = documents;
  renderDatabases();
}

async function refreshDatabases() {
  await refreshHealth();
  await refreshDocuments();
}

async function openDrawer(kind) {
  if (kind === "records") {
    drawerKind.textContent = "Qdrant vector database";
    drawerTitle.textContent = "Distilled Records";
    drawerDocs.innerHTML = `<div class="drawer-empty">Loading records...</div>`;
    drawer.hidden = false;
    overlay.hidden = false;
    const { records } = await api("/api/records?limit=300");
    drawerDocs.innerHTML = records.length ? records.map((record) => `
      <article class="drawer-doc record-doc">
        <header>
          <div>
            <strong>${escapeHtml(record.document_name)}</strong>
            <span>${escapeHtml(record.source || `chunk ${record.chunk_index + 1}`)}</span>
          </div>
        </header>
        <div class="record-keywords">${(record.keywords || []).map((keyword) => `<span>${escapeHtml(keyword)}</span>`).join("")}</div>
        <div class="chunk-body rendered">${renderMarkdown(record.information || record.text)}</div>
      </article>
    `).join("") : `<div class="drawer-empty">No Qdrant records yet.</div>`;
    return;
  }
  const summary = databaseSummary(kind);
  drawerKind.textContent = "Raw SQLite database";
  drawerTitle.textContent = "Raw SQLite";
  drawerDocs.innerHTML = summary.docs.length ? summary.docs.map((doc) => {
    const date = new Date(doc.created_at * 1000).toLocaleString();
    const canDistill = doc.kind === "raw";
    return `
      <article class="drawer-doc">
        <header>
          <div>
            <strong>${escapeHtml(doc.name)}</strong>
            <span>${doc.chunk_count} chunks · ${date}</span>
          </div>
          <div class="drawer-actions">
            ${canDistill ? `<button type="button" data-distill="${doc.id}">Build records</button>` : ""}
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
    textInput.placeholder = "Image selected. Vision text will be extracted, stored raw, then distilled.";
    return;
  }

  textInput.value = await file.text();
  textInput.placeholder = "Paste notes, logs, specs, or article text";
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
        distill: distillUploadInput.checked,
        ...(selectedFilePayload || {}),
      }),
    });
    addMessage("assistant", "Indexing queued. I will update the database counts when the background job finishes.");
    ingestForm.reset();
    selectedFilePayload = null;
    textInput.placeholder = "Paste notes, logs, specs, or article text";
    await refreshJobs();
  } catch (error) {
    addMessage("assistant", `Indexing failed: ${error.message}`);
  } finally {
    button.disabled = false;
    button.textContent = "Index";
  }
});

dbList.addEventListener("click", (event) => {
  const card = event.target.closest("[data-db]");
  if (card) openDrawer(card.dataset.db);
});

drawerDocs.addEventListener("click", async (event) => {
  const chunkButton = event.target.closest("[data-chunks]");
  if (chunkButton) {
    await loadChunks(chunkButton.dataset.chunks);
    return;
  }

  const deleteButton = event.target.closest("[data-delete]");
  if (deleteButton) {
    await api(`/api/documents/${deleteButton.dataset.delete}`, { method: "DELETE" });
    await refreshDocuments();
    await refreshHealth();
    openDrawer("raw");
    await refreshHealth();
    return;
  }

  const distillButton = event.target.closest("[data-distill]");
  if (distillButton) {
    distillButton.disabled = true;
    distillButton.textContent = "Working...";
    try {
      await api(`/api/documents/${distillButton.dataset.distill}/distill`, { method: "POST" });
      addMessage("assistant", "Qdrant record build queued.");
      await refreshJobs();
      openDrawer("raw");
    } catch (error) {
      addMessage("assistant", `Qdrant record build failed: ${error.message}`);
      distillButton.disabled = false;
      distillButton.textContent = "Build records";
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
      body: JSON.stringify({ question, top_k: Number(topKInput.value || 6) }),
    });
    addMessage("assistant", result.answer, result.sources);
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
document.querySelector("#refreshDocs").addEventListener("click", refreshDatabases);
refreshJobsButton.addEventListener("click", refreshJobs);
rebuildNotesButton.addEventListener("click", () => queueMaintenance("/api/rebuild-records", "Qdrant record rebuild"));
settingsToggle.addEventListener("click", async () => {
  settingsPanel.hidden = !settingsPanel.hidden;
  if (!settingsPanel.hidden) {
    await refreshSettings();
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
        notes_concurrency: Number(settingConcurrency.value || 1),
        notes_group_chunks: Number(settingGroupChunks.value || 30),
        notes_max_tokens: Number(settingMaxTokens.value || 12000),
        extraction_workers: Number(settingExtractionWorkers.value || 12),
        rerank_enabled: settingRerank.checked,
      }),
    });
    fillSettings(settings);
    await refreshHealth();
    addMessage("assistant", "Settings applied. New jobs and questions will use the updated parameters.");
  } catch (error) {
    addMessage("assistant", `Settings failed: ${error.message}`);
  } finally {
    button.disabled = false;
    button.textContent = "Apply";
  }
});

async function init() {
  setEmptyState();
  await refreshHealth();
  await refreshDocuments();
  refreshJobs();
  refreshSettings();
}

init();
