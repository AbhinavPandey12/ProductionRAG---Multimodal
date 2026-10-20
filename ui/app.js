// Financial RAG UI Application Logic

// Auto-detect API Base URL (FastAPI backend)
const API_BASE = window.location.origin.startsWith("http")
  ? window.location.origin
  : "http://localhost:8080";

// DOM Elements
const tabBtns = document.querySelectorAll(".tab-btn");
const tabPanes = document.querySelectorAll(".tab-pane");
const statusBadge = document.getElementById("statusBadge");
const statusText = document.getElementById("statusText");
const targetBucketBadge = document.getElementById("targetBucketBadge");

// Thread & Chat Feed Elements
const currentThreadDisplay = document.getElementById("currentThreadDisplay");
const copyThreadBtn = document.getElementById("copyThreadBtn");
const threadSelect = document.getElementById("threadSelect");
const newChatBtn = document.getElementById("newChatBtn");
const chatFeed = document.getElementById("chatFeed");

// Query Elements
const queryForm = document.getElementById("queryForm");
const queryInputWrap = document.getElementById("queryInputWrap");
const questionInput = document.getElementById("questionInput");
const submitBtn = document.getElementById("submitBtn");
const docFilterSelect = document.getElementById("docFilterSelect");
const limitSelect = document.getElementById("limitSelect");
const modelSelect = document.getElementById("modelSelect");

// Image / Screenshot Attachment Elements
const attachedImageContainer = document.getElementById("attachedImageContainer");
const attachedImagePreview = document.getElementById("attachedImagePreview");
const attachedImageName = document.getElementById("attachedImageName");
const attachedImageSize = document.getElementById("attachedImageSize");
const removeAttachedImageBtn = document.getElementById("removeAttachedImageBtn");
const attachImageBtn = document.getElementById("attachImageBtn");
const screenshotFileInput = document.getElementById("screenshotFileInput");

// Upload Elements (Multi-Format & Bulk)
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const fileStagingArea = document.getElementById("fileStagingArea");
const stagedCountBadge = document.getElementById("stagedCountBadge");
const stagedFilesList = document.getElementById("stagedFilesList");
const clearAllFilesBtn = document.getElementById("clearAllFilesBtn");
const uploadSubmitBtn = document.getElementById("uploadSubmitBtn");
const uploadStatusBox = document.getElementById("uploadStatusBox");

// Modal Elements
const imageModal = document.getElementById("imageModal");
const modalImageSrc = document.getElementById("modalImageSrc");
const modalImageTitle = document.getElementById("modalImageTitle");
const modalCloseBtn = document.getElementById("modalCloseBtn");

let stagedFiles = [];
let currentThreadId = generateUUID();
let currentAttachedImageBase64 = null;
let currentAttachedImageName = "";

// Configure Marked for GitHub Flavored Markdown & Code Highlighting
if (window.marked) {
  marked.setOptions({
    gfm: true,
    breaks: true,
    highlight: function(code, lang) {
      if (window.hljs && lang && hljs.getLanguage(lang)) {
        try {
          return hljs.highlight(code, { language: lang }).value;
        } catch (err) {}
      }
      return code;
    }
  });
}

/**
 * Renders Markdown content and LaTeX math equations ($...$ inline and $$...$$ block)
 * using KaTeX with protection against Markdown character collisions (like _ and * in equations).
 */
function renderMarkdownWithMath(rawText) {
  if (!rawText) return "";

  // If KaTeX is not loaded on window, fallback to marked.parse and let MathJax typeset
  if (typeof katex === "undefined" || !window.katex) {
    console.warn("[Academic RAG] KaTeX not yet loaded; using marked.parse (MathJax will typeset DOM elements).");
    return marked.parse(rawText);
  }

  const mathPlaceholders = [];

  // Step 1: Protect code blocks (fenced ```...``` and inline `...`)
  const codeBlocks = [];
  let processed = rawText.replace(/(```[\s\S]*?```|`[^`\n]+`)/g, (match) => {
    const placeholder = `%%CODE_BLOCK_${codeBlocks.length}%%`;
    codeBlocks.push(match);
    return placeholder;
  });

  // Step 2: Extract block/display math $$...$$ and \[...\]
  processed = processed.replace(/\$\$([\s\S]*?)\$\$/g, (match, equation) => {
    const placeholder = `%%MATH_BLOCK_${mathPlaceholders.length}%%`;
    let rendered = "";
    try {
      rendered = `<div class="katex-display-wrapper">${katex.renderToString(equation.trim(), {
        displayMode: true,
        throwOnError: false
      })}</div>`;
    } catch (err) {
      rendered = `<div class="katex-error">${escapeHtml(match)}</div>`;
    }
    mathPlaceholders.push({ placeholder, rendered, isBlock: true });
    return placeholder;
  });

  processed = processed.replace(/\\\[([\s\S]*?)\\\]/g, (match, equation) => {
    const placeholder = `%%MATH_BLOCK_${mathPlaceholders.length}%%`;
    let rendered = "";
    try {
      rendered = `<div class="katex-display-wrapper">${katex.renderToString(equation.trim(), {
        displayMode: true,
        throwOnError: false
      })}</div>`;
    } catch (err) {
      rendered = `<div class="katex-error">${escapeHtml(match)}</div>`;
    }
    mathPlaceholders.push({ placeholder, rendered, isBlock: true });
    return placeholder;
  });

  // Step 3: Extract inline math $...$ and \(...\)
  processed = processed.replace(/\\\(([\s\S]*?)\\\)/g, (match, equation) => {
    const placeholder = `%%MATH_INLINE_${mathPlaceholders.length}%%`;
    let rendered = "";
    try {
      rendered = katex.renderToString(equation.trim(), {
        displayMode: false,
        throwOnError: false
      });
    } catch (err) {
      rendered = escapeHtml(match);
    }
    mathPlaceholders.push({ placeholder, rendered, isBlock: false });
    return placeholder;
  });

  processed = processed.replace(/(?<!\\)\$([^\$\n]+?)(?<!\\)\$/g, (match, equation) => {
    // Avoid false positives for standalone currency numbers (e.g. $10, $5.99)
    if (/^\s*\d+([.,]\d+)?\s*$/.test(equation)) {
      return match;
    }
    const placeholder = `%%MATH_INLINE_${mathPlaceholders.length}%%`;
    let rendered = "";
    try {
      rendered = katex.renderToString(equation.trim(), {
        displayMode: false,
        throwOnError: false
      });
    } catch (err) {
      rendered = escapeHtml(match);
    }
    mathPlaceholders.push({ placeholder, rendered, isBlock: false });
    return placeholder;
  });

  // Step 4: Restore code blocks before marked runs
  codeBlocks.forEach((code, idx) => {
    processed = processed.replace(`%%CODE_BLOCK_${idx}%%`, () => code);
  });

  // Step 5: Parse Markdown into HTML
  let html = marked.parse(processed);

  // Step 6: Restore rendered math into HTML
  mathPlaceholders.forEach(({ placeholder, rendered, isBlock }) => {
    if (isBlock) {
      // Remove any <p> tag that marked may have wrapped around the block placeholder
      const pWrapped = new RegExp(`<p>\\s*${placeholder}\\s*<\\/p>`, "g");
      html = html.replace(pWrapped, rendered);
    }
    html = html.replace(placeholder, () => rendered);
  });

  return html;
}

// -----------------------------------------------------------------------------
// 1. Tab Switching
// -----------------------------------------------------------------------------
tabBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    tabBtns.forEach(b => b.classList.remove("active"));
    tabPanes.forEach(p => p.classList.remove("active"));

    btn.classList.add("active");
    const targetTab = btn.getAttribute("data-tab");
    document.getElementById(targetTab).classList.add("active");
  });
});

// -----------------------------------------------------------------------------
// 2. Health Check, Document List & Threads
// -----------------------------------------------------------------------------
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      statusBadge.className = "badge badge-success";
      statusText.textContent = `Connected: S3 Vectors (${data.details?.vector_index || "active"})`;
      if (data.details?.vector_bucket) {
        targetBucketBadge.textContent = `Target Bucket: ${data.details.vector_bucket}`;
      }
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (err) {
    statusBadge.className = "badge badge-error";
    statusText.textContent = "Backend Disconnected";
    console.warn("Could not reach backend /health:", err);
  }
}

async function loadDocumentsList() {
  try {
    const res = await fetch(`${API_BASE}/documents`);
    if (res.ok) {
      const data = await res.json();
      const docs = data.documents || [];
      docFilterSelect.innerHTML = '<option value="">Agent Auto-Selects Document</option>';
      docs.forEach(doc => {
        const opt = document.createElement("option");
        opt.value = doc;
        opt.textContent = doc;
        docFilterSelect.appendChild(opt);
      });
    }
  } catch (err) {
    console.warn("Could not load /documents:", err);
  }
}

async function loadThreadsList() {
  try {
    const res = await fetch(`${API_BASE}/threads`);
    if (res.ok) {
      const data = await res.json();
      const threads = data.threads || [];
      threadSelect.innerHTML = '<option value="">Switch Saved Conversation...</option>';
      threads.forEach(tid => {
        const opt = document.createElement("option");
        opt.value = tid;
        opt.textContent = `Thread: ${tid.substring(0, 12)}...`;
        if (tid === currentThreadId) {
          opt.selected = true;
        }
        threadSelect.appendChild(opt);
      });
    }
  } catch (err) {
    console.warn("Could not load /threads:", err);
  }
}

// -----------------------------------------------------------------------------
// 3. Thread Management & Prompt Helpers
// -----------------------------------------------------------------------------
function generateUUID() {
  if (crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function(c) {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

function updateThreadDisplay() {
  if (currentThreadDisplay) {
    currentThreadDisplay.textContent = currentThreadId;
    currentThreadDisplay.title = `Full Thread ID: ${currentThreadId}`;
  }
}

copyThreadBtn.addEventListener("click", () => {
  navigator.clipboard.writeText(currentThreadId);
  const originalTitle = copyThreadBtn.title;
  copyThreadBtn.title = "Copied!";
  copyThreadBtn.innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2">
      <polyline points="20 6 9 17 4 12"></polyline>
    </svg>
  `;
  setTimeout(() => {
    copyThreadBtn.title = originalTitle;
    copyThreadBtn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
    `;
  }, 2000);
});

newChatBtn.addEventListener("click", () => {
  startNewConversation();
});

function startNewConversation() {
  currentThreadId = generateUUID();
  updateThreadDisplay();
  threadSelect.value = "";
  chatFeed.innerHTML = `
    <div class="welcome-card card">
      <h3>👋 Welcome to Academic & Engineering RAG Agent</h3>
      <p>I am an autonomous tutor equipped with <strong>Amazon S3 Vectors</strong> retrieval and <strong>SQLite Conversation Memory</strong> for academic and engineering documents.</p>
      <div class="capabilities-grid">
        <div class="capability-item">
          <span class="cap-badge cap-direct">⚡ Direct Answer</span>
          <p>Ask conceptual questions, math formulas, or code generation without wasting vector searches.</p>
          <div class="prompt-chip" onclick="fillPrompt('Explain the difference between a BJT and a MOSFET.')">
            "BJT vs MOSFET comparison"
          </div>
        </div>
        <div class="capability-item">
          <span class="cap-badge cap-tool">🔍 S3 Vectors Search</span>
          <p>Ask about specific textbooks (Electronics, Programming in C, Fluid Mechanics, etc.).</p>
          <div class="prompt-chip" onclick="fillPrompt('Explain the V-I characteristics of a PN junction diode from Electronic Devices and Circuits. Display key formulas in a table.')">
            "PN Diode V-I characteristics table"
          </div>
        </div>
      </div>
    </div>
  `;
}

threadSelect.addEventListener("change", async () => {
  const selectedThread = threadSelect.value;
  if (!selectedThread) return;
  currentThreadId = selectedThread;
  updateThreadDisplay();

  // Load conversation history from backend
  try {
    chatFeed.innerHTML = `
      <div class="agent-thinking" style="padding: 2rem; justify-content: center;">
        <div class="spinner"></div>
        <span>Loading thread history...</span>
      </div>
    `;
    const res = await fetch(`${API_BASE}/threads/${selectedThread}/history`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const messages = data.messages || [];

    chatFeed.innerHTML = "";
    if (messages.length === 0) {
      chatFeed.innerHTML = '<p class="text-muted" style="text-align: center; padding: 2rem;">No messages found in this conversation.</p>';
      return;
    }

    messages.forEach(msg => {
      if (msg.role === "user") {
        renderUserMessage(msg.content, msg.user_image);
      } else {
        renderAgentMessage({
          answer: msg.content,
          tool_called: false,
          sources: [],
          images: [],
          thread_id: selectedThread
        });
      }
    });
    scrollToBottom();
  } catch (err) {
    alert(`Could not load thread history: ${err.message}`);
  }
});

// Prompt chip helper
window.fillPrompt = function(promptText) {
  questionInput.value = promptText;
  questionInput.focus();
};

// -----------------------------------------------------------------------------
// 4. Screenshot / Image Attachment & Clipboard Paste Logic
// -----------------------------------------------------------------------------
function handleImageAttachment(file) {
  if (!file || !file.type.startsWith("image/")) {
    alert("Please select or paste a valid image (PNG, JPG, WebP, etc.)");
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    currentAttachedImageBase64 = e.target.result;
    currentAttachedImageName = file.name || "Pasted Screenshot";

    attachedImagePreview.src = currentAttachedImageBase64;
    attachedImageName.textContent = currentAttachedImageName;
    attachedImageSize.textContent = formatBytes(file.size);
    attachedImageContainer.classList.remove("hidden");
    questionInput.focus();
  };
  reader.readAsDataURL(file);
}

function clearAttachedImage() {
  currentAttachedImageBase64 = null;
  currentAttachedImageName = "";
  attachedImagePreview.src = "";
  attachedImageContainer.classList.add("hidden");
  screenshotFileInput.value = "";
}

removeAttachedImageBtn.addEventListener("click", clearAttachedImage);
attachImageBtn.addEventListener("click", () => screenshotFileInput.click());

screenshotFileInput.addEventListener("change", () => {
  if (screenshotFileInput.files && screenshotFileInput.files.length > 0) {
    handleImageAttachment(screenshotFileInput.files[0]);
  }
});

attachedImagePreview.addEventListener("click", () => {
  if (currentAttachedImageBase64) {
    openModal(currentAttachedImageBase64, currentAttachedImageName);
  }
});

// Clipboard Paste Handler (Ctrl+V / Cmd+V)
function handlePaste(e) {
  const items = (e.clipboardData || window.clipboardData)?.items;
  if (!items) return;
  for (let i = 0; i < items.length; i++) {
    if (items[i].type.indexOf("image") !== -1) {
      const file = items[i].getAsFile();
      if (file) {
        e.preventDefault();
        handleImageAttachment(file);
        return;
      }
    }
  }
}
window.addEventListener("paste", handlePaste);

// Drag & Drop onto Query Box
["dragenter", "dragover"].forEach(eventName => {
  queryInputWrap.addEventListener(eventName, (e) => {
    e.preventDefault();
    queryInputWrap.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach(eventName => {
  queryInputWrap.addEventListener(eventName, (e) => {
    e.preventDefault();
    queryInputWrap.classList.remove("dragover");
  });
});

queryInputWrap.addEventListener("drop", (e) => {
  const files = e.dataTransfer.files;
  if (files && files.length > 0) {
    const file = files[0];
    if (file.type.startsWith("image/")) {
      handleImageAttachment(file);
    }
  }
});

// -----------------------------------------------------------------------------
// 5. Autonomous Agent Chat & Multi-Turn Logic
// -----------------------------------------------------------------------------
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    queryForm.dispatchEvent(new Event("submit"));
  }
});

queryForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  const attachedImg = currentAttachedImageBase64;

  if (!question && !attachedImg) return;

  const docFilter = docFilterSelect.value || null;
  const limit = parseInt(limitSelect.value, 10) || 3;
  const selectedModel = modelSelect ? modelSelect.value : "nova";

  // Clear input & attached image preview
  questionInput.value = "";
  clearAttachedImage();

  // Remove welcome card if present
  const welcomeCard = document.querySelector(".welcome-card");
  if (welcomeCard) {
    welcomeCard.remove();
  }

  // 1. Render User Message Bubble (with screenshot preview if attached)
  renderUserMessage(question, attachedImg);

  // 2. Render Temporary Thinking Bubble
  const thinkingBubble = renderThinkingBubble();
  scrollToBottom();

  // UI state: disable submit
  submitBtn.disabled = true;

  try {
    const payload = {
      question: question || "Please analyze and explain this attached screenshot/image.",
      thread_id: currentThreadId,
      limit: limit,
      document_name: docFilter,
      image_base64: attachedImg,
      model_provider: selectedModel,
      llm_model: selectedModel === "nova" ? "amazon.nova-2-lite-v1:0" : "gemini-3.1-flash-lite"
    };

    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(errData.detail || `Server error ${res.status}`);
    }

    const data = await res.json();

    // Remove thinking bubble
    if (thinkingBubble && thinkingBubble.parentNode) {
      thinkingBubble.remove();
    }

    // Update active thread ID if assigned/returned
    if (data.thread_id) {
      currentThreadId = data.thread_id;
      updateThreadDisplay();
    }

    // 3. Render Agent Response Message
    renderAgentMessage(data);
    scrollToBottom();

    // Refresh threads list
    loadThreadsList();

  } catch (err) {
    if (thinkingBubble && thinkingBubble.parentNode) {
      thinkingBubble.remove();
    }
    renderErrorMessage(`Error: ${err.message}`);
    scrollToBottom();
  } finally {
    submitBtn.disabled = false;
    questionInput.focus();
  }
});

function renderUserMessage(text, imageSrc) {
  const msgEl = document.createElement("div");
  msgEl.className = "chat-msg chat-msg-user";

  const imageHtml = imageSrc ? `
    <div class="user-attached-image" onclick="openModal('${imageSrc}', 'User Attached Screenshot')">
      <img src="${imageSrc}" alt="Attached screenshot">
    </div>
  ` : "";

  msgEl.innerHTML = `
    <div class="chat-avatar chat-avatar-user" title="You">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
        <circle cx="12" cy="7" r="4"></circle>
      </svg>
    </div>
    <div class="chat-bubble">
      <div class="chat-header-row">
        <div class="chat-sender-meta">
          <span class="chat-sender-name">You</span>
        </div>
        <span class="chat-time">${formatTime(new Date())}</span>
      </div>
      <div class="markdown-content">
        ${text ? `<p>${escapeHtml(text)}</p>` : ""}
        ${imageHtml}
      </div>
    </div>
  `;
  chatFeed.appendChild(msgEl);
}

function renderThinkingBubble() {
  const el = document.createElement("div");
  el.className = "chat-msg chat-msg-agent";
  el.id = "activeThinkingBubble";
  el.innerHTML = `
    <div class="chat-avatar chat-avatar-agent">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"></circle>
        <path d="M12 16v-4"></path>
        <path d="M12 8h.01"></path>
      </svg>
    </div>
    <div class="chat-bubble">
      <div class="agent-thinking">
        <div class="typing-dots">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>
        <span>Agent analyzing query & evaluating vector search necessity...</span>
      </div>
    </div>
  `;
  chatFeed.appendChild(el);
  return el;
}

function renderAgentMessage(data) {
  const msgEl = document.createElement("div");
  msgEl.className = "chat-msg chat-msg-agent";

  const isTool = !!data.tool_called;
  const sources = data.sources || [];
  const images = data.images || [];

  let badgeHtml = "";
  if (isTool) {
    if (sources.length > 0) {
      badgeHtml = `<span class="decision-badge decision-tool" title="Agent retrieved matching context from Amazon S3 Vectors">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
          S3 Vectors (${data.pages_retrieved || sources.length} pages)
        </span>`;
    } else {
      badgeHtml = `<span class="decision-badge decision-tool" style="background: rgba(245, 158, 11, 0.15); color: #fbbf24; border-color: rgba(245, 158, 11, 0.3);" title="Searched Amazon S3 Vectors (no document matches found); answered from general academic knowledge">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
          S3 Vectors Searched • Parametric Knowledge
        </span>`;
    }
  } else {
    badgeHtml = `<span class="decision-badge decision-direct" title="Casual conversation / greeting (no vector search needed)">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
          </svg>
          ⚡ Direct Greeting
        </span>`;
  }

  // Model provider badge
  const modelProvider = (data.model_provider || "nova").toLowerCase();
  const modelUsed = data.model_used || (modelProvider === "nova" ? "amazon.nova-2-lite-v1:0" : "gemini-3.1-flash-lite");
  let modelBadgeHtml = "";
  if (modelProvider === "nova" || modelUsed.includes("nova")) {
    modelBadgeHtml = `
      <span class="decision-badge model-badge model-badge-nova" title="Synthesized by Amazon Nova 2 Lite (AWS Bedrock Converse API)">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path>
        </svg>
        Amazon Nova 2 Lite
      </span>
    `;
  } else {
    modelBadgeHtml = `
      <span class="decision-badge model-badge model-badge-gemini" title="Synthesized by Google Gemini 3.1 Flash Lite">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <path d="M12 8v8M8 12h8"></path>
        </svg>
        Gemini 3.1 Flash Lite
      </span>
    `;
  }

  // Parse markdown with LaTeX math equations
  const answerMarkdown = renderMarkdownWithMath(data.answer || "");

  // Optional Citations Section
  let citationsHtml = "";
  if (isTool && (sources.length > 0 || images.length > 0)) {
    let metadataCards = "";
    sources.forEach((src, idx) => {
      const docName = src.document_name || "Document";
      const pageNum = src.page_number || "?";
      const totalPages = src.total_pages || "?";
      const dist = typeof src.distance === "number" ? src.distance.toFixed(4) : (src.distance || "N/A");
      const snippet = src.text_snippet || "";

      metadataCards += `
        <div class="metadata-item">
          <div class="meta-header">
            <span class="meta-doc-name" title="${escapeHtml(docName)}">${escapeHtml(docName)}</span>
            <span class="meta-badge meta-distance">Dist: ${dist}</span>
          </div>
          <div class="meta-badges">
            <span class="meta-badge">Page ${pageNum} of ${totalPages}</span>
            <span class="meta-badge">S3 Match #${idx + 1}</span>
          </div>
          ${snippet ? `<div class="meta-snippet-box"><strong>Excerpt:</strong><br>${escapeHtml(snippet)}</div>` : ""}
        </div>
      `;
    });

    let imageCards = "";
    images.forEach((b64Img, idx) => {
      const srcMeta = sources[idx] || {};
      const docLabel = srcMeta.document_name || "Page";
      const pageLabel = srcMeta.page_number ? `Page ${srcMeta.page_number}` : `View ${idx + 1}`;
      imageCards += `
        <div class="gallery-card" onclick="openModal('data:image/jpeg;base64,${b64Img}', '${escapeHtml(docLabel)} — ${pageLabel}')">
          <img src="data:image/jpeg;base64,${b64Img}" alt="${docLabel} ${pageLabel}">
          <div class="gallery-label">
            <span>${escapeHtml(docLabel)}</span>
            <strong>${pageLabel}</strong>
          </div>
        </div>
      `;
    });

    citationsHtml = `
      <details class="sources-details" open>
        <summary class="sources-summary">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
          </svg>
          Retrieved S3 Vector Context & Metadata (${sources.length} matches)
        </summary>
        <div class="sources-body">
          <div class="metadata-grid">
            ${metadataCards}
          </div>
          ${imageCards ? `<div class="image-gallery">${imageCards}</div>` : ""}
        </div>
      </details>
    `;
  }

  msgEl.innerHTML = `
    <div class="chat-avatar chat-avatar-agent" title="Academic RAG Agent">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 2a10 10 0 1 0 10 10H12V2z"></path>
        <path d="M12 12 2.1 10.4a10 10 0 0 1 9.9-8.4z"></path>
      </svg>
    </div>
    <div class="chat-bubble">
      <div class="chat-header-row">
        <div class="chat-sender-meta">
          <span class="chat-sender-name">Academic RAG Agent</span>
          ${modelBadgeHtml}
          ${badgeHtml}
        </div>
        <span class="chat-time">${formatTime(new Date())}</span>
      </div>
      <div class="markdown-content">
        ${answerMarkdown}
      </div>
      ${citationsHtml}
      <div class="chat-msg-actions">
        <button class="btn btn-secondary btn-sm copy-answer-btn">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
          </svg>
          Copy Response
        </button>
      </div>
    </div>
  `;

  // Apply Highlight.js to code blocks and attach copy buttons
  msgEl.querySelectorAll("pre").forEach(preBlock => {
    const codeEl = preBlock.querySelector("code");
    if (codeEl && window.hljs) {
      try {
        hljs.highlightElement(codeEl);
      } catch (err) {}
    }
    const copyBtn = document.createElement("button");
    copyBtn.className = "copy-code-btn";
    copyBtn.textContent = "Copy";
    copyBtn.addEventListener("click", () => {
      const code = codeEl?.innerText || preBlock.innerText;
      navigator.clipboard.writeText(code);
      copyBtn.textContent = "Copied!";
      setTimeout(() => { copyBtn.textContent = "Copy"; }, 2000);
    });
    preBlock.appendChild(copyBtn);
  });

  // Attach full answer copy button
  const copyAnsBtn = msgEl.querySelector(".copy-answer-btn");
  if (copyAnsBtn) {
    copyAnsBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(data.answer || "");
      copyAnsBtn.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2">
          <polyline points="20 6 9 17 4 12"></polyline>
        </svg> Copied!
      `;
      setTimeout(() => {
        copyAnsBtn.innerHTML = `
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
          </svg> Copy Response
        `;
      }, 2000);
    });
  }

  chatFeed.appendChild(msgEl);

  // If MathJax is loaded and any unrendered math remains, typeset the new message element
  if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
    try {
      window.MathJax.typesetPromise([msgEl]).catch(err => {
        console.warn("[Academic RAG] MathJax typesetting notice:", err);
      });
    } catch (err) {}
  }
}

function renderErrorMessage(text) {
  const el = document.createElement("div");
  el.className = "chat-msg chat-msg-agent";
  el.innerHTML = `
    <div class="chat-avatar chat-avatar-agent" style="border-color: rgba(244, 63, 94, 0.4);">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fb7185" stroke-width="2">
        <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon>
        <line x1="12" y1="8" x2="12" y2="12"></line>
        <line x1="12" y1="16" x2="12.01" y2="16"></line>
      </svg>
    </div>
    <div class="chat-bubble" style="border-color: rgba(244, 63, 94, 0.4);">
      <div class="chat-header-row">
        <span class="chat-sender-name" style="color: #fb7185;">System Error</span>
        <span class="chat-time">${formatTime(new Date())}</span>
      </div>
      <p style="color: #fecdd3; font-size: 0.92rem;">${escapeHtml(text)}</p>
    </div>
  `;
  chatFeed.appendChild(el);
}

function scrollToBottom() {
  chatFeed.scrollTop = chatFeed.scrollHeight;
}

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// 4. File Upload (Tab 2 - Multi-Format & Bulk Support)
// -----------------------------------------------------------------------------
const SUPPORTED_EXTS = [".pdf", ".docx", ".doc", ".pptx", ".ppt"];

function getFileFormatMeta(filename) {
  const ext = filename.slice(filename.lastIndexOf(".")).toLowerCase();
  if (ext === ".pdf") {
    return { label: "PDF", iconClass: "file-icon-pdf" };
  } else if (ext === ".docx" || ext === ".doc") {
    return { label: "WORD", iconClass: "file-icon-word" };
  } else if (ext === ".pptx" || ext === ".ppt") {
    return { label: "PPT", iconClass: "file-icon-ppt" };
  }
  return { label: "DOC", iconClass: "file-icon-other" };
}

dropZone.addEventListener("click", () => fileInput.click());

["dragenter", "dragover"].forEach(eventName => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach(eventName => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
  });
});

dropZone.addEventListener("drop", (e) => {
  const files = Array.from(e.dataTransfer.files || []);
  if (files.length > 0) {
    handleIncomingFiles(files);
  }
});

fileInput.addEventListener("change", () => {
  const files = Array.from(fileInput.files || []);
  if (files.length > 0) {
    handleIncomingFiles(files);
  }
});

function handleIncomingFiles(newFiles) {
  let acceptedCount = 0;
  let rejectedCount = 0;

  for (const file of newFiles) {
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!SUPPORTED_EXTS.includes(ext)) {
      rejectedCount++;
      continue;
    }
    // Deduplicate by filename + size
    const exists = stagedFiles.some(f => f.name === file.name && f.size === file.size);
    if (!exists) {
      stagedFiles.push(file);
      acceptedCount++;
    }
  }

  if (rejectedCount > 0) {
    alert(`Notice: ${rejectedCount} file(s) were skipped because only PDF (.pdf), Word (.docx, .doc), and PowerPoint (.pptx, .ppt) are supported.`);
  }

  renderStagedFiles();
  uploadStatusBox.className = "alert-box hidden";
}

function renderStagedFiles() {
  if (stagedFiles.length === 0) {
    fileStagingArea.classList.add("hidden");
    fileInput.value = "";
    return;
  }

  fileStagingArea.classList.remove("hidden");
  stagedCountBadge.textContent = `${stagedFiles.length} file${stagedFiles.length > 1 ? "s" : ""}`;

  stagedFilesList.innerHTML = "";
  stagedFiles.forEach((file, idx) => {
    const meta = getFileFormatMeta(file.name);
    const itemEl = document.createElement("div");
    itemEl.className = "staged-file-item";
    itemEl.innerHTML = `
      <div class="file-icon-box ${meta.iconClass}">${meta.label}</div>
      <div class="staged-file-meta">
        <div class="staged-file-name" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</div>
        <div class="staged-file-sub">
          <span>${formatBytes(file.size)}</span>
          <span>•</span>
          <span>Ready to upload</span>
        </div>
      </div>
      <button type="button" class="btn-icon remove-single-file-btn" data-index="${idx}" title="Remove file">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    `;
    stagedFilesList.appendChild(itemEl);
  });

  // Attach individual remove handlers
  document.querySelectorAll(".remove-single-file-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const removeIdx = parseInt(btn.getAttribute("data-index"), 10);
      if (!isNaN(removeIdx)) {
        stagedFiles.splice(removeIdx, 1);
        renderStagedFiles();
      }
    });
  });
}

clearAllFilesBtn.addEventListener("click", () => {
  stagedFiles = [];
  fileInput.value = "";
  renderStagedFiles();
});

uploadSubmitBtn.addEventListener("click", async () => {
  if (stagedFiles.length === 0) return;

  const totalToUpload = stagedFiles.length;
  uploadSubmitBtn.disabled = true;
  uploadSubmitBtn.innerHTML = `
    <div class="spinner" style="width: 18px; height: 18px; margin: 0; border-width: 2px;"></div>
    <span>Uploading ${totalToUpload} file(s) to S3 & SQS...</span>
  `;

  uploadStatusBox.className = "alert-box hidden";

  try {
    const formData = new FormData();
    for (const file of stagedFiles) {
      formData.append("files", file);
    }

    const res = await fetch(`${API_BASE}/upload/bulk`, {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `Bulk upload failed with status ${res.status}`);
    }

    const isSuccess = data.status === "success";
    const isPartial = data.status === "partial";

    uploadStatusBox.className = isSuccess || isPartial ? "alert-box alert-success" : "alert-box alert-error";

    let rowsHtml = "";
    if (data.files && data.files.length > 0) {
      rowsHtml = `
        <table class="bulk-results-table">
          <thead>
            <tr>
              <th>Document</th>
              <th>Format</th>
              <th>Size</th>
              <th>SQS Status</th>
            </tr>
          </thead>
          <tbody>
            ${data.files.map(f => `
              <tr>
                <td><code>${escapeHtml(f.filename)}</code></td>
                <td><span style="text-transform:uppercase; font-weight:600;">${escapeHtml(f.file_type)}</span></td>
                <td>${formatBytes(f.file_size)}</td>
                <td>${f.sqs_queued ? "✅ Queued in SQS" : "⚠️ Uploaded to S3"}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
    }

    uploadStatusBox.innerHTML = `
      <div>
        <strong>${isSuccess ? "🎉 Bulk Upload Successful!" : (isPartial ? "⚠️ Partial Upload Completed:" : "❌ Bulk Upload Failed:")}</strong><br>
        <span>${escapeHtml(data.message)}</span><br>
        <div style="margin-top: 6px; font-size: 0.85rem;">
          Target S3 Bucket: <code>${escapeHtml(data.files?.[0]?.bucket || "S3 Bucket")}</code>
        </div>
        ${rowsHtml}
      </div>
    `;

    // Clear staged files queue
    stagedFiles = [];
    renderStagedFiles();

    // Refresh documents filter dropdown
    loadDocumentsList();

  } catch (err) {
    uploadStatusBox.className = "alert-box alert-error";
    uploadStatusBox.innerHTML = `
      <div>
        <strong>❌ Bulk Upload Failed:</strong><br>
        ${escapeHtml(err.message)}
      </div>
    `;
  } finally {
    uploadSubmitBtn.disabled = false;
    uploadSubmitBtn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
        <polyline points="17 8 12 3 7 8"></polyline>
        <line x1="12" y1="3" x2="12" y2="15"></line>
      </svg>
      <span>Upload All Files to S3 & Trigger Ingestion</span>
    `;
  }
});

// -----------------------------------------------------------------------------
// 5. Image Modal Lightbox
// -----------------------------------------------------------------------------
function openModal(src, title) {
  modalImageSrc.src = src;
  modalImageTitle.textContent = title;
  imageModal.classList.remove("hidden");
}
window.openModal = openModal;

modalCloseBtn.addEventListener("click", () => {
  imageModal.classList.add("hidden");
});

document.querySelector(".modal-backdrop").addEventListener("click", () => {
  imageModal.classList.add("hidden");
});

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------
function formatBytes(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Initialize on page load
window.addEventListener("DOMContentLoaded", () => {
  updateThreadDisplay();
  checkHealth();
  loadDocumentsList();
  loadThreadsList();

  // Restore and persist Inference LLM selection
  if (modelSelect) {
    const savedModel = localStorage.getItem("academic_rag_model");
    if (savedModel) {
      modelSelect.value = savedModel;
    }
    modelSelect.addEventListener("change", () => {
      localStorage.setItem("academic_rag_model", modelSelect.value);
    });
  }
});
