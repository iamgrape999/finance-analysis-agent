const form = document.getElementById("chatForm");
const input = document.getElementById("messageInput");
const maxTokensInput = document.getElementById("maxTokensInput");
const temperatureInput = document.getElementById("temperatureInput");
const providerOrderInput = document.getElementById("providerOrderInput");
const responseModeInput = document.getElementById("responseModeInput");
const apiBaseInput = document.getElementById("apiBaseInput");
const webSearchProviderButtons = document.getElementById("webSearchProviderButtons");
const forceWebSearchInput = document.getElementById("forceWebSearchInput");
const toggleThemeButton = document.getElementById("toggleThemeButton");
const toggleSidebarButton = document.getElementById("toggleSidebarButton");
const toggleSettingsButton = document.getElementById("toggleSettingsButton");
const toggleLogButton = document.getElementById("toggleLogButton");
const closeLogButton = document.getElementById("closeLogButton");
const settingsPanel = document.getElementById("settingsPanel");
const closeSettingsButton = document.getElementById("closeSettingsButton");
const modelInputs = {
  nvidia: document.getElementById("nvidiaModelInput"),
  cerebras: document.getElementById("cerebrasModelInput"),
  mistral: document.getElementById("mistralModelInput"),
  openrouter: document.getElementById("openrouterModelInput"),
  fireworks: document.getElementById("fireworksModelInput"),
  gemini: document.getElementById("geminiModelInput"),
  cloudflare: document.getElementById("cloudflareModelInput"),
  groq: document.getElementById("groqModelInput"),
  aws: document.getElementById("awsModelInput")
};
const threadInput = document.getElementById("threadInput");
const messages = document.getElementById("messages");
const button = document.getElementById("sendButton");
const statusEl = document.getElementById("status");
const logs = document.getElementById("logs");
const logPanel = document.getElementById("logPanel");
const selfTestButton = document.getElementById("selfTestButton");
const newThreadButton = document.getElementById("newThreadButton");
const clearButton = document.getElementById("clearButton");
const refreshThreadsButton = document.getElementById("refreshThreadsButton");
const threadList = document.getElementById("threadList");
const globalMemoryList = document.getElementById("globalMemoryList");
const globalMemoryKeyInput = document.getElementById("globalMemoryKeyInput");
const globalMemoryValueInput = document.getElementById("globalMemoryValueInput");
const saveGlobalMemoryButton = document.getElementById("saveGlobalMemoryButton");
const passwordGate = document.getElementById("passwordGate");
const protectedApp = document.getElementById("protectedApp");
const mainWorkspace = document.getElementById("mainWorkspace");
const passwordInput = document.getElementById("passwordInput");
let passwordUserInput = document.getElementById("passwordUserInput");
const passwordApiBaseInput = document.getElementById("passwordApiBaseInput");
const passwordButton = document.getElementById("passwordButton");
const passwordError = document.getElementById("passwordError");
const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const imagePreviewImg = document.getElementById("imagePreviewImg");
const removeImageBtn = document.getElementById("removeImageBtn");
const attachButton = document.getElementById("attachButton");

let attachedImage = null; // { base64: string, mimeType: string, dataUrl: string } | null

function handleImageFile(file) {
  if (!file) return;
  const allowedTypes = ["image/jpeg", "image/png", "image/webp", "image/gif"];
  if (!allowedTypes.includes(file.type)) {
    log(`不支援的圖片格式：${file.type}，請上傳 JPG/PNG/WEBP/GIF`, "WARN");
    return;
  }
  if (file.size > 2 * 1024 * 1024) {
    log(`圖片超過 2 MB（${(file.size / 1024 / 1024).toFixed(1)} MB），請縮小後再上傳`, "WARN");
    alert("圖片超過 2 MB 限制，請縮小後再上傳。");
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target.result;
    const base64 = dataUrl.split(",")[1];
    attachedImage = { base64, mimeType: file.type, dataUrl };
    imagePreviewImg.src = dataUrl;
    imagePreview.hidden = false;
    log(`已附加圖片：${file.name} (${file.type}, ${(file.size / 1024).toFixed(0)} KB)`, "TRACE");
  };
  reader.readAsDataURL(file);
}

function removeImage() {
  attachedImage = null;
  imagePreview.hidden = true;
  imagePreviewImg.src = "";
  imageInput.value = "";
}

attachButton.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  if (file) handleImageFile(file);
});

removeImageBtn.addEventListener("click", removeImage);
if (!passwordUserInput) {
  passwordUserInput = document.createElement("input");
  passwordUserInput.id = "passwordUserInput";
  passwordUserInput.type = "text";
  passwordUserInput.placeholder = "User ID，例如 hanli / cathy / user01";
  passwordUserInput.autocomplete = "username";
  passwordInput.parentNode.insertBefore(passwordUserInput, passwordInput);
}

const threadKey = "finance_agent_thread_id";
const userIdKey = "finance_agent_user_id";
const apiBaseKey = "finance_agent_api_base";
const passwordKey = "finance_agent_app_password";
const responseModeKey = "finance_agent_response_mode";
const webSearchProviderKey = "finance_agent_web_search_provider";
const forceWebSearchKey = "finance_agent_force_web_search";
const themeModeKey = "finance_agent_theme_mode";
const FRONTEND_BUILD = "2026-04-21-route-budget-v2";
const responsePresets = {
  fast: {
    label: "快速短答",
    maxTokens: 768,
    historyTurns: 2,
    providerOrder: "cerebras,groq,mistral,gemini,openrouter,cloudflare,fireworks,aws,nvidia"
  },
  stable: {
    label: "穩定聊天",
    maxTokens: 2048,
    historyTurns: 4,
    providerOrder: "cerebras,groq,mistral,gemini,openrouter,cloudflare,fireworks,aws,nvidia"
  },
  deep: {
    label: "深度分析",
    maxTokens: 4096,
    historyTurns: 6,
    providerOrder: "cerebras,groq,mistral,gemini,openrouter,cloudflare,fireworks,aws,nvidia"
  }
};
const mistralModeDefaults = {
  fast: "mistral-small-latest",
  stable: "mistral-medium-latest",
  deep: "mistral-large-latest"
};
let userId = sanitizeUserId(localStorage.getItem(userIdKey) || "");
passwordUserInput.value = userId;
let threadId = localStorage.getItem(threadStorageKey()) || makeThreadId();
localStorage.setItem(threadStorageKey(), threadId);
threadInput.value = threadId;
apiBaseInput.value = localStorage.getItem(apiBaseKey) || "";
passwordApiBaseInput.value = apiBaseInput.value;
let providerReadiness = {};
let modelDefaults = {};
let appPassword = sessionStorage.getItem(passwordKey) || "";
responseModeInput.value = localStorage.getItem(responseModeKey) || "fast";
let webSearchProviderMode = localStorage.getItem(webSearchProviderKey) || "auto";
forceWebSearchInput.checked = localStorage.getItem(forceWebSearchKey) === "true";
let themeMode = localStorage.getItem(themeModeKey) || "";
let lastMobileLayoutState = isMobileLayout();

function makeThreadId() {
  const stamp = new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14);
  return `WEB_${stamp}_${crypto.randomUUID().slice(0, 6)}`;
}

function sanitizeUserId(value) {
  return String(value || "").trim().replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 80);
}

function threadStorageKey() {
  return `${threadKey}_${userId || "default"}`;
}

function preferredThemeMode() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function effectiveThemeMode() {
  return themeMode || preferredThemeMode();
}

function applyTheme() {
  const mode = effectiveThemeMode();
  document.body.dataset.theme = mode;
  if (toggleThemeButton) {
    toggleThemeButton.textContent = mode === "dark" ? "日間" : "夜間";
    toggleThemeButton.setAttribute("aria-label", mode === "dark" ? "切換到日間模式" : "切換到夜間模式");
    toggleThemeButton.title = mode === "dark" ? "切換到日間模式" : "切換到夜間模式";
  }
}

function toggleThemeMode() {
  const current = effectiveThemeMode();
  themeMode = current === "dark" ? "light" : "dark";
  localStorage.setItem(themeModeKey, themeMode);
  applyTheme();
  log(`theme mode=${themeMode}`, "DEBUG");
}

function log(message, level = "INFO") {
  const ts = new Date().toLocaleTimeString();
  const el = document.createElement("div");
  el.className = "log-line";
  el.textContent = `[${ts}] [${level}] ${message}`;
  logs.appendChild(el);
  logs.scrollTop = logs.scrollHeight;
}

function normalizeApiBase(raw) {
  let value = (raw || "").trim();
  if (!value) return "";
  if (!/^https?:\/\//i.test(value)) {
    value = `https://${value}`;
  }
  try {
    const url = new URL(value);
    return url.origin.replace(/\/+$/, "");
  } catch (err) {
    return value.replace(/\/+$/, "");
  }
}

function apiUrl(path) {
  const base = normalizeApiBase(apiBaseInput.value);
  return base ? `${base}${path}` : path;
}

function syncApiBaseFromLogin() {
  const value = normalizeApiBase(passwordApiBaseInput.value);
  passwordApiBaseInput.value = value;
  apiBaseInput.value = value;
  localStorage.setItem(apiBaseKey, value);
}

function friendlyNetworkError(err) {
  const message = err?.message || String(err);
  if (/load failed|failed to fetch|networkerror/i.test(message)) {
    return [
      "後端連線失敗。請確認：",
      "1. Backend API URL 使用 https://，不是 http://。",
      "2. 只填 Render 根網址，不要加 /api/health 或 /api/chat。",
      "3. iPhone Safari 可直接打開 Render 的 /api/health。",
      "4. Render Free 可能在冷啟動，等 30-60 秒後重試。",
      `原始錯誤：${message}`
    ].join("\n");
  }
  return message;
}

function authHeaders(extra = {}) {
  const headers = { ...extra };
  if (appPassword) {
    headers["X-App-Password"] = appPassword;
  }
  if (userId) {
    headers["X-User-Id"] = userId;
  }
  return headers;
}

function currentResponsePreset() {
  return responsePresets[responseModeInput.value] || responsePresets.fast;
}

function mobileProviderOrderFor(modeKey) {
  if (modeKey === "deep") {
    return "cerebras,mistral,gemini,openrouter,fireworks,groq,cloudflare,aws,nvidia";
  }
  if (modeKey === "stable") {
    return "cerebras,groq,mistral,gemini,openrouter,fireworks,cloudflare,aws,nvidia";
  }
  return "cerebras,groq,mistral,gemini,openrouter,fireworks,aws,nvidia";
}

function chatRequestTimeoutMsForMode(modeKey) {
  if (modeKey === "deep") {
    return 130000;
  }
  if (modeKey === "stable") {
    return 100000;
  }
  return 65000;
}

function applyResponsePreset(shouldLog = true) {
  const preset = currentResponsePreset();
  maxTokensInput.value = String(preset.maxTokens);
  if (!providerOrderInput.value) {
    providerOrderInput.value = isMobileLayout() ? mobileProviderOrderFor(responseModeInput.value) : preset.providerOrder;
  }
  localStorage.setItem(responseModeKey, responseModeInput.value);
  if (shouldLog) {
    log(`response mode: ${preset.label}; max_tokens=${preset.maxTokens}; history_turns=${preset.historyTurns}; provider_order=${providerOrderInput.value || "backend"}`, "DEBUG");
  }
}

function isMobileLayout() {
  return window.matchMedia("(max-width: 720px)").matches;
}

function collapseMobilePanels() {
  if (!isMobileLayout()) return;
  mainWorkspace.classList.add("sidebar-collapsed");
  logPanel.classList.remove("is-open");
  mainWorkspace.classList.remove("settings-open");
}

function setUnlocked(unlocked) {
  passwordGate.classList.toggle("app-hidden", unlocked);
  protectedApp.classList.toggle("app-locked", !unlocked);
  input.disabled = !unlocked;
  button.disabled = !unlocked;
}

function updateWebSearchProviderButtons() {
  if (!webSearchProviderButtons) return;
  for (const button of webSearchProviderButtons.querySelectorAll("[data-search-provider]")) {
    button.classList.toggle("is-active", button.dataset.searchProvider === webSearchProviderMode);
  }
}

function formatUsage(usage) {
  if (!usage || typeof usage !== "object") return "";
  const prompt = usage.prompt_tokens ?? usage.input_tokens ?? usage.prompt_token_count;
  const completion = usage.completion_tokens ?? usage.output_tokens ?? usage.candidates_token_count;
  const total = usage.total_tokens ?? usage.total_token_count;
  const parts = [];
  if (prompt !== undefined) parts.push(`prompt=${prompt}`);
  if (completion !== undefined) parts.push(`completion=${completion}`);
  if (total !== undefined) parts.push(`total=${total}`);
  return parts.join(" / ");
}

function summarizeWebSearchDecision(webSearch) {
  const decision = webSearch?.decision;
  if (!decision || typeof decision !== "object") return "";
  const reason = decision.reason || "";
  const priority = decision.priority ? `priority=${decision.priority}` : "";
  const confidence = typeof decision.confidence === "number" ? `confidence=${decision.confidence}` : "";
  const provider = webSearch?.provider ? `provider=${webSearch.provider}` : "";
  const override = webSearch?.provider_override && webSearch.provider_override !== "auto" ? `override=${webSearch.provider_override}` : "";
  const forced = webSearch?.forced ? "forced=true" : "";
  return [reason, priority, confidence, provider, override, forced].filter(Boolean).join(" | ");
}

function buildWebSourcesElement(webSearch) {
  if (!webSearch || typeof webSearch !== "object") return null;
  const results = Array.isArray(webSearch.results) ? webSearch.results : [];
  const reasonText = summarizeWebSearchDecision(webSearch);
  if (!reasonText && !results.length) return null;

  const wrap = document.createElement("section");
  wrap.className = "message-sources";

  if (reasonText) {
    const reasonEl = document.createElement("div");
    reasonEl.className = "message-sources-reason";
    reasonEl.textContent = `web search: ${reasonText}`;
    wrap.appendChild(reasonEl);
  }

  if (results.length) {
    const titleEl = document.createElement("div");
    titleEl.className = "message-sources-title";
    titleEl.textContent = "來源";
    wrap.appendChild(titleEl);

    const list = document.createElement("ul");
    list.className = "message-sources-list";
    for (const item of results.slice(0, 5)) {
      const li = document.createElement("li");
      const link = document.createElement("a");
      const url = String(item?.url || "").trim();
      const title = String(item?.title || url || "未命名來源").trim();
      link.textContent = title;
      if (url) {
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
      } else {
        link.href = "#";
        link.addEventListener("click", (event) => event.preventDefault());
      }
      li.appendChild(link);
      list.appendChild(li);
    }
    wrap.appendChild(list);
  }

  return wrap;
}

function buildWebSearchBadge(webSearch) {
  if (!webSearch || typeof webSearch !== "object") return null;
  if (!webSearch.used || !webSearch.provider) return null;

  const badge = document.createElement("div");
  badge.className = "message-search-badge";
  const providerLabel = String(webSearch.provider || "").trim();
  badge.textContent = `Web search: ${providerLabel}`;
  badge.title = summarizeWebSearchDecision(webSearch) || `web search provider=${providerLabel}`;
  return badge;
}

function addMessage(role, text, meta = {}) {
  const el = document.createElement("article");
  el.className = `message ${role}`;
  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;
  el.appendChild(body);

  if (role !== "user") {
    const searchBadge = buildWebSearchBadge(meta.web_search);
    if (searchBadge) {
      el.appendChild(searchBadge);
    }
  }

  const metaParts = [];
  if (meta.provider) metaParts.push(`platform=${meta.provider}`);
  if (meta.model) metaParts.push(`model=${meta.model}`);
  const usageText = formatUsage(meta.usage);
  if (usageText) metaParts.push(`tokens: ${usageText}`);
  if (meta.latency_s !== undefined && meta.latency_s !== null) metaParts.push(`time=${meta.latency_s}s`);
  if (meta.provider_attempts?.length) metaParts.push(`attempts=${meta.provider_attempts.join(">")}`);
  if (meta.continue_rounds) metaParts.push(`continue=${meta.continue_rounds}`);
  if (metaParts.length) {
    const metaEl = document.createElement("div");
    metaEl.className = "message-meta";
    metaEl.textContent = metaParts.join(" | ");
    el.appendChild(metaEl);
  }

  if (role !== "user") {
    const sourcesEl = buildWebSourcesElement(meta.web_search);
    if (sourcesEl) {
      el.appendChild(sourcesEl);
    }
  }

  messages.appendChild(el);
  requestAnimationFrame(() => {
    messages.scrollTop = messages.scrollHeight;
  });
}

function logProviderTrace(providerTrace) {
  if (!Array.isArray(providerTrace) || !providerTrace.length) return;
  for (const item of providerTrace) {
    const parts = [
      `trace ${item.provider || "unknown"}`,
      `status=${item.status || "unknown"}`,
      `context=${item.context_kind || "unknown"}`,
      `tokens=${item.prompt_tokens ?? "?"}`,
      `chars=${item.prompt_chars ?? "?"}`,
      `bytes=${item.estimated_request_bytes ?? "?"}`,
      `elapsed=${item.elapsed_ms ?? 0}ms`
    ];
    if (item.attempt_timeout_sec !== undefined) parts.push(`attempt_timeout=${item.attempt_timeout_sec}s`);
    if (item.remaining_budget_sec !== undefined) parts.push(`budget_left=${item.remaining_budget_sec}s`);
    if (item.reason) parts.push(`reason=${item.reason}`);
    log(parts.join(" | "), item.status === "ok" ? "TRACE" : "WARN");
  }
}

function logWebSearch(webSearch) {
  if (!webSearch || typeof webSearch !== "object") return;
  const reasonText = summarizeWebSearchDecision(webSearch);
  if (reasonText) {
    log(`web search decision: ${reasonText}`, "TRACE");
  }
  if (webSearch.error) {
    log(`web search error: ${webSearch.error}`, "WARN");
  }
  if (webSearch.provider_override && webSearch.provider_override !== "auto") {
    log(`web search override requested: ${webSearch.provider_override}`, "TRACE");
  }
  if (webSearch.used) {
    if (webSearch.provider) {
      log(`web search provider: ${webSearch.provider}`, "TRACE");
    }
    if (Array.isArray(webSearch.attempts) && webSearch.attempts.length) {
      log(`web search attempts: ${webSearch.attempts.join(">")}`, "TRACE");
    }
    if (Array.isArray(webSearch.fallback_errors) && webSearch.fallback_errors.length) {
      log(`web search fallback: ${webSearch.fallback_errors.join(" ; ")}`, "WARN");
    }
    const titles = (Array.isArray(webSearch.results) ? webSearch.results : [])
      .slice(0, 5)
      .map((item) => String(item?.title || item?.url || "").trim())
      .filter(Boolean);
    if (titles.length) {
      log(`web search sources: ${titles.join(" | ")}`, "TRACE");
    }
  }
}

function renderStoredMessages(records) {
  messages.innerHTML = "";
  if (!records.length) {
    addMessage("assistant", "這個 Thread 目前沒有對話紀錄，可以直接開始輸入。");
    return;
  }
  for (const record of records) {
    const role = record.role === "user" ? "user" : "assistant";
    addMessage(role, record.content || "", record.meta || {});
  }
}

function setBusy(isBusy) {
  button.disabled = isBusy;
  input.disabled = isBusy;
  button.textContent = isBusy ? "思考中" : "送出";
}

function collectModelOverrides() {
  const out = {};
  for (const [provider, el] of Object.entries(modelInputs)) {
    if (el && el.value) {
      out[provider] = el.value;
    }
  }
  if (!out.mistral) {
    out.mistral = mistralModeDefaults[responseModeInput.value] || mistralModeDefaults.fast;
  }
  return out;
}

function formatErrorDetails(err) {
  const lines = [];
  if (err?.name) lines.push(`name=${err.name}`);
  if (err?.message) lines.push(`message=${err.message}`);
  if (err?.stack) lines.push(`stack=${err.stack}`);
  return lines.join(" | ");
}

function effectiveProviderOrderForRequest() {
  const explicitOrder = (providerOrderInput.value || "").trim();
  if (explicitOrder) {
    return explicitOrder;
  }
  return isMobileLayout() ? mobileProviderOrderFor(responseModeInput.value) : null;
}

async function checkHealth() {
  try {
    if (!userId) {
      setUnlocked(false);
      passwordError.textContent = "請先輸入 User ID，系統會依 User ID 隔離歷史對話與全域記憶。";
      log("missing user id", "WARN");
      return;
    }
    passwordButton.disabled = true;
    passwordButton.textContent = "連線中";
    const res = await fetch(apiUrl("/api/health"), { headers: authHeaders() });
    const data = await res.json().catch(() => ({}));
    if (res.status === 401) {
      setUnlocked(false);
      passwordError.textContent = "密碼不正確，請重新輸入。";
      log("health unauthorized: invalid password", "WARN");
      return;
    }
    if (!res.ok) {
      setUnlocked(false);
      passwordError.textContent = `後端連線失敗：HTTP ${res.status}`;
      log(`health failed: HTTP ${res.status}`, "FAIL");
      return;
    }
    if (data.user_required !== true || data.user_id !== userId) {
      setUnlocked(false);
      passwordError.textContent = "後端尚未更新到 User ID 隔離版本，請重啟本機 uvicorn 或重新部署 Render 後再測。";
      log(`backend user scope mismatch; expected=${userId} got=${data.user_id || "none"} user_required=${data.user_required}`, "FAIL");
      return;
    }
    providerReadiness = data.providers || {};
    modelDefaults = data.model_defaults || {};
    if (data.backend_version) {
      log(`backend version=${data.backend_version}`, "TRACE");
    }
    const diagnostics = data.provider_diagnostics || {};
    if (Array.isArray(diagnostics.web_search_provider_order) && diagnostics.web_search_provider_order.length) {
      log(`web search providers=${diagnostics.web_search_provider_order.join(">")}`, "TRACE");
    }
    if (diagnostics.nvidia_key_present === false) {
      log("nvidia health: NVIDIA_API_KEY is missing; NVIDIA will be skipped", "WARN");
    } else if (diagnostics.nvidia_model) {
      const nvidiaDiff = diagnostics.nvidia_raw_model && diagnostics.nvidia_effective_model && diagnostics.nvidia_raw_model !== diagnostics.nvidia_effective_model
        ? `; effective=${diagnostics.nvidia_effective_model}`
        : "";
      log(`nvidia health: model=${diagnostics.nvidia_model}; base_url=${diagnostics.nvidia_base_url || "default"}${nvidiaDiff}`, "TRACE");
    }
    if (diagnostics.cerebras_key_present === false) {
      log("cerebras health: CEREBRAS_API_KEY is missing; Cerebras will be skipped", "WARN");
    } else if (diagnostics.cerebras_model) {
      const cerebrasDiff = diagnostics.cerebras_raw_model && diagnostics.cerebras_effective_model && diagnostics.cerebras_raw_model !== diagnostics.cerebras_effective_model
        ? `; effective=${diagnostics.cerebras_effective_model}`
        : "";
      log(`cerebras health: model=${diagnostics.cerebras_model}${cerebrasDiff}`, "TRACE");
    }
    if (!data.password_required) {
      log("backend reports password_required=false; set APP_PASSWORD on Render to enforce login", "WARN");
    }
    const ready = Object.entries(data.providers || {})
      .filter(([, ok]) => ok)
      .map(([name]) => name);
    statusEl.textContent = ready.length ? `已連線：${ready.join(", ")}` : "尚未設定模型金鑰";
    statusEl.classList.toggle("warn", ready.length === 0);
    setUnlocked(true);
    log(`health ok; user=${data.user_id}; scope=${data.storage_scope || "unknown"}; providers=${ready.join(", ") || "none"}`, ready.length ? "OK" : "WARN");
    if (providerReadiness.fireworks) {
      loadFireworksModels();
    }
    loadThreads();
    loadGlobalMemory();
  } catch (err) {
    statusEl.textContent = "後端未連線";
    statusEl.classList.add("warn");
    setUnlocked(false);
    passwordError.textContent = `後端連線失敗：${err.message}`;
    log(`health failed: ${err.message}`, "FAIL");
  } finally {
    passwordButton.disabled = false;
    passwordButton.textContent = "連線";
  }
}

async function loadThreads() {
  if (!threadList) return;
  try {
    const res = await fetch(apiUrl("/api/threads"), { headers: authHeaders() });
    const data = await res.json().catch(() => []);
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }
    threadList.innerHTML = "";
    if (!data.length) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = "尚無歷史對話";
      threadList.appendChild(empty);
      return;
    }
    for (const item of data) {
      const row = document.createElement("div");
      row.className = `thread-row${item.thread_id === threadId ? " active" : ""}`;

      const loadBtn = document.createElement("button");
      loadBtn.type = "button";
      loadBtn.className = "thread-load";
      loadBtn.dataset.threadId = item.thread_id;
      loadBtn.textContent = `${item.preview || item.thread_id}\n${item.message_count || 0} 則 | ${item.updated_at || item.thread_id}`;

      const deleteBtn = document.createElement("button");
      deleteBtn.type = "button";
      deleteBtn.className = "thread-delete";
      deleteBtn.dataset.threadId = item.thread_id;
      deleteBtn.textContent = "刪除";

      row.appendChild(loadBtn);
      row.appendChild(deleteBtn);
      threadList.appendChild(row);
    }
  } catch (err) {
    log(`load threads failed: ${err.message}`, "WARN");
  }
}

async function loadThread(threadIdToLoad) {
  if (!threadIdToLoad) return;
  try {
    const res = await fetch(apiUrl(`/api/threads/${encodeURIComponent(threadIdToLoad)}/messages`), { headers: authHeaders() });
    const data = await res.json().catch(() => []);
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }
    threadId = threadIdToLoad;
    threadInput.value = threadId;
    localStorage.setItem(threadStorageKey(), threadId);
    renderStoredMessages(data);
    loadThreads();
    log(`loaded thread: ${threadId}`, "OK");
  } catch (err) {
    log(`load thread failed: ${err.message}`, "FAIL");
  }
}

async function deleteThread(threadIdToDelete) {
  if (!threadIdToDelete) return;
  if (!window.confirm(`確定刪除此對話？\n${threadIdToDelete}`)) return;
  try {
    const res = await fetch(apiUrl(`/api/threads/${encodeURIComponent(threadIdToDelete)}`), {
      method: "DELETE",
      headers: authHeaders()
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.ok === false) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }
    if (threadIdToDelete === threadId) {
      threadId = makeThreadId();
      threadInput.value = threadId;
      localStorage.setItem(threadStorageKey(), threadId);
      messages.innerHTML = "";
      addMessage("assistant", `已刪除原 Thread，並切換新 Thread：${threadId}`);
    }
    loadThreads();
    log(`deleted thread: ${threadIdToDelete}`, "OK");
  } catch (err) {
    log(`delete thread failed: ${err.message}`, "FAIL");
  }
}

async function loadGlobalMemory() {
  if (!globalMemoryList) return;
  try {
    const res = await fetch(apiUrl("/api/global-memory"), { headers: authHeaders() });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }
    renderGlobalMemory(data.facts || {});
  } catch (err) {
    log(`load global memory failed: ${err.message}`, "WARN");
  }
}

function renderGlobalMemory(facts) {
  globalMemoryList.innerHTML = "";
  const entries = Object.entries(facts || {}).sort(([a], [b]) => a.localeCompare(b));
  if (!entries.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "尚無全域記憶";
    globalMemoryList.appendChild(empty);
    return;
  }
  for (const [key, raw] of entries) {
    const value = raw && typeof raw === "object" ? raw.value : raw;
    const row = document.createElement("div");
    row.className = "memory-row";
    const text = document.createElement("div");
    text.className = "memory-text";
    text.textContent = `${key} = ${value ?? ""}`;
    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "memory-delete";
    deleteBtn.dataset.key = key;
    deleteBtn.textContent = "刪除";
    row.appendChild(text);
    row.appendChild(deleteBtn);
    globalMemoryList.appendChild(row);
  }
}

async function saveGlobalMemory() {
  saveGlobalMemoryButton.disabled = true;
  const rawKey = (globalMemoryKeyInput.value || "").trim();
  let key = rawKey;
  let value = (globalMemoryValueInput.value || "").trim();
  if (rawKey.includes("=") && !value) {
    const parts = rawKey.split("=");
    key = parts.shift().trim();
    value = parts.join("=").trim();
  }
  if (!key) {
    log("global memory key is required", "WARN");
    saveGlobalMemoryButton.disabled = false;
    return;
  }
  try {
    log(`saving global memory: ${key}`, "CALL");
    const res = await fetch(apiUrl("/api/global-memory"), {
      method: "PUT",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ key, value })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }
    globalMemoryKeyInput.value = "";
    globalMemoryValueInput.value = "";
    renderGlobalMemory(data.facts || {});
    await loadGlobalMemory();
    log(`saved global memory: ${key}`, "OK");
  } catch (err) {
    log(`save global memory failed: ${err.message}`, "FAIL");
  } finally {
    saveGlobalMemoryButton.disabled = false;
  }
}

async function deleteGlobalMemory(key) {
  if (!key) return;
  try {
    const res = await fetch(apiUrl(`/api/global-memory/${encodeURIComponent(key)}`), {
      method: "DELETE",
      headers: authHeaders()
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }
    renderGlobalMemory(data.facts || {});
    log(`deleted global memory: ${key}`, "OK");
  } catch (err) {
    log(`delete global memory failed: ${err.message}`, "FAIL");
  }
}

async function loadFireworksModels() {
  const select = modelInputs.fireworks;
  if (!select) return;
  try {
    const res = await fetch(apiUrl("/api/provider-models/fireworks"), { headers: authHeaders() });
    const data = await res.json().catch(() => ({ ok: false, error: "JSON parse failed", models: [] }));
    if (!data.ok) {
      log(`fireworks model list failed: ${data.error || "unknown error"}`, "WARN");
      return;
    }
    const models = data.models || [];
    const currentModel = data.current_model || {};
    if (!models.length) {
      log("fireworks model list returned no serverless models", "WARN");
      return;
    }
    models.sort((a, b) => fireworksRank(a.name) - fireworksRank(b.name));

    const previous = select.value;
    select.innerHTML = "";
    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    let defaultLabel = modelDefaults.fireworks
      ? `使用 .env 預設：${modelDefaults.fireworks}`
      : "使用 .env 預設";
    if (currentModel.name) {
      if (!currentModel.online_in_catalog) {
        defaultLabel += " [not in account list]";
      } else if (!currentModel.serverless_supported) {
        defaultLabel += " [catalog online, not serverless]";
      }
    }
    defaultOption.textContent = defaultLabel;
    select.appendChild(defaultOption);

    const group = document.createElement("optgroup");
    group.label = "Fireworks 可用 serverless 模型";
    for (const item of models) {
      const option = document.createElement("option");
      option.value = item.name;
      option.textContent = item.display_name && item.display_name !== item.name
        ? `${item.display_name} (${item.name})`
        : item.name;
      group.appendChild(option);
    }
    select.appendChild(group);

    if (currentModel.name && !currentModel.serverless_supported) {
      const legacyGroup = document.createElement("optgroup");
      legacyGroup.label = "目前預設模型狀態";
      const option = document.createElement("option");
      option.value = currentModel.name;
      const currentLabel = currentModel.display_name && currentModel.display_name !== currentModel.name
        ? `${currentModel.display_name} (${currentModel.name})`
        : currentModel.name;
      option.textContent = !currentModel.online_in_catalog
        ? `${currentLabel} [deprecated or not in account list]`
        : `${currentLabel} [online in catalog, not serverless]`;
      legacyGroup.appendChild(option);
      select.appendChild(legacyGroup);
      log(
        currentModel.online_in_catalog
          ? `fireworks current model is online but not serverless: ${currentModel.name}`
          : `fireworks current model is not in account list: ${currentModel.name}`,
        "WARN"
      );
    }

    if ([...select.options].some((option) => option.value === previous)) {
      select.value = previous;
    }
    log(`loaded ${models.length} fireworks serverless models`, "OK");
    if (currentModel.name) {
      log(
        `fireworks current model status: name=${currentModel.name}; online_in_catalog=${Boolean(currentModel.online_in_catalog)}; serverless_supported=${Boolean(currentModel.serverless_supported)}`,
        "TRACE"
      );
    }
  } catch (err) {
    log(`fireworks model list request failed: ${err.message}`, "WARN");
  }
}

function fireworksRank(name) {
  const n = String(name || "").toLowerCase();
  if (n.includes("minimax-m2p7")) return 10;
  if (n.includes("minimax")) return 20;
  if (n.includes("deepseek-v3p2")) return 30;
  if (n.includes("deepseek")) return 40;
  if (n.includes("gpt-oss-120b")) return 50;
  if (n.includes("glm-5")) return 60;
  if (n.includes("qwen3p6")) return 70;
  if (n.includes("kimi")) return 80;
  if (n.includes("llama") && n.includes("70b")) return 90;
  if (n.includes("gpt-oss-20b")) return 100;
  if (n.includes("qwen3-8b")) return 110;
  if (n.includes("vl")) return 120;
  return 999;
}

async function sendMessage(message, endpoint = "/api/chat") {
  if (!message) return;
  if (protectedApp.classList.contains("app-locked")) {
    passwordError.textContent = "請先輸入密碼並連線。";
    return;
  }

  addMessage("user", message);
  collapseMobilePanels();
  input.value = "";
  setBusy(true);
  const effectiveProviderOrder = effectiveProviderOrderForRequest();
  const firstProvider = (effectiveProviderOrder || "").split(",")[0]?.trim();
  if (firstProvider && providerReadiness[firstProvider] === false) {
    log(`${firstProvider} is selected first but not connected; backend will fail over to the next ready provider`, "WARN");
  }
  const imageSnapshot = attachedImage;
  if (imageSnapshot) {
    log(`image attached: mimeType=${imageSnapshot.mimeType}; forcing provider_order=gemini`, "TRACE");
  }
  log(`calling ${endpoint}; user=${userId}; thread=${threadId}; provider_order=${imageSnapshot ? "gemini" : (effectiveProviderOrder || "backend")}`, "CALL");
  log(`web search request: provider=${webSearchProviderMode}; force=${forceWebSearchInput.checked}`, "TRACE");

  let timeoutId = null;
  try {
    const started = performance.now();
    const preset = currentResponsePreset();
    const isMobile = isMobileLayout();
    const effectiveMaxTokens = Number(maxTokensInput.value || preset.maxTokens);
    const effectiveHistoryTurns = isMobile ? Math.min(Number(preset.historyTurns || 2), 2) : Number(preset.historyTurns || 2);
    const requestTimeoutMs = chatRequestTimeoutMsForMode(responseModeInput.value);
    const disableAutoContinue = isMobile;
    const controller = new AbortController();
    timeoutId = window.setTimeout(() => controller.abort(new DOMException("Request timed out", "TimeoutError")), requestTimeoutMs);
    const res = await fetch(apiUrl(endpoint), {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      signal: controller.signal,
      body: JSON.stringify({
        thread_id: threadId,
        message,
        max_tokens: effectiveMaxTokens,
        temperature: Number(temperatureInput.value || 0.2),
        use_history: true,
        history_turns: effectiveHistoryTurns,
        response_mode: responseModeInput.value,
        disable_auto_continue: disableAutoContinue,
        provider_order: imageSnapshot ? "gemini" : effectiveProviderOrder,
        model_overrides: collectModelOverrides(),
        web_search_provider: webSearchProviderMode,
        force_web_search: forceWebSearchInput.checked,
        ...(imageSnapshot ? { image_base64: imageSnapshot.base64, image_mime_type: imageSnapshot.mimeType } : {})
      })
    });

    const data = await res.json();
    if (res.status === 401) {
      setUnlocked(false);
      passwordError.textContent = "密碼不正確，請重新輸入。";
      throw new Error("Invalid app password");
    }
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }
    addMessage("assistant", data.reply, {
      provider: data.provider,
      model: data.model,
      usage: data.usage,
      latency_s: data.latency_s,
      provider_attempts: data.provider_attempts,
      failover_errors: data.failover_errors,
      continue_rounds: data.continue_rounds,
      web_search: data.web_search
    });
    if (data.failover_errors?.length) {
      log(`failover: ${data.failover_errors.map((item) => `${item.provider} -> ${item.error}`).join(" ; ")}`, "WARN");
    }
    logProviderTrace(data.provider_trace);
    logWebSearch(data.web_search);
    if (data.context_sizes) {
      log(
        `context sizes: heavy=${data.context_sizes.heavy_prompt_tokens || "?"}t/${data.context_sizes.heavy_prompt_chars || "?"}c | ` +
        `light=${data.context_sizes.light_prompt_tokens || "?"}t/${data.context_sizes.light_prompt_chars || "?"}c`,
        "TRACE"
      );
    }
    if (data.route_timeout_sec !== undefined && data.route_timeout_sec !== null) {
      log(`route timeout budget=${data.route_timeout_sec}s`, "TRACE");
    }
    log(`reply ok; provider=${data.provider || "unknown"} model=${data.model || "unknown"} latency=${data.latency_s ?? ((performance.now() - started) / 1000).toFixed(2)}s`, "OK");
    loadThreads();
    loadGlobalMemory();
  } catch (err) {
    const rawMessage = err?.name === "AbortError" || /timed out/i.test(err?.message || "")
      ? "請求逾時，已中止這次對話。請稍後再試，或改用更快的回覆模式。"
      : friendlyNetworkError(err);
    addMessage("assistant", `發生錯誤：${rawMessage}`);
    log(`chat failed: ${err?.name || "Error"} ${err?.message || String(err)}; thread=${threadId}; provider_order=${effectiveProviderOrder || "backend"}`, "FAIL");
    if (err?.name === "SyntaxError" || /expected pattern/i.test(err?.message || "")) {
      log(`syntax error details: ${formatErrorDetails(err)}`, "TRACE");
      log(`syntax error context: api_base=${apiBaseInput.value || "(same origin)"}; provider_order=${effectiveProviderOrder || "backend"}; user=${userId}; thread=${threadId}`, "TRACE");
    }
  } finally {
    if (timeoutId !== null) {
      window.clearTimeout(timeoutId);
    }
    forceWebSearchInput.checked = false;
    localStorage.setItem(forceWebSearchKey, "false");
    removeImage();
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await sendMessage(input.value.trim());
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    form.requestSubmit();
  }
});

toggleSidebarButton.addEventListener("click", () => {
  mainWorkspace.classList.toggle("sidebar-collapsed");
  mainWorkspace.classList.remove("settings-open");
  logPanel.classList.remove("is-open");
});

toggleSettingsButton.addEventListener("click", () => {
  if (isMobileLayout()) {
    mainWorkspace.classList.add("sidebar-collapsed");
    logPanel.classList.remove("is-open");
  }
  mainWorkspace.classList.toggle("settings-open");
});

toggleLogButton.addEventListener("click", () => {
  if (isMobileLayout()) {
    mainWorkspace.classList.add("sidebar-collapsed");
    mainWorkspace.classList.remove("settings-open");
  }
  logPanel.classList.toggle("is-open");
});

closeLogButton.addEventListener("click", () => {
  logPanel.classList.remove("is-open");
});

closeSettingsButton.addEventListener("click", () => {
  mainWorkspace.classList.remove("settings-open");
});

threadInput.addEventListener("change", () => {
  threadId = threadInput.value.trim() || makeThreadId();
  threadInput.value = threadId;
  localStorage.setItem(threadStorageKey(), threadId);
  log(`thread updated: ${threadId}`, "DEBUG");
  loadThread(threadId);
});

apiBaseInput.addEventListener("change", () => {
  const value = normalizeApiBase(apiBaseInput.value);
  apiBaseInput.value = value;
  passwordApiBaseInput.value = value;
  localStorage.setItem(apiBaseKey, value);
  log(`backend api url updated: ${value || "same-origin"}`, "DEBUG");
  checkHealth();
});

responseModeInput.addEventListener("change", () => {
  applyResponsePreset(true);
});

if (toggleThemeButton) {
  toggleThemeButton.addEventListener("click", () => {
    toggleThemeMode();
  });
}

if (webSearchProviderButtons) {
  webSearchProviderButtons.addEventListener("click", (event) => {
    const buttonEl = event.target.closest("[data-search-provider]");
    if (!buttonEl) return;
    webSearchProviderMode = buttonEl.dataset.searchProvider || "auto";
    localStorage.setItem(webSearchProviderKey, webSearchProviderMode);
    updateWebSearchProviderButtons();
    log(`web search provider override=${webSearchProviderMode}`, "DEBUG");
  });
}

forceWebSearchInput.addEventListener("change", () => {
  localStorage.setItem(forceWebSearchKey, forceWebSearchInput.checked ? "true" : "false");
  log(`force web search=${forceWebSearchInput.checked}`, "DEBUG");
});

window.addEventListener("resize", () => {
  const nowMobile = isMobileLayout();
  if (nowMobile === lastMobileLayoutState) return;
  lastMobileLayoutState = nowMobile;
  applyResponsePreset(false);
  log(`layout changed: ${nowMobile ? "mobile" : "desktop"}; provider_order=${providerOrderInput.value || "backend"}`, "DEBUG");
  if (nowMobile) {
    mainWorkspace.classList.add("sidebar-collapsed");
    mainWorkspace.classList.remove("settings-open");
    logPanel.classList.remove("is-open");
  }
});

selfTestButton.addEventListener("click", async () => {
  await sendMessage("SelfTest：請只回覆 OK，並用繁體中文。", "/api/selftest");
});

newThreadButton.addEventListener("click", () => {
  threadId = makeThreadId();
  threadInput.value = threadId;
  localStorage.setItem(threadStorageKey(), threadId);
  messages.innerHTML = "";
  addMessage("assistant", `已切換新 Thread：${threadId}`);
  log(`new thread: ${threadId}`, "DEBUG");
  loadThreads();
});

clearButton.addEventListener("click", () => {
  messages.innerHTML = "";
  addMessage("assistant", "畫面已清空，不會刪除後端 memory。");
  log("screen cleared", "DEBUG");
});

refreshThreadsButton.addEventListener("click", () => {
  loadThreads();
});

threadList.addEventListener("click", (event) => {
  const loadBtn = event.target.closest(".thread-load");
  if (loadBtn) {
    loadThread(loadBtn.dataset.threadId);
    return;
  }
  const deleteBtn = event.target.closest(".thread-delete");
  if (deleteBtn) {
    deleteThread(deleteBtn.dataset.threadId);
  }
});

saveGlobalMemoryButton.addEventListener("click", () => {
  saveGlobalMemory();
});

globalMemoryKeyInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    saveGlobalMemory();
  }
});

globalMemoryValueInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    saveGlobalMemory();
  }
});

globalMemoryList.addEventListener("click", (event) => {
  const deleteBtn = event.target.closest(".memory-delete");
  if (deleteBtn) {
    deleteGlobalMemory(deleteBtn.dataset.key);
  }
});

passwordButton.addEventListener("click", () => {
  syncApiBaseFromLogin();
  const nextUserId = sanitizeUserId(passwordUserInput.value);
  if (!nextUserId) {
    passwordError.textContent = "請輸入 User ID，避免多人共用同一份對話紀錄。";
    log("login blocked: missing user id", "WARN");
    return;
  }
  const userChanged = nextUserId !== userId;
  userId = nextUserId;
  passwordUserInput.value = userId;
  localStorage.setItem(userIdKey, userId);
  if (userChanged) {
    threadId = localStorage.getItem(threadStorageKey()) || makeThreadId();
    localStorage.setItem(threadStorageKey(), threadId);
    threadInput.value = threadId;
    messages.innerHTML = "";
    addMessage("assistant", `已切換使用者：${userId}。歷史對話與全域記憶只會顯示此使用者的資料。`);
  }
  appPassword = passwordInput.value;
  sessionStorage.setItem(passwordKey, appPassword);
  passwordError.textContent = "";
  checkHealth();
});

passwordInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    passwordButton.click();
  }
});

passwordUserInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    passwordButton.click();
  }
});

log("UI initialized", "STEP");
log(`frontend build=${FRONTEND_BUILD}`, "TRACE");
applyTheme();
updateWebSearchProviderButtons();
setUnlocked(false);
applyResponsePreset(false);
if (isMobileLayout()) {
  mainWorkspace.classList.add("sidebar-collapsed");
}
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (!themeMode) {
    applyTheme();
  }
});
checkHealth();
