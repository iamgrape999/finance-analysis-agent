const form = document.getElementById("chatForm");
const input = document.getElementById("messageInput");
const maxTokensInput = document.getElementById("maxTokensInput");
const temperatureInput = document.getElementById("temperatureInput");
const providerOrderInput = document.getElementById("providerOrderInput");
const responseModeInput = document.getElementById("responseModeInput");
const apiBaseInput = document.getElementById("apiBaseInput");
const toggleSidebarButton = document.getElementById("toggleSidebarButton");
const toggleSettingsButton = document.getElementById("toggleSettingsButton");
const toggleLogButton = document.getElementById("toggleLogButton");
const closeLogButton = document.getElementById("closeLogButton");
const advancedSettings = document.getElementById("advancedSettings");
const modelInputs = {
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
const CHAT_REQUEST_TIMEOUT_MS = 90000;
const FRONTEND_BUILD = "2026-04-21-route-budget-v2";
const responsePresets = {
  fast: {
    label: "快速短答",
    maxTokens: 768,
    historyTurns: 2,
    providerOrder: "cerebras,mistral,openrouter,groq,cloudflare,aws,fireworks,gemini"
  },
  stable: {
    label: "穩定聊天",
    maxTokens: 2048,
    historyTurns: 4,
    providerOrder: "cerebras,mistral,openrouter,cloudflare,groq,aws,fireworks,gemini"
  },
  deep: {
    label: "深度分析",
    maxTokens: 4096,
    historyTurns: 6,
    providerOrder: "cerebras,mistral,openrouter,aws,cloudflare,groq,fireworks,gemini"
  }
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
let appPassword = localStorage.getItem(passwordKey) || "";
responseModeInput.value = localStorage.getItem(responseModeKey) || "fast";
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
    return "cerebras,mistral,openrouter,aws,fireworks,cloudflare,groq,gemini";
  }
  if (modeKey === "stable") {
    return "cerebras,mistral,openrouter,fireworks,aws,gemini";
  }
  return "cerebras,mistral,openrouter,fireworks,gemini";
}

function chatRequestTimeoutMsForMode(modeKey) {
  if (modeKey === "deep") {
    return 95000;
  }
  if (modeKey === "stable") {
    return 80000;
  }
  return 60000;
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
}

function setUnlocked(unlocked) {
  passwordGate.classList.toggle("app-hidden", unlocked);
  protectedApp.classList.toggle("app-locked", !unlocked);
  input.disabled = !unlocked;
  button.disabled = !unlocked;
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

function addMessage(role, text, meta = {}) {
  const el = document.createElement("article");
  el.className = `message ${role}`;
  const body = document.createElement("div");
  body.textContent = text;
  el.appendChild(body);

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
  return out;
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
    const data = await res.json();
    if (!data.ok) {
      log(`fireworks model list failed: ${data.error || "unknown error"}`, "WARN");
      return;
    }
    const models = data.models || [];
    if (!models.length) {
      log("fireworks model list returned no serverless models", "WARN");
      return;
    }
    models.sort((a, b) => fireworksRank(a.name) - fireworksRank(b.name));

    const previous = select.value;
    select.innerHTML = "";
    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = modelDefaults.fireworks
      ? `使用 .env 預設：${modelDefaults.fireworks}`
      : "使用 .env 預設";
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

    if ([...select.options].some((option) => option.value === previous)) {
      select.value = previous;
    }
    log(`loaded ${models.length} fireworks serverless models`, "OK");
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
  log(`calling ${endpoint}; user=${userId}; thread=${threadId}; provider_order=${effectiveProviderOrder || "backend"}`, "CALL");

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
        provider_order: effectiveProviderOrder,
        model_overrides: collectModelOverrides()
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
      continue_rounds: data.continue_rounds
    });
    if (data.failover_errors?.length) {
      log(`failover: ${data.failover_errors.map((item) => `${item.provider} -> ${item.error}`).join(" ; ")}`, "WARN");
    }
    logProviderTrace(data.provider_trace);
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
  } finally {
    if (timeoutId !== null) {
      window.clearTimeout(timeoutId);
    }
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
  logPanel.classList.remove("is-open");
});

toggleSettingsButton.addEventListener("click", () => {
  mainWorkspace.classList.remove("sidebar-collapsed");
  advancedSettings.open = !advancedSettings.open;
});

toggleLogButton.addEventListener("click", () => {
  if (isMobileLayout()) {
    mainWorkspace.classList.add("sidebar-collapsed");
  }
  logPanel.classList.toggle("is-open");
});

closeLogButton.addEventListener("click", () => {
  logPanel.classList.remove("is-open");
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

window.addEventListener("resize", () => {
  const nowMobile = isMobileLayout();
  if (nowMobile === lastMobileLayoutState) return;
  lastMobileLayoutState = nowMobile;
  applyResponsePreset(false);
  log(`layout changed: ${nowMobile ? "mobile" : "desktop"}; provider_order=${providerOrderInput.value || "backend"}`, "DEBUG");
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
  localStorage.setItem(passwordKey, appPassword);
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
setUnlocked(false);
applyResponsePreset(false);
if (isMobileLayout()) {
  mainWorkspace.classList.add("sidebar-collapsed");
}
window.addEventListener("resize", () => {
  if (isMobileLayout()) {
    mainWorkspace.classList.add("sidebar-collapsed");
    logPanel.classList.remove("is-open");
  }
});
checkHealth();
