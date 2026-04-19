const form = document.getElementById("chatForm");
const input = document.getElementById("messageInput");
const maxTokensInput = document.getElementById("maxTokensInput");
const temperatureInput = document.getElementById("temperatureInput");
const providerOrderInput = document.getElementById("providerOrderInput");
const apiBaseInput = document.getElementById("apiBaseInput");
const modelInputs = {
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
const selfTestButton = document.getElementById("selfTestButton");
const newThreadButton = document.getElementById("newThreadButton");
const clearButton = document.getElementById("clearButton");

const threadKey = "finance_agent_thread_id";
const apiBaseKey = "finance_agent_api_base";
let threadId = localStorage.getItem(threadKey) || makeThreadId();
localStorage.setItem(threadKey, threadId);
threadInput.value = threadId;
apiBaseInput.value = localStorage.getItem(apiBaseKey) || "";
let providerReadiness = {};

function makeThreadId() {
  const stamp = new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14);
  return `WEB_${stamp}_${crypto.randomUUID().slice(0, 6)}`;
}

function log(message, level = "INFO") {
  const ts = new Date().toLocaleTimeString();
  const el = document.createElement("div");
  el.className = "log-line";
  el.textContent = `[${ts}] [${level}] ${message}`;
  logs.appendChild(el);
  logs.scrollTop = logs.scrollHeight;
}

function apiUrl(path) {
  const base = (apiBaseInput.value || "").trim().replace(/\/+$/, "");
  return base ? `${base}${path}` : path;
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
  if (meta.failover_errors?.length) metaParts.push(`failover=${meta.failover_errors.map((item) => `${item.provider}: ${item.error}`).join(" ; ")}`);
  if (metaParts.length) {
    const metaEl = document.createElement("div");
    metaEl.className = "message-meta";
    metaEl.textContent = metaParts.join(" | ");
    el.appendChild(metaEl);
  }

  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
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

async function checkHealth() {
  try {
    const res = await fetch(apiUrl("/api/health"));
    const data = await res.json();
    providerReadiness = data.providers || {};
    const ready = Object.entries(data.providers || {})
      .filter(([, ok]) => ok)
      .map(([name]) => name);
    statusEl.textContent = ready.length ? `已連線：${ready.join(", ")}` : "尚未設定模型金鑰";
    statusEl.classList.toggle("warn", ready.length === 0);
    log(`health ok; providers=${ready.join(", ") || "none"}`, ready.length ? "OK" : "WARN");
    if (providerReadiness.fireworks) {
      loadFireworksModels();
    }
  } catch (err) {
    statusEl.textContent = "後端未連線";
    statusEl.classList.add("warn");
    log(`health failed: ${err.message}`, "FAIL");
  }
}

async function loadFireworksModels() {
  const select = modelInputs.fireworks;
  if (!select) return;
  try {
    const res = await fetch(apiUrl("/api/provider-models/fireworks"));
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
    defaultOption.textContent = "使用 .env 預設";
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
  if (n.includes("deepseek")) return 10;
  if (n.includes("gpt-oss-120b")) return 20;
  if (n.includes("glm-5")) return 30;
  if (n.includes("qwen3p6")) return 40;
  if (n.includes("kimi")) return 50;
  if (n.includes("llama") && n.includes("70b")) return 60;
  if (n.includes("minimax")) return 70;
  if (n.includes("gpt-oss-20b")) return 80;
  if (n.includes("qwen3-8b")) return 90;
  if (n.includes("vl")) return 100;
  return 999;
}

async function sendMessage(message, endpoint = "/api/chat") {
  if (!message) return;

  addMessage("user", message);
  input.value = "";
  setBusy(true);
  const firstProvider = (providerOrderInput.value || "").split(",")[0]?.trim();
  if (firstProvider && providerReadiness[firstProvider] === false) {
    log(`${firstProvider} is selected first but not connected; backend will fail over to the next ready provider`, "WARN");
  }
  log(`calling ${endpoint}; thread=${threadId}`, "CALL");

  try {
    const started = performance.now();
    const res = await fetch(apiUrl(endpoint), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        thread_id: threadId,
        message,
        max_tokens: Number(maxTokensInput.value || 4096),
        temperature: Number(temperatureInput.value || 0.2),
        use_history: true,
        provider_order: providerOrderInput.value || null,
        model_overrides: collectModelOverrides()
      })
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }
    addMessage("assistant", data.reply, {
      provider: data.provider,
      model: data.model,
      usage: data.usage,
      latency_s: data.latency_s,
      provider_attempts: data.provider_attempts,
      failover_errors: data.failover_errors
    });
    if (data.failover_errors?.length) {
      log(`failover: ${data.failover_errors.map((item) => `${item.provider} -> ${item.error}`).join(" ; ")}`, "WARN");
    }
    log(`reply ok; provider=${data.provider || "unknown"} model=${data.model || "unknown"} latency=${data.latency_s ?? ((performance.now() - started) / 1000).toFixed(2)}s`, "OK");
  } catch (err) {
    addMessage("assistant", `發生錯誤：${err.message}`);
    log(`chat failed: ${err.message}`, "FAIL");
  } finally {
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

threadInput.addEventListener("change", () => {
  threadId = threadInput.value.trim() || makeThreadId();
  threadInput.value = threadId;
  localStorage.setItem(threadKey, threadId);
  log(`thread updated: ${threadId}`, "DEBUG");
});

apiBaseInput.addEventListener("change", () => {
  const value = apiBaseInput.value.trim().replace(/\/+$/, "");
  apiBaseInput.value = value;
  localStorage.setItem(apiBaseKey, value);
  log(`backend api url updated: ${value || "same-origin"}`, "DEBUG");
  checkHealth();
});

selfTestButton.addEventListener("click", async () => {
  await sendMessage("SelfTest：請只回覆 OK，並用繁體中文。", "/api/selftest");
});

newThreadButton.addEventListener("click", () => {
  threadId = makeThreadId();
  threadInput.value = threadId;
  localStorage.setItem(threadKey, threadId);
  addMessage("assistant", `已切換新 Thread：${threadId}`);
  log(`new thread: ${threadId}`, "DEBUG");
});

clearButton.addEventListener("click", () => {
  messages.innerHTML = "";
  addMessage("assistant", "畫面已清空，不會刪除後端 memory。");
  log("screen cleared", "DEBUG");
});

log("UI initialized", "STEP");
checkHealth();
