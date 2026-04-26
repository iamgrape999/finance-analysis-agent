# -*- coding: utf-8 -*-
"""
Deployment-safe Finance/Analysis Agent wrapper.

This module preserves the public entrypoints used by the Colab version:
- chat_once(...)
- provider_readiness()
- provider_health_check(...)

The web deployment intentionally removes hard dependencies on Google Colab and
Google Drive. Secrets are read from environment variables, and memory is stored
under MEMORY_ROOT so Render can mount a persistent disk there.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(override=True)
except Exception:
    pass


DEBUG = os.getenv("DEBUG", "false").lower() == "true"

MEMORY_ROOT = os.environ.get("MEMORY_ROOT", "./data/memory")
os.makedirs(MEMORY_ROOT, exist_ok=True)

CHAT_FILENAME = "chat.jsonl"
USERS_MEMORY_DIRNAME = "users"
GLOBAL_MEMORY_DIRNAME = "_global"
GLOBAL_FACTS_FILENAME = "facts.json"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free").strip()

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama3.1-8b").strip()
CEREBRAS_MODEL_ALIASES = {
    "qwen/qwen3-235b-a22b": "qwen-3-235b-a22b-instruct-2507",
}

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-4-maverick-17b-128e-instruct").strip()
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1/chat/completions").strip()
NVIDIA_NEMOTRON_SUPER_MODEL = "nvidia/nemotron-3-super-120b-a12b"
NVIDIA_MODEL_ALIASES = {
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "nemotron-3-super-120b-a12b": NVIDIA_NEMOTRON_SUPER_MODEL,
    "qwen-3-235b-a22b-instruct-2507": "qwen/qwen3-coder-480b-a35b-instruct",
    "qwen/qwen-3-235b-a22b-instruct-2507": "qwen/qwen3-coder-480b-a35b-instruct",
    "qwen/qwen3-235b-a22b": "qwen/qwen3-coder-480b-a35b-instruct",
    "qwen3.5-122b-a10b": "qwen/qwen3.5-122b-a10b",
    "zai-glm-4.7": "z-ai/glm-4.7",
    "zai-org/glm-4.7": "z-ai/glm-4.7",
    "z-ai/glm4_7": "z-ai/glm-4.7",
    "glm4_7": "z-ai/glm-4.7",
    "glm-4.7": "z-ai/glm-4.7",
    "glm5_1": "z-ai/glm-5.1",
    "zai-glm5_1": "z-ai/glm-5.1",
    "z-ai/glm5_1": "z-ai/glm-5.1",
    "zai-glm-5.1": "z-ai/glm-5.1",
    "zai-org/glm-5.1": "z-ai/glm-5.1",
    "glm-5.1": "z-ai/glm-5.1",
    "minimax-m2.7": "minimaxai/minimax-m2.7",
    "minimax/minimax-m2.7": "minimaxai/minimax-m2.7",
    "kimi-k2-instruct": "moonshotai/kimi-k2-instruct",
    "kimi-k2-instruct-0905": "moonshotai/kimi-k2-instruct",
    "moonshotai/kimi-k2-instruct-0905": "moonshotai/kimi-k2-instruct",
    "kimi-k2.5": "moonshotai/kimi-k2.5",
    "kimi-k2p5": "moonshotai/kimi-k2.5",
    "deepseek-v3.2": "deepseek-ai/deepseek-v3.2",
    "deepseek-v3.1-terminus": "deepseek-ai/deepseek-v3.1-terminus",
    "deepseek-r1": "deepseek-ai/deepseek-r1",
    "step-3.5-flash": "stepfun-ai/step-3.5-flash",
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "gemma-4-31b-it": "google/gemma-4-31b-it",
    "mistral-large-3.675b-instruct-2512": "mistralai/mistral-large-2411",
    "mistralai/mistral-large-3.675b-instruct-2512": "mistralai/mistral-large-2411",
    "mistral-nemotron": "mistralai/mistral-nemotron",
    "qwen3-coder-480b-a35b-instruct": "qwen/qwen3-coder-480b-a35b-instruct",
}

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()

CF_API_TOKEN = os.getenv("CF_API_TOKEN", os.getenv("CF_API_KEY", "")).strip()
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "").strip()
CF_MODEL = os.getenv("CF_MODEL", "@cf/meta/llama-3.1-8b-instruct").strip()

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "").strip()
FIREWORKS_MODEL = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/minimax-m2p7").strip()
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1").rstrip("/")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")).strip()
AWS_BEDROCK_MODEL = os.getenv("AWS_BEDROCK_MODEL", "anthropic.claude-3-haiku-20240307-v1:0").strip()

AGENT_PROVIDER_ORDER = os.getenv("AGENT_PROVIDER_ORDER", "cerebras,groq,mistral,gemini,openrouter,cloudflare,fireworks,aws,nvidia")
AGENT_PROVIDER_DISABLE = os.getenv("AGENT_PROVIDER_DISABLE", "").lower()
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))
REQUEST_TIMEOUT_SEC = int(os.getenv("REQUEST_TIMEOUT_SEC", "120"))
DEFAULT_PROVIDER_TIMEOUT_SEC = int(os.getenv("DEFAULT_PROVIDER_TIMEOUT_SEC", os.getenv("CHAT_PROVIDER_TIMEOUT_SEC", "18")))
OPENROUTER_TIMEOUT_SEC = int(os.getenv("OPENROUTER_TIMEOUT_SEC", "25"))
PROVIDER_READ_TIMEOUTS = {
    "cerebras": int(os.getenv("CEREBRAS_TIMEOUT_SEC", "5")),
    "groq": int(os.getenv("GROQ_TIMEOUT_SEC", "8")),
    "gemini": int(os.getenv("GEMINI_TIMEOUT_SEC", "25")),
    "mistral": int(os.getenv("MISTRAL_TIMEOUT_SEC", "6")),
    "openrouter": OPENROUTER_TIMEOUT_SEC,
    "cloudflare": int(os.getenv("CLOUDFLARE_TIMEOUT_SEC", "10")),
    "fireworks": int(os.getenv("FIREWORKS_TIMEOUT_SEC", "15")),
    "nvidia": int(os.getenv("NVIDIA_TIMEOUT_SEC", "30")),
    "aws": int(os.getenv("AWS_TIMEOUT_SEC", str(DEFAULT_PROVIDER_TIMEOUT_SEC))),
    "default": DEFAULT_PROVIDER_TIMEOUT_SEC,
}
TOKEN_EST_RATIO = float(os.getenv("TOKEN_EST_RATIO", "1.0"))
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "9000"))
MAX_USER_MESSAGE_CHARS = int(os.getenv("MAX_USER_MESSAGE_CHARS", "20000"))
MAX_HISTORY_TURN_CHARS = int(os.getenv("MAX_HISTORY_TURN_CHARS", "800"))
MAX_SUMMARY_CHARS = int(os.getenv("MAX_SUMMARY_CHARS", "1200"))
MAX_GLOBAL_MEMORY_CHARS = int(os.getenv("MAX_GLOBAL_MEMORY_CHARS", "1600"))
LIGHT_PROVIDER_MAX_PROMPT_TOKENS = int(os.getenv("LIGHT_PROVIDER_MAX_PROMPT_TOKENS", "1000"))
GROQ_MAX_OUTPUT_TOKENS = int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", "1536"))
CLOUDFLARE_MAX_OUTPUT_TOKENS = int(os.getenv("CLOUDFLARE_MAX_OUTPUT_TOKENS", "1536"))
LIGHT_SUMMARY_CHARS = int(os.getenv("LIGHT_SUMMARY_CHARS", "480"))
LIGHT_GLOBAL_MEMORY_CHARS = int(os.getenv("LIGHT_GLOBAL_MEMORY_CHARS", "320"))
LIGHT_USER_MESSAGE_CHARS = int(os.getenv("LIGHT_USER_MESSAGE_CHARS", "1600"))
LIGHT_HISTORY_TURNS = int(os.getenv("LIGHT_HISTORY_TURNS", "0"))
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").lower() == "true"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))
WEB_SEARCH_TIMEOUT_SEC = int(os.getenv("WEB_SEARCH_TIMEOUT_SEC", "8"))
WEB_SEARCH_MAX_CONTEXT_CHARS = int(os.getenv("WEB_SEARCH_MAX_CONTEXT_CHARS", "2400"))
LIGHT_WEB_SEARCH_MAX_CONTEXT_CHARS = int(os.getenv("LIGHT_WEB_SEARCH_MAX_CONTEXT_CHARS", "900"))
WEB_SEARCH_SEARCH_DEPTH = os.getenv("WEB_SEARCH_SEARCH_DEPTH", "basic").strip() or "basic"
WEB_SEARCH_TOPIC = os.getenv("WEB_SEARCH_TOPIC", "general").strip() or "general"
WEB_SEARCH_PROVIDER_ORDER = [
    item.strip().lower()
    for item in os.getenv("WEB_SEARCH_PROVIDER_ORDER", "tavily,serper").split(",")
    if item.strip()
]
SERPER_GL = os.getenv("SERPER_GL", "tw").strip() or "tw"
SERPER_HL = os.getenv("SERPER_HL", "zh-tw").strip() or "zh-tw"
SEARCH_PRIORITY_1_KEYWORDS = {
    "今天", "現在", "目前", "剛剛", "最新",
    "最近", "本週", "本月", "今年", "昨天",
    "明天", "這幾天", "上週", "最新消息", "剛出",
    "today", "now", "current", "currently", "latest",
    "recent", "recently", "this week", "this month", "this year",
    "yesterday", "tomorrow", "just released", "just announced", "breaking",
    "live", "real-time", "updated", "as of", "2026",
    "股價", "匯率", "油價", "金價", "利率",
    "指數", "天氣", "氣溫", "空氣品質", "地震",
    "颱風", "停電", "停水",
    "stock price", "exchange rate", "weather", "temperature", "earthquake",
    "hurricane", "typhoon", "oil price", "gold price", "interest rate",
    "inflation rate", "unemployment rate",
}
SEARCH_PRIORITY_2_KEYWORDS = {
    "版本", "更新", "下架", "發布", "上市", "漏洞", "當機", "維護",
    "version", "update", "release", "deprecated", "end of life",
    "patch", "vulnerability", "outage", "status", "downtime",
    "changelog", "roadmap", "beta", "stable release", "EOL", "CVE",
    "多少錢", "費用", "票價", "學費", "房租",
    "price", "cost", "how much", "pricing", "rate", "subscription",
}
SEARCH_PRIORITY_3_KEYWORDS = {
    "法規", "政策", "規定", "申請", "條件", "期限", "表格",
    "regulation", "policy", "requirement", "eligibility",
    "deadline", "application", "compliance", "official",
    "排名", "最好的", "推薦", "比較",
    "best", "top", "ranking", "vs", "benchmark", "leaderboard",
}
NO_SEARCH_PATTERNS = {
    "什麼是", "定義", "解釋", "原理",
    "what is", "define", "explain", "how does",
    "計算", "算出", "等於",
    "calculate", "compute", "equals",
    "寫一篇", "幫我寫", "翻譯",
    "write", "translate", "summarize",
    "歷史上", "過去", "曾經",
    "historically", "in the past", "used to",
}
WEB_SEARCH_EXTRA_KEYWORDS = [
    kw.strip() for kw in os.getenv("WEB_SEARCH_EXTRA_KEYWORDS", "").split(",") if kw.strip()
]
AUTO_CONTINUE_ENABLED = os.getenv("AUTO_CONTINUE_ENABLED", "true").lower() == "true"
AUTO_CONTINUE_MAX_ROUNDS = int(os.getenv("AUTO_CONTINUE_MAX_ROUNDS", "2"))
AUTO_CONTINUE_TAIL_CHARS = int(os.getenv("AUTO_CONTINUE_TAIL_CHARS", "1200"))
FAST_MODE_ROUTE_TIMEOUT_SEC = int(os.getenv("FAST_MODE_ROUTE_TIMEOUT_SEC", "35"))
STABLE_MODE_ROUTE_TIMEOUT_SEC = int(os.getenv("STABLE_MODE_ROUTE_TIMEOUT_SEC", "55"))
DEEP_MODE_ROUTE_TIMEOUT_SEC = int(os.getenv("DEEP_MODE_ROUTE_TIMEOUT_SEC", "75"))
FAST_MODE_CONTINUE_ROUNDS = int(os.getenv("FAST_MODE_CONTINUE_ROUNDS", "0"))
STABLE_MODE_CONTINUE_ROUNDS = int(os.getenv("STABLE_MODE_CONTINUE_ROUNDS", "1"))
DEEP_MODE_CONTINUE_ROUNDS = int(os.getenv("DEEP_MODE_CONTINUE_ROUNDS", str(AUTO_CONTINUE_MAX_ROUNDS)))

SYSTEM_POLICY = (
    "你是 CathyChang AI，請用繁體中文回答。"
    "你可以協助一般知識、生活、娛樂、商業、財務、投資與資料分析問題。"
    "若問題涉及財務、投資、法律、醫療或其他高風險決策，請提醒使用者需要自行查證或諮詢專業人士。"
    "回答要清楚、友善、具體；資料不足時請明確說明，不要編造。"
    "不要輸出內部推理過程、<think> 標籤或隱藏思考內容。"
)

_CALL_CONTEXT: threading.local = threading.local()


def _run_with_timeout(fn: Any, timeout_sec: float) -> Any:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=max(1.0, float(timeout_sec)))
        except concurrent.futures.TimeoutError:
            raise RuntimeError(f"Call timed out after {timeout_sec:.0f}s")


def _ctx_model(provider_key: str, env_key: str, default_val: str) -> str:
    overrides = getattr(_CALL_CONTEXT, "model_overrides", {}) or {}
    return overrides.get(provider_key) or os.getenv(env_key, default_val).strip() or default_val


@dataclass
class ModelReply:
    text: str
    meta: Dict[str, Any]


def debug_log(*parts: Any) -> None:
    if DEBUG:
        print("[DEBUG]", *parts)


def _sanitize_thread_id(value: str) -> str:
    value = value or "THREAD_DEFAULT"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:80] or "THREAD_DEFAULT"


def _sanitize_user_id(value: str) -> str:
    value = value or "default"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())[:80] or "default"


def _memory_base(user_id: Optional[str] = None) -> str:
    if not user_id:
        return MEMORY_ROOT
    base = os.path.join(MEMORY_ROOT, USERS_MEMORY_DIRNAME, _sanitize_user_id(user_id))
    os.makedirs(base, exist_ok=True)
    return base


def _lock_path(path: str) -> str:
    return path + ".lock"


def _acquire_lock(path: str, ttl_sec: int = 120, max_wait_sec: float = 60.0) -> str:
    lock = _lock_path(path)
    payload = {"pid": os.getpid(), "ts": time.time(), "path": path}
    deadline = time.monotonic() + max_wait_sec
    while True:
        try:
            if os.path.exists(lock) and time.time() - os.path.getmtime(lock) > ttl_sec:
                os.remove(lock)
        except Exception:
            pass

        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            return lock
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Could not acquire lock on {path} within {max_wait_sec}s")
            time.sleep(0.05)


def _release_lock(lock: str) -> None:
    try:
        if os.path.exists(lock):
            os.remove(lock)
    except Exception:
        pass


def _atomic_write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock = _acquire_lock(path)
    try:
        fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
    finally:
        _release_lock(lock)


PII_PATTERNS = [
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    (re.compile(r"\b09\d{8}\b"), "[MOBILE_TW]"),
    (re.compile(r"\b[A-Z][12]\d{8}\b"), "[TW_ID]"),
    (re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{4}\b"), "[CARD]"),
]


def mask_pii(text: str) -> str:
    out = text or ""
    for pattern, repl in PII_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def strip_redundant(text: str, max_len: int = 200_000) -> str:
    if text is None:
        return ""
    out = str(text)
    if len(out) > max_len:
        out = out[:max_len]
    out = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", out)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"[ \u3000]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def token_est(text: Any) -> int:
    try:
        return max(1, int(len(str(text or "")) * TOKEN_EST_RATIO))
    except Exception:
        return 1


def _clip_text(text: str, max_chars: int, label: str = "") -> str:
    text = str(text or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    suffix = f"\n\n[系統註記] {label or '內容'}已因長度限制截斷。"
    keep = max(0, int(max_chars) - len(suffix))
    return text[:keep].rstrip() + suffix


def _extract_prompt_sections(prompt: str) -> Dict[str, str]:
    sections = {"global": "", "summary": "", "history": "", "web": "", "user": "", "preamble": ""}
    current = "preamble"
    buffer: List[str] = []
    markers = {
        "[全域記憶]": "global",
        "[對話摘要]": "summary",
        "[最近對話]": "history",
        "[即時網路搜尋結果]": "web",
        "[本輪使用者問題]": "user",
    }

    def _flush() -> None:
        nonlocal buffer, current
        text = "\n".join(buffer).strip()
        if text:
            existing = sections.get(current, "")
            sections[current] = (existing + "\n" + text).strip() if existing else text
        buffer = []

    for raw_line in str(prompt or "").splitlines():
        line = raw_line.strip()
        if line in markers:
            _flush()
            current = markers[line]
            continue
        buffer.append(raw_line)
    _flush()
    return sections


def _compact_prompt_for_provider(prompt: str, provider: str) -> Tuple[str, Dict[str, Any]]:
    provider = (provider or "").strip().lower()
    if provider not in {"groq", "cloudflare"}:
        return prompt, {"mode": "full", "prompt_tokens": token_est(prompt)}

    sections = _extract_prompt_sections(prompt)
    header = (
        "[系統註記] 為降低 413 / timeout 風險，已自動省略較長的歷史脈絡；"
        "請優先根據本輪問題與必要摘要回答。"
    )

    candidate_parts: List[str] = []
    if sections.get("user"):
        candidate_parts.append("[本輪使用者問題]\n" + sections["user"])
    if sections.get("web"):
        candidate_parts.insert(0, "[即時網路搜尋結果]\n" + sections["web"])
    if sections.get("summary"):
        candidate_parts.insert(0, "[對話摘要]\n" + sections["summary"])
    if sections.get("global"):
        candidate_parts.insert(0, "[全域記憶]\n" + sections["global"])

    compact = "\n\n".join([header] + candidate_parts).strip()

    # Groq / Cloudflare 優先走「無歷史」版本，真的還太長才再縮 user。
    if token_est(compact) > LIGHT_PROVIDER_MAX_PROMPT_TOKENS:
        user_text = sections.get("user") or prompt
        web_text = sections.get("web") or ""
        max_user_chars = max(1000, int(LIGHT_PROVIDER_MAX_PROMPT_TOKENS / max(TOKEN_EST_RATIO, 0.1)) - 400)
        compact_parts = [header]
        if web_text:
            compact_parts.append("[即時網路搜尋結果]\n" + _clip_text(web_text, min(600, max_user_chars // 2), f"{provider} 搜尋結果"))
        compact_parts.append("[本輪使用者問題]\n" + _clip_text(user_text, max_user_chars, f"{provider} 轻量輸入"))
        compact = "\n\n".join(compact_parts)
        return compact, {
            "mode": "user_only",
            "prompt_tokens": token_est(compact),
            "light_budget": LIGHT_PROVIDER_MAX_PROMPT_TOKENS,
        }

    return compact, {
        "mode": "no_history",
        "prompt_tokens": token_est(compact),
        "light_budget": LIGHT_PROVIDER_MAX_PROMPT_TOKENS,
    }


def search_intent_classifier(query: str) -> Dict[str, Any]:
    text = strip_redundant(query or "")
    lowered = text.lower()
    if not WEB_SEARCH_ENABLED or not (TAVILY_API_KEY or SERPER_API_KEY):
        return {
            "must_search": False,
            "confidence": 0.0,
            "reason": "web search disabled",
            "priority": "disabled",
            "matched_keyword": "",
        }
    if not text:
        return {
            "must_search": False,
            "confidence": 0.0,
            "reason": "empty query",
            "priority": "empty",
            "matched_keyword": "",
        }

    explicit_patterns = [r"查(一下|詢|查)", r"搜尋", r"\bsearch\b", r"\blook up\b"]
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in explicit_patterns):
        return {
            "must_search": True,
            "confidence": 0.98,
            "reason": "explicit search request",
            "priority": "explicit",
            "matched_keyword": "explicit_search",
        }

    for pattern in NO_SEARCH_PATTERNS:
        if pattern.lower() in lowered:
            return {
                "must_search": False,
                "confidence": 0.99,
                "reason": f"no-search pattern: {pattern}",
                "priority": "blocked",
                "matched_keyword": pattern,
            }

    priority_1 = list(SEARCH_PRIORITY_1_KEYWORDS) + WEB_SEARCH_EXTRA_KEYWORDS
    for keyword in priority_1:
        if keyword.lower() in lowered:
            return {
                "must_search": True,
                "confidence": 0.95,
                "reason": f"即時資訊需求：{keyword}",
                "priority": "p1",
                "matched_keyword": keyword,
            }

    for keyword in SEARCH_PRIORITY_2_KEYWORDS:
        if keyword.lower() in lowered:
            return {
                "must_search": True,
                "confidence": 0.75,
                "reason": f"時效性資訊：{keyword}",
                "priority": "p2",
                "matched_keyword": keyword,
            }

    for keyword in SEARCH_PRIORITY_3_KEYWORDS:
        if keyword.lower() in lowered:
            return {
                "must_search": False,
                "confidence": 0.50,
                "reason": f"可能需要更新資訊：{keyword}",
                "priority": "p3",
                "matched_keyword": keyword,
            }

    return {
        "must_search": False,
        "confidence": 0.05,
        "reason": "純知識性問題，LLM 直接回答",
        "priority": "none",
        "matched_keyword": "",
    }


def _search_web_tavily(query: str) -> Dict[str, Any]:
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not configured")
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": strip_redundant(query or ""),
        "topic": WEB_SEARCH_TOPIC,
        "search_depth": WEB_SEARCH_SEARCH_DEPTH,
        "max_results": int(WEB_SEARCH_MAX_RESULTS),
        "include_answer": False,
        "include_raw_content": False,
    }
    response = requests.post(
        "https://api.tavily.com/search",
        json=payload,
        timeout=(5, int(WEB_SEARCH_TIMEOUT_SEC)),
    )
    if response.status_code >= 400:
        body = (response.text or "")[:800].replace("\n", "\\n").replace("\r", "\\r")
        raise RuntimeError(f"Tavily error {response.status_code}: {body}")
    data = response.json() or {}
    normalized_results: List[Dict[str, Any]] = []
    for item in (data.get("results") or [])[: int(WEB_SEARCH_MAX_RESULTS)]:
        normalized_results.append(
            {
                "title": str(item.get("title") or "").strip(),
                "url": str(item.get("url") or "").strip(),
                "content": strip_redundant(str(item.get("content") or "")),
                "score": item.get("score"),
            }
        )
    return {
        "provider": "tavily",
        "query": payload["query"],
        "results": normalized_results,
        "response_time": data.get("response_time"),
    }


def _search_web_serper(query: str) -> Dict[str, Any]:
    if not SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY is not configured")
    payload = {
        "q": strip_redundant(query or ""),
        "gl": SERPER_GL,
        "hl": SERPER_HL,
        "num": int(WEB_SEARCH_MAX_RESULTS),
    }
    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        json=payload,
        timeout=(5, int(WEB_SEARCH_TIMEOUT_SEC)),
    )
    if response.status_code >= 400:
        body = (response.text or "")[:800].replace("\n", "\\n").replace("\r", "\\r")
        raise RuntimeError(f"Serper error {response.status_code}: {body}")
    data = response.json() or {}
    normalized_results: List[Dict[str, Any]] = []
    for item in (data.get("organic") or [])[: int(WEB_SEARCH_MAX_RESULTS)]:
        normalized_results.append(
            {
                "title": str(item.get("title") or "").strip(),
                "url": str(item.get("link") or "").strip(),
                "content": strip_redundant(str(item.get("snippet") or "")),
                "score": item.get("position"),
            }
        )
    return {
        "provider": "serper",
        "query": payload["q"],
        "results": normalized_results,
        "response_time": None,
    }


def search_web(query: str, provider_override: Optional[str] = None) -> Dict[str, Any]:
    if not (TAVILY_API_KEY or SERPER_API_KEY):
        raise RuntimeError("No web search provider is configured")
    provider_override = (provider_override or "").strip().lower()
    if provider_override in {"tavily", "serper"}:
        providers = [provider_override]
    else:
        providers = list(WEB_SEARCH_PROVIDER_ORDER)
    errors: List[str] = []
    attempts: List[str] = []
    for provider in providers:
        if provider == "tavily":
            try:
                data = _search_web_tavily(query)
                attempts.append("tavily")
                data["attempts"] = attempts[:]
                if errors:
                    data["fallback_errors"] = errors
                return data
            except Exception as exc:
                attempts.append("tavily")
                errors.append(f"tavily -> {type(exc).__name__}: {exc}")
                continue
        if provider == "serper":
            try:
                data = _search_web_serper(query)
                attempts.append("serper")
                data["attempts"] = attempts[:]
                if errors:
                    data["fallback_errors"] = errors
                return data
            except Exception as exc:
                attempts.append("serper")
                errors.append(f"serper -> {type(exc).__name__}: {exc}")
                continue
    raise RuntimeError(" ; ".join(errors) if errors else "No web search provider could answer")


def _format_web_search_context(search_meta: Dict[str, Any], max_chars: int) -> str:
    results = list(search_meta.get("results") or [])
    if not results:
        return ""
    lines = [
        "[即時網路搜尋結果]",
        "以下資訊來自即時網路搜尋；若與既有知識衝突，請優先採用較新的來源，並在回答中提到來源。",
    ]
    for idx, item in enumerate(results, start=1):
        title = str(item.get("title") or f"結果 {idx}")
        url = str(item.get("url") or "")
        content = _clip_text(str(item.get("content") or ""), 700, f"搜尋結果 {idx}")
        lines.append(f"{idx}. {title}")
        if url:
            lines.append(f"來源：{url}")
        if content:
            lines.append(f"摘要：{content}")
    return _clip_text("\n".join(lines), max_chars, "即時網路搜尋結果")


SYSTEM_POLICY_LIGHT = (
    "你是 CathyChang AI，請用繁體中文回答。"
    "資料不足時請直接說明。"
    "不要輸出<think>、內部推理或多餘說明。"
)


def _provider_request_profile(provider: str, prompt: str) -> Dict[str, Any]:
    provider = (provider or "").strip().lower()
    prompt_tokens = token_est(prompt)
    if provider in {"groq", "cloudflare"}:
        if prompt_tokens > LIGHT_PROVIDER_MAX_PROMPT_TOKENS:
            return {
                "skip": True,
                "reason": f"prompt_too_large:{prompt_tokens}>{LIGHT_PROVIDER_MAX_PROMPT_TOKENS}",
                "system_policy": SYSTEM_POLICY_LIGHT,
                "max_tokens": min(768, GROQ_MAX_OUTPUT_TOKENS if provider == "groq" else CLOUDFLARE_MAX_OUTPUT_TOKENS),
                "prompt_tokens": prompt_tokens,
            }
        return {
            "skip": False,
            "reason": "light_ok",
            "system_policy": SYSTEM_POLICY_LIGHT,
            "max_tokens": min(1024, GROQ_MAX_OUTPUT_TOKENS if provider == "groq" else CLOUDFLARE_MAX_OUTPUT_TOKENS),
            "prompt_tokens": prompt_tokens,
        }
    return {
        "skip": False,
        "reason": "default",
        "system_policy": SYSTEM_POLICY,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "prompt_tokens": prompt_tokens,
    }


def _estimate_request_bytes(system_policy: str, prompt: str, max_tokens: int, temperature: float) -> int:
    payload = {
        "messages": [
            {"role": "system", "content": system_policy},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def _usage_completion_tokens(meta: Dict[str, Any]) -> int:
    usage = (meta or {}).get("usage") or {}
    if not isinstance(usage, dict):
        return 0
    value = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("candidates_token_count")
        or 0
    )
    try:
        return int(value)
    except Exception:
        return 0


def _looks_truncated_text(text: str) -> bool:
    t = strip_redundant(text or "")
    if not t:
        return False
    if t.endswith(("。", "！", "？", ".", "!", "?", "」", "』", "）", ")", "]", "】")):
        return False
    if t.endswith(("|", "：", ":", "、", "，", ",")):
        return True
    return True


def _needs_auto_continue(reply: ModelReply, max_tokens: int) -> bool:
    if not AUTO_CONTINUE_ENABLED:
        return False
    text = strip_redundant(reply.text or "")
    if not text:
        return False
    completion_tokens = _usage_completion_tokens(reply.meta or {})
    hit_output_cap = completion_tokens >= max(64, int(max_tokens) - 16)
    return hit_output_cap and _looks_truncated_text(text)


def _build_continue_prompt(base_prompt: str, partial_text: str) -> str:
    tail = strip_redundant(partial_text or "")[-AUTO_CONTINUE_TAIL_CHARS:]
    return (
        f"{base_prompt}\n\n"
        "[續寫要求]\n"
        "上一則回答因輸出長度限制被截斷。請直接從中斷處續寫，不要重複前文，不要重新開頭。\n"
        "若上一段是表格，請從尚未完成的列繼續補完；若上一段是條列，請接著補完剩餘項目。\n"
        "只輸出續寫內容。\n\n"
        "[上一版回答尾段]\n"
        f"{tail}"
    )


def _call_provider_once(
    provider: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    provider_profile: Optional[Dict[str, Any]] = None,
    timeout_override: Optional[float] = None,
) -> Optional[ModelReply]:
    effective_max_tokens = min(int(max_tokens), int((provider_profile or {}).get("max_tokens", max_tokens)))
    effective_system_policy = str((provider_profile or {}).get("system_policy", SYSTEM_POLICY))
    if provider == "openrouter":
        return call_openrouter(prompt, max_tokens=effective_max_tokens, temperature=temperature, timeout_override=timeout_override)
    if provider == "nvidia":
        return call_nvidia(prompt, max_tokens=effective_max_tokens, temperature=temperature, timeout_override=timeout_override)
    if provider == "cerebras":
        return call_cerebras(prompt, max_tokens=effective_max_tokens, temperature=temperature, timeout_override=timeout_override)
    if provider == "mistral":
        return call_mistral(prompt, max_tokens=effective_max_tokens, temperature=temperature, timeout_override=timeout_override)
    if provider == "gemini":
        return call_gemini(prompt, max_tokens=effective_max_tokens, temperature=temperature, timeout_override=timeout_override)
    if provider == "cloudflare":
        return call_cloudflare(
            prompt,
            max_tokens=effective_max_tokens,
            temperature=temperature,
            system_policy=effective_system_policy,
            timeout_override=timeout_override,
        )
    if provider == "groq":
        return call_groq(
            prompt,
            max_tokens=effective_max_tokens,
            temperature=temperature,
            system_policy=effective_system_policy,
            timeout_override=timeout_override,
        )
    if provider == "fireworks":
        return call_fireworks(prompt, max_tokens=effective_max_tokens, temperature=temperature, timeout_override=timeout_override)
    if provider == "aws":
        return call_aws(prompt, max_tokens=effective_max_tokens, temperature=temperature, timeout_override=timeout_override)
    return None


def clean_model_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


class MemoryAdapter:
    def __init__(self, memory: Any = None, THREAD_ID: str = "THREAD_DEFAULT", USER_ID: Optional[str] = None):
        del memory
        self.user_id = _sanitize_user_id(USER_ID) if USER_ID else ""
        self.thread_id = _sanitize_thread_id(THREAD_ID)
        self.thread_dir = os.path.join(_memory_base(self.user_id), self.thread_id)
        os.makedirs(self.thread_dir, exist_ok=True)
        self.chat_path = os.path.join(self.thread_dir, CHAT_FILENAME)
        self.summary_path = os.path.join(self.thread_dir, "summary.md")

    def load_chat_raw(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.chat_path):
            return []
        rows: List[Dict[str, Any]] = []
        with open(self.chat_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except Exception:
                    continue
        return rows

    def save_chat(self, rows: List[Dict[str, Any]]) -> None:
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        _atomic_write(self.chat_path, text)

    def add_turn(self, role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be user or assistant")
        record = {
            "ts": datetime.now().isoformat(),
            "role": role,
            "content": content or "",
            "meta": meta or {},
        }
        os.makedirs(self.thread_dir, exist_ok=True)
        lock = _acquire_lock(self.chat_path)
        try:
            with open(self.chat_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        finally:
            _release_lock(lock)

    def get_recent_turns(self, n: int = 8) -> List[Dict[str, Any]]:
        rows = self.load_chat_raw()
        return rows[-max(0, int(n)) :]

    def load_summary(self) -> str:
        if not os.path.exists(self.summary_path):
            return ""
        try:
            with open(self.summary_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""

    def save_summary(self, text: str) -> None:
        _atomic_write(self.summary_path, text or "")


def _global_facts_path(user_id: Optional[str] = None) -> str:
    root = os.path.join(_memory_base(user_id), GLOBAL_MEMORY_DIRNAME)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, GLOBAL_FACTS_FILENAME)


def load_global_facts(user_id: Optional[str] = None) -> Dict[str, Any]:
    path = _global_facts_path(user_id)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_global_facts(facts: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
    clean = {}
    for key, value in (facts or {}).items():
        k = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(key).strip())[:80]
        if not k:
            continue
        clean[k] = {
            "value": str(value.get("value", "")) if isinstance(value, dict) else str(value),
            "ts": str(value.get("ts", datetime.now().isoformat())) if isinstance(value, dict) else datetime.now().isoformat(),
        }
    _atomic_write(_global_facts_path(user_id), json.dumps(clean, ensure_ascii=False, indent=2))
    return clean


def upsert_global_fact(key: str, value: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    facts = load_global_facts(user_id)
    raw_key = str(key).strip()
    if "=" in raw_key and not str(value).strip():
        raw_key, value = raw_key.split("=", 1)
    k = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_key.strip())[:80]
    if not k:
        return facts
    facts[k] = {"value": str(value).strip(), "ts": datetime.now().isoformat()}
    return save_global_facts(facts, user_id)


def delete_global_fact(key: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    facts = load_global_facts(user_id)
    facts.pop(str(key).strip(), None)
    return save_global_facts(facts, user_id)


def upsert_global_facts_from_text(text: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    t = text or ""
    if not re.search(r"(請)?記住|remember|全域記憶|長期記憶", t, flags=re.IGNORECASE):
        return load_global_facts(user_id)
    facts = load_global_facts(user_id)
    now = datetime.now().isoformat()
    for key, value in re.findall(r"([A-Za-z][A-Za-z0-9_.-]{1,79})\s*=\s*([^\s,;，；。]+)", t):
        facts[key] = {"value": value.strip()[:500], "ts": now}
    return save_global_facts(facts, user_id)


def list_thread_summaries(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    base = _memory_base(user_id)
    if not os.path.isdir(base):
        return []
    out: List[Dict[str, Any]] = []
    for name in os.listdir(base):
        if name in {GLOBAL_MEMORY_DIRNAME, USERS_MEMORY_DIRNAME}:
            continue
        thread_dir = os.path.join(base, name)
        chat_path = os.path.join(thread_dir, CHAT_FILENAME)
        if not os.path.isdir(thread_dir) or not os.path.exists(chat_path):
            continue
        try:
            with open(chat_path, "rb") as _fh:
                _fh.seek(0, 2)
                size = _fh.tell()
                _fh.seek(max(0, size - 8192))
                tail_bytes = _fh.read()
        except OSError:
            continue
        tail_lines = [ln for ln in tail_bytes.decode("utf-8", errors="replace").splitlines() if ln.strip()]
        if not tail_lines:
            continue
        last_row: Dict[str, Any] = {}
        for ln in reversed(tail_lines):
            try:
                last_row = json.loads(ln)
                break
            except Exception:
                pass
        preview = ""
        try:
            with open(chat_path, "r", encoding="utf-8", errors="replace") as _fh2:
                for ln in _fh2:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        row = json.loads(ln)
                    except Exception:
                        continue
                    if row.get("role") == "user" and row.get("content"):
                        preview = strip_redundant(str(row["content"]))[:120]
                        break
        except OSError:
            pass
        message_count = sum(1 for ln in tail_lines if ln.strip())
        updated_at = last_row.get("ts", "")
        try:
            mtime = os.path.getmtime(chat_path)
        except Exception:
            mtime = 0
        out.append({
            "thread_id": name,
            "updated_at": updated_at,
            "message_count": message_count,
            "preview": preview or name,
            "mtime": mtime,
        })
    out.sort(key=lambda item: float(item.get("mtime") or 0), reverse=True)
    for item in out:
        item.pop("mtime", None)
    return out


def delete_thread_memory(thread_id: str, user_id: Optional[str] = None) -> bool:
    safe_id = _sanitize_thread_id(thread_id)
    if not safe_id or safe_id in {GLOBAL_MEMORY_DIRNAME, USERS_MEMORY_DIRNAME}:
        return False
    base = _memory_base(user_id)
    target = os.path.abspath(os.path.join(base, safe_id))
    root = os.path.abspath(base)
    if not (target == root or target.startswith(root + os.sep)):
        return False
    if not os.path.isdir(target):
        return False
    shutil.rmtree(target)
    return True


def provider_readiness() -> Dict[str, bool]:
    return {
        "nvidia": bool(NVIDIA_API_KEY),
        "cerebras": bool(CEREBRAS_API_KEY),
        "mistral": bool(MISTRAL_API_KEY),
        "openrouter": bool(OPENROUTER_API_KEY),
        "gemini": bool(GEMINI_API_KEY),
        "cloudflare": bool(CF_API_TOKEN and CF_ACCOUNT_ID),
        "groq": bool(GROQ_API_KEY),
        "fireworks": bool(FIREWORKS_API_KEY),
        "aws": bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY),
    }


def provider_diagnostics() -> Dict[str, Any]:
    raw_nvidia_model = os.getenv("NVIDIA_MODEL", NVIDIA_MODEL).strip() or NVIDIA_MODEL
    raw_cerebras_model = os.getenv("CEREBRAS_MODEL", CEREBRAS_MODEL).strip() or CEREBRAS_MODEL
    diag: Dict[str, Any] = {
        "web_search_enabled": bool(WEB_SEARCH_ENABLED),
        "tavily_key_present": bool(TAVILY_API_KEY),
        "serper_key_present": bool(SERPER_API_KEY),
        "web_search_provider_order": list(WEB_SEARCH_PROVIDER_ORDER),
        "web_search_max_results": int(WEB_SEARCH_MAX_RESULTS),
        "nvidia_key_present": bool(NVIDIA_API_KEY),
        "nvidia_model": raw_nvidia_model,
        "nvidia_raw_model": raw_nvidia_model,
        "nvidia_effective_model": _resolve_nvidia_model(raw_nvidia_model),
        "nvidia_base_url": os.getenv("NVIDIA_BASE_URL", NVIDIA_BASE_URL).strip() or NVIDIA_BASE_URL,
        "cerebras_key_present": bool(CEREBRAS_API_KEY),
        "cerebras_model": raw_cerebras_model,
        "cerebras_raw_model": raw_cerebras_model,
        "cerebras_effective_model": CEREBRAS_MODEL_ALIASES.get(raw_cerebras_model, raw_cerebras_model),
        "mistral_import_ok": True,
        "mistral_import_error": "",
        "mistral_client_mode": "http",
        "mistral_key_present": bool(MISTRAL_API_KEY),
        "mistral_model": os.getenv("MISTRAL_MODEL", MISTRAL_MODEL).strip() or MISTRAL_MODEL,
        "mistral_base_url": os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1/chat/completions").strip(),
    }
    return diag


def resolve_provider_order() -> List[str]:
    ready = provider_readiness()
    provider_order = (
        getattr(_CALL_CONTEXT, "provider_order_override", None)
        or os.getenv("AGENT_PROVIDER_ORDER", AGENT_PROVIDER_ORDER)
    )
    provider_disable = os.getenv("AGENT_PROVIDER_DISABLE", AGENT_PROVIDER_DISABLE).lower()
    disabled = {p.strip() for p in provider_disable.split(",") if p.strip()}
    out: List[str] = []
    for raw in provider_order.split(","):
        provider = raw.strip().lower()
        if provider and ready.get(provider) and provider not in disabled and provider not in out:
            out.append(provider)
    return out


def list_nvidia_free_models() -> List[Dict[str, Any]]:
    return [
        {
            "name": "meta/llama-4-maverick-17b-128e-instruct",
            "display_name": "Llama 4 Maverick 17B 128E Instruct",
            "tier": "production",
            "free_endpoint": True,
        },
        {
            "name": "openai/gpt-oss-120b",
            "display_name": "OpenAI GPT OSS 120B",
            "tier": "production",
            "free_endpoint": True,
        },
        {
            "name": NVIDIA_NEMOTRON_SUPER_MODEL,
            "display_name": "NVIDIA Nemotron 3 Super 120B A12B",
            "tier": "preview",
            "free_endpoint": True,
            "attempt_timeout_sec": 30,
        },
        {
            "name": "qwen/qwen3-coder-480b-a35b-instruct",
            "display_name": "Qwen3 Coder 480B A35B Instruct",
            "tier": "preview",
            "free_endpoint": True,
        },
        {
            "name": "qwen/qwen3.5-122b-a10b",
            "display_name": "Qwen 3.5 122B A10B",
            "tier": "preview",
            "free_endpoint": True,
        },
        {
            "name": "z-ai/glm-4.7",
            "display_name": "Z.ai GLM 4.7",
            "tier": "preview",
            "free_endpoint": True,
        },
        {
            "name": "z-ai/glm-5.1",
            "display_name": "Z.ai GLM 5.1",
            "tier": "preview",
            "free_endpoint": True,
        },
        {
            "name": "minimaxai/minimax-m2.7",
            "display_name": "MiniMax M2.7",
            "tier": "preview",
            "free_endpoint": True,
        },
        {
            "name": "moonshotai/kimi-k2-instruct",
            "display_name": "Moonshot Kimi K2 Instruct",
            "tier": "preview",
            "free_endpoint": True,
        },
        {
            "name": "moonshotai/kimi-k2.5",
            "display_name": "Moonshot Kimi K2.5",
            "tier": "preview",
            "free_endpoint": True,
        },
        {
            "name": "deepseek-ai/deepseek-v3.2",
            "display_name": "DeepSeek V3.2",
            "tier": "preview",
            "free_endpoint": True,
        },
        {
            "name": "deepseek-ai/deepseek-v3.1-terminus",
            "display_name": "DeepSeek V3.1 Terminus",
            "tier": "preview",
            "free_endpoint": True,
        },
        {
            "name": "deepseek-ai/deepseek-r1",
            "display_name": "DeepSeek R1",
            "tier": "preview",
            "free_endpoint": True,
        },
        {
            "name": "stepfun-ai/step-3.5-flash",
            "display_name": "Stepfun Step 3.5 Flash",
            "tier": "preview",
            "free_endpoint": True,
        },
        {
            "name": "google/gemma-3-27b-it",
            "display_name": "Google Gemma 3 27B IT",
            "tier": "preview",
            "free_endpoint": True,
        },
        {
            "name": "google/gemma-4-31b-it",
            "display_name": "Google Gemma 4 31B IT",
            "tier": "preview",
            "free_endpoint": True,
        },
        {
            "name": "mistralai/mistral-large-2411",
            "display_name": "Mistral Large 3.675B Instruct 2512",
            "tier": "preview",
            "free_endpoint": True,
        },
        {
            "name": "mistralai/mistral-nemotron",
            "display_name": "Mistral Nemotron",
            "tier": "preview",
            "free_endpoint": True,
        },
    ]


def _extract_openai_compatible_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = message.get("content") or choice.get("text") or ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return clean_model_output("\n".join(parts))
    return clean_model_output(str(content))


def _summarize_openrouter_response(data: Any) -> str:
    """
    Keep the summary short so it can safely surface in Log Panel failover text.
    """
    try:
        if not isinstance(data, dict):
            return f"type={type(data).__name__}"

        keys = sorted(list(data.keys()))
        parts = [f"keys={keys}"]
        choices = data.get("choices")
        if isinstance(choices, list):
            parts.append(f"choices_len={len(choices)}")
            choice_summaries: List[str] = []
            for idx, choice in enumerate(choices[:2]):
                if not isinstance(choice, dict):
                    choice_summaries.append(f"{idx}:type={type(choice).__name__}")
                    continue
                c_keys = sorted(list(choice.keys()))
                msg = choice.get("message")
                msg_keys = sorted(list(msg.keys())) if isinstance(msg, dict) else []
                content = msg.get("content") if isinstance(msg, dict) else choice.get("text")
                if isinstance(content, list):
                    content_desc = f"list[{len(content)}]"
                elif isinstance(content, dict):
                    content_desc = f"dict[{sorted(list(content.keys()))}]"
                elif isinstance(content, str):
                    content_desc = f"str[{len(content)}]"
                elif content is None:
                    content_desc = "none"
                else:
                    content_desc = type(content).__name__
                choice_summaries.append(
                    f"{idx}:keys={c_keys};message_keys={msg_keys};content={content_desc}"
                )
            if choice_summaries:
                parts.append("choices=" + " | ".join(choice_summaries))
        else:
            parts.append(f"choices_type={type(choices).__name__}")

        if "usage" in data:
            usage = data.get("usage")
            if isinstance(usage, dict):
                parts.append(f"usage_keys={sorted(list(usage.keys()))}")
        return "; ".join(parts)
    except Exception as exc:
        return f"summary_error={type(exc).__name__}:{exc}"


def _chat_timeout(provider: str) -> int:
    provider = (provider or "").strip().lower()
    candidate = PROVIDER_READ_TIMEOUTS.get(provider, PROVIDER_READ_TIMEOUTS["default"])
    return max(10, min(int(REQUEST_TIMEOUT_SEC), int(candidate)))


def _route_timeout_budget(response_mode: str) -> int:
    mode = (response_mode or "fast").strip().lower()
    mapping = {
        "fast": FAST_MODE_ROUTE_TIMEOUT_SEC,
        "stable": STABLE_MODE_ROUTE_TIMEOUT_SEC,
        "deep": DEEP_MODE_ROUTE_TIMEOUT_SEC,
    }
    return max(15, min(int(REQUEST_TIMEOUT_SEC), int(mapping.get(mode, STABLE_MODE_ROUTE_TIMEOUT_SEC))))


def _continue_round_limit(response_mode: str) -> int:
    mode = (response_mode or "fast").strip().lower()
    mapping = {
        "fast": FAST_MODE_CONTINUE_ROUNDS,
        "stable": STABLE_MODE_CONTINUE_ROUNDS,
        "deep": DEEP_MODE_CONTINUE_ROUNDS,
    }
    return max(0, min(int(AUTO_CONTINUE_MAX_ROUNDS), int(mapping.get(mode, AUTO_CONTINUE_MAX_ROUNDS))))


def _per_attempt_timeout_budget(response_mode: str, provider: Optional[str] = None) -> int:
    del response_mode
    return _chat_timeout(provider or "")


def _effective_timeout(provider: str, timeout_override: Optional[float] = None) -> int:
    base_timeout = _chat_timeout(provider)
    if timeout_override is None:
        return base_timeout
    try:
        requested = int(timeout_override)
    except Exception:
        return base_timeout
    return max(3, min(int(REQUEST_TIMEOUT_SEC), requested))


def _resolve_nvidia_model(model: Optional[str] = None) -> str:
    requested = (model or os.getenv("NVIDIA_MODEL", NVIDIA_MODEL) or NVIDIA_MODEL).strip()
    return NVIDIA_MODEL_ALIASES.get(requested, requested)


def call_openrouter(prompt: str, max_tokens: int, temperature: float, timeout_override: Optional[float] = None) -> ModelReply:
    model = _ctx_model("openrouter", "OPENROUTER_MODEL", OPENROUTER_MODEL)
    url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "Finance Analysis Agent Web",
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_POLICY},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    timeout_sec = _effective_timeout("openrouter", timeout_override)
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    text = _extract_openai_compatible_text(data)
    if not text:
        summary = _summarize_openrouter_response(data)
        debug_log("openrouter", f"[EMPTY CONTENT] {summary}")
        raise RuntimeError(f"OpenRouter returned empty content; {summary}")
    return ModelReply(text=text, meta={"provider": "openrouter", "model": data.get("model", model), "usage": data.get("usage")})


def call_nvidia(prompt: str, max_tokens: int, temperature: float, timeout_override: Optional[float] = None) -> ModelReply:
    requested_model = _ctx_model("nvidia", "NVIDIA_MODEL", NVIDIA_MODEL)
    model = _resolve_nvidia_model(requested_model)
    url = os.getenv("NVIDIA_BASE_URL", NVIDIA_BASE_URL).strip() or NVIDIA_BASE_URL
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_POLICY},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stream": False,
    }
    timeout_sec = _effective_timeout("nvidia", timeout_override)
    if model == NVIDIA_NEMOTRON_SUPER_MODEL:
        timeout_sec = max(timeout_sec, min(30, int(REQUEST_TIMEOUT_SEC)))
    response = requests.post(url, headers=headers, json=payload, timeout=(10, timeout_sec))
    if response.status_code >= 400:
        content_type = response.headers.get("content-type", "")
        body = (response.text or "")[:1200].replace("\n", "\\n").replace("\r", "\\r")
        raise RuntimeError(
            "NVIDIA error "
            f"status={response.status_code}; "
            f"model={model}; "
            f"url={url}; "
            f"content_type={content_type}; "
            f"body={body}"
        )
    try:
        data = response.json()
    except Exception as exc:
        body = (response.text or "")[:1200].replace("\n", "\\n").replace("\r", "\\r")
        raise RuntimeError(
            "NVIDIA returned non-JSON response; "
            f"model={model}; "
            f"url={url}; "
            f"status={response.status_code}; "
            f"body={body}"
        ) from exc
    text = _extract_openai_compatible_text(data)
    if not text:
        raise RuntimeError(
            "NVIDIA returned empty content; "
            f"model={model}; "
            f"url={url}; "
            f"keys={list(data.keys())}"
        )
    usage = data.get("usage") or {}
    return ModelReply(
        text=text,
        meta={
            "provider": "nvidia",
            "model": data.get("model", model),
            "requested_model": requested_model,
            "usage": usage,
        },
    )


def call_cerebras(prompt: str, max_tokens: int, temperature: float, timeout_override: Optional[float] = None) -> ModelReply:
    timeout_sec = _effective_timeout("cerebras", timeout_override)
    requested_model = _ctx_model("cerebras", "CEREBRAS_MODEL", CEREBRAS_MODEL)
    model = CEREBRAS_MODEL_ALIASES.get(requested_model, requested_model)
    try:
        from cerebras.cloud.sdk import Cerebras  # type: ignore
    except Exception as exc:
        raise RuntimeError("cerebras-cloud-sdk is not installed") from exc

    client = Cerebras(api_key=CEREBRAS_API_KEY)

    def _do_call() -> Any:
        return client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_POLICY},
                {"role": "user", "content": prompt},
            ],
            model=model,
            max_completion_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=1,
            stream=False,
        )

    completion = _run_with_timeout(_do_call, timeout_sec)
    choices = getattr(completion, "choices", None) or []
    text = ""
    if choices:
        message = getattr(choices[0], "message", None)
        text = getattr(message, "content", "") or ""
    if not text:
        raise RuntimeError("Cerebras returned empty content")

    usage_obj = getattr(completion, "usage", None)
    usage: Dict[str, Any] = {}
    if usage_obj is not None:
        if hasattr(usage_obj, "model_dump"):
            try:
                usage = usage_obj.model_dump()
            except Exception:
                usage = {}
        elif isinstance(usage_obj, dict):
            usage = usage_obj
        else:
            for attr in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage_obj, attr, None)
                if value is not None:
                    usage[attr] = value
    return ModelReply(
        text=str(text).strip(),
        meta={"provider": "cerebras", "model": model, "requested_model": requested_model, "usage": usage},
    )


def call_mistral(prompt: str, max_tokens: int, temperature: float, timeout_override: Optional[float] = None) -> ModelReply:
    model = _ctx_model("mistral", "MISTRAL_MODEL", MISTRAL_MODEL)
    url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1/chat/completions").strip()
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_POLICY},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }
    timeout_sec = _effective_timeout("mistral", timeout_override)
    response = requests.post(url, headers=headers, json=payload, timeout=(10, timeout_sec))
    if response.status_code >= 400:
        detail = response.text[:1000]
        raise RuntimeError(f"Mistral error {response.status_code}: {detail}")
    data = response.json()
    choices = data.get("choices") or []
    text = ""
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    for key in ("text", "content", "value"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            parts.append(value.strip())
                            break
            text = "\n".join(parts).strip()
    if not text:
        raise RuntimeError(f"Mistral returned empty content; keys={list(data.keys())} choices_len={len(choices)}")
    usage = data.get("usage") or {}
    return ModelReply(text=text, meta={"provider": "mistral", "model": model, "usage": usage})


def call_groq(
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_policy: str = SYSTEM_POLICY,
    timeout_override: Optional[float] = None,
) -> ModelReply:
    model = _ctx_model("groq", "GROQ_MODEL", GROQ_MODEL)
    url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_policy},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": min(int(max_tokens), GROQ_MAX_OUTPUT_TOKENS),
        "temperature": float(temperature),
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=_effective_timeout("groq", timeout_override))
    if resp.status_code >= 400:
        raise RuntimeError(f"Groq error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    text = _extract_openai_compatible_text(data)
    if not text:
        raise RuntimeError("Groq returned empty content")
    return ModelReply(text=text, meta={"provider": "groq", "model": model, "usage": data.get("usage")})


def call_fireworks(prompt: str, max_tokens: int, temperature: float, timeout_override: Optional[float] = None) -> ModelReply:
    model = _ctx_model("fireworks", "FIREWORKS_MODEL", FIREWORKS_MODEL)
    base_url = os.getenv("FIREWORKS_BASE_URL", FIREWORKS_BASE_URL).rstrip("/")
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {FIREWORKS_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_POLICY},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=_effective_timeout("fireworks", timeout_override))
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Fireworks error {resp.status_code}: {resp.text[:300]} "
            "Hint: verify the model is available to your account/serverless deployment via /api/provider-models/fireworks."
        )
    data = resp.json()
    text = _extract_openai_compatible_text(data)
    if not text:
        raise RuntimeError("Fireworks returned empty content")
    return ModelReply(text=text, meta={"provider": "fireworks", "model": model, "usage": data.get("usage")})


def list_fireworks_serverless_models(page_size: int = 100) -> List[Dict[str, Any]]:
    if not FIREWORKS_API_KEY:
        return []
    url = f"{FIREWORKS_BASE_URL.replace('/inference/v1', '')}/v1/accounts/fireworks/models"
    headers = {"Authorization": f"Bearer {FIREWORKS_API_KEY}"}
    params = {"filter": "supports_serverless=true", "pageSize": int(page_size)}
    resp = requests.get(url, headers=headers, params=params, timeout=max(20, _chat_timeout("fireworks")))
    if resp.status_code >= 400:
        raise RuntimeError(f"Fireworks List Models error {resp.status_code}: {resp.text[:300]}")
    data = resp.json() or {}
    models = data.get("models") or data.get("results") or data.get("items") or []
    out: List[Dict[str, Any]] = []
    blocked_name_parts = (
        "embedding",
        "reranker",
        "flux",
        "kontext",
    )
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("id") or item.get("model")
        if not name:
            continue
        name_l = str(name).lower()
        if any(part in name_l for part in blocked_name_parts):
            continue
        out.append({
            "name": name,
            "display_name": item.get("displayName") or item.get("display_name") or name,
            "supports_serverless": item.get("supportsServerless") or item.get("supports_serverless"),
        })
    return out


def first_fireworks_serverless_model() -> str:
    models = list_fireworks_serverless_models(page_size=50)
    preferred = os.getenv("FIREWORKS_MODEL", FIREWORKS_MODEL).strip()
    if preferred:
        for item in models:
            name = str(item.get("name") or "").strip()
            if name == preferred:
                return name
    for item in models:
        name = str(item.get("name") or "").strip()
        if name:
            return name
    return ""


def call_cloudflare(
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_policy: str = SYSTEM_POLICY,
    timeout_override: Optional[float] = None,
) -> ModelReply:
    model = _ctx_model("cloudflare", "CF_MODEL", CF_MODEL)
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{model}"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": system_policy},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": min(int(max_tokens), CLOUDFLARE_MAX_OUTPUT_TOKENS),
        "temperature": float(temperature),
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=_effective_timeout("cloudflare", timeout_override))
    if resp.status_code >= 400:
        raise RuntimeError(f"Cloudflare error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    result = data.get("result") or {}
    text = result.get("response") or result.get("text") or result.get("output") or ""
    if not text and isinstance(result.get("choices"), list):
        text = _extract_openai_compatible_text(result)
    if not text:
        raise RuntimeError("Cloudflare returned empty content")
    usage = data.get("usage") or result.get("usage") or {}
    return ModelReply(text=str(text).strip(), meta={"provider": "cloudflare", "model": model, "usage": usage})


def call_gemini(prompt: str, max_tokens: int, temperature: float, timeout_override: Optional[float] = None) -> ModelReply:
    timeout_sec = _effective_timeout("gemini", timeout_override)
    model = _ctx_model("gemini", "GEMINI_MODEL", GEMINI_MODEL)
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except Exception as exc:
        raise RuntimeError("google-genai is not installed") from exc

    client = genai.Client(api_key=GEMINI_API_KEY)

    def _do_call() -> Any:
        return client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_POLICY,
                max_output_tokens=int(max_tokens),
                temperature=float(temperature),
            ),
        )

    response = _run_with_timeout(_do_call, timeout_sec)
    text = getattr(response, "text", "") or ""
    if not text:
        raise RuntimeError("Gemini returned empty content")
    usage_meta = getattr(response, "usage_metadata", None) or getattr(response, "usageMetadata", None)
    usage: Dict[str, Any] = {}
    if usage_meta is not None:
        for attr in ("prompt_token_count", "candidates_token_count", "total_token_count"):
            value = getattr(usage_meta, attr, None)
            if value is not None:
                usage[attr] = value
    return ModelReply(text=text.strip(), meta={"provider": "gemini", "model": model, "usage": usage})


def call_aws(prompt: str, max_tokens: int, temperature: float, timeout_override: Optional[float] = None) -> ModelReply:
    model = _ctx_model("aws", "AWS_BEDROCK_MODEL", AWS_BEDROCK_MODEL)
    try:
        import boto3  # type: ignore
        import botocore  # type: ignore
    except Exception as exc:
        raise RuntimeError("boto3/botocore are not installed") from exc

    session = boto3.Session(region_name=AWS_REGION)
    client = session.client(
        "bedrock-runtime",
        config=botocore.config.Config(connect_timeout=10, read_timeout=_effective_timeout("aws", timeout_override)),
    )
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": min(int(max_tokens), 4096),
        "temperature": float(temperature),
        "system": SYSTEM_POLICY,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    response = client.invoke_model(
        modelId=model,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body, ensure_ascii=False),
    )
    data = json.loads(response["body"].read())
    text = "".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text").strip()
    if not text:
        raise RuntimeError("AWS Bedrock returned empty content")
    return ModelReply(text=text, meta={"provider": "aws", "model": model, "usage": data.get("usage")})


def local_fallback_generate(prompt: str) -> str:
    del prompt
    return (
        "目前沒有可用的模型供應商或模型呼叫失敗。請在部署平台設定至少一組 API Key，"
        "例如 OPENROUTER_API_KEY 或 FIREWORKS_API_KEY，並確認 AGENT_PROVIDER_ORDER 包含該 provider。"
    )


def route_generate(
    prompt: Any,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float = 0.2,
    response_mode: str = "fast",
    disable_auto_continue: bool = False,
) -> ModelReply:
    providers = resolve_provider_order()
    if not providers:
        return ModelReply(local_fallback_generate(str(prompt or "")), {"provider": "local_fallback"})

    if isinstance(prompt, dict):
        prompt_bundle = {
            "heavy": str(prompt.get("heavy") or ""),
            "light": str(prompt.get("light") or prompt.get("heavy") or ""),
        }
    else:
        prompt_bundle = {"heavy": str(prompt or ""), "light": str(prompt or "")}

    errors: List[Tuple[str, str]] = []
    attempts: List[str] = []
    provider_trace: List[Dict[str, Any]] = []
    best_reply: Optional[ModelReply] = None
    route_started = time.perf_counter()
    route_timeout_sec = _route_timeout_budget(response_mode)
    max_continue_rounds = 0 if disable_auto_continue else _continue_round_limit(response_mode)
    for provider in providers:
        per_attempt_timeout_sec = _per_attempt_timeout_budget(response_mode, provider)
        route_elapsed_sec = time.perf_counter() - route_started
        if route_elapsed_sec >= route_timeout_sec:
            provider_trace.append({
                "provider": provider,
                "status": "skipped",
                "context_kind": "route",
                "prompt_tokens": 0,
                "prompt_chars": 0,
                "estimated_request_bytes": 0,
                "elapsed_ms": round(route_elapsed_sec * 1000, 1),
                "reason": f"route_timeout_budget_exhausted:{route_timeout_sec}s",
            })
            errors.append((provider, f"route_timeout_budget_exhausted:{route_timeout_sec}s"))
            break
        attempts.append(provider)
        trace: Dict[str, Any] = {"provider": provider}
        started = None
        try:
            remaining_budget_sec = max(3.0, route_timeout_sec - (time.perf_counter() - route_started))
            effective_attempt_timeout_sec = min(remaining_budget_sec, float(per_attempt_timeout_sec))
            if provider == "nvidia" and _resolve_nvidia_model() == NVIDIA_NEMOTRON_SUPER_MODEL:
                effective_attempt_timeout_sec = min(remaining_budget_sec, 30.0)
            context_kind = "light" if provider in {"groq", "cloudflare"} else "heavy"
            provider_prompt = prompt_bundle[context_kind]
            prompt_meta: Dict[str, Any] = {"mode": context_kind, "prompt_tokens": token_est(provider_prompt)}
            provider_profile = _provider_request_profile(provider, provider_prompt)
            if provider in {"groq", "cloudflare"}:
                provider_prompt, prompt_meta = _compact_prompt_for_provider(provider_prompt, provider)
                provider_profile = _provider_request_profile(provider, provider_prompt)
                debug_log(
                    provider,
                    f"prompt_compact mode={prompt_meta.get('mode')} prompt_tokens={prompt_meta.get('prompt_tokens')} "
                    f"budget={prompt_meta.get('light_budget', LIGHT_PROVIDER_MAX_PROMPT_TOKENS)}"
                )
                if provider_profile.get("skip"):
                    reason = str(provider_profile.get("reason", "prompt_too_large"))
                    errors.append((provider, reason))
                    trace.update({
                        "context_kind": context_kind,
                        "prompt_tokens": prompt_meta.get("prompt_tokens"),
                        "prompt_chars": len(provider_prompt),
                        "estimated_request_bytes": _estimate_request_bytes(
                            str(provider_profile.get("system_policy", SYSTEM_POLICY_LIGHT)),
                            provider_prompt,
                            int(provider_profile.get("max_tokens", 0) or 0),
                            temperature,
                        ),
                        "elapsed_ms": 0,
                        "status": "skipped",
                        "reason": reason,
                    })
                    provider_trace.append(trace)
                    debug_log(provider, f"skip {reason}")
                    continue

            reply: Optional[ModelReply] = None
            effective_max_tokens = max_tokens
            effective_system_policy = SYSTEM_POLICY
            if provider_profile:
                effective_max_tokens = min(max_tokens, int(provider_profile.get("max_tokens", max_tokens)))
                effective_system_policy = str(provider_profile.get("system_policy", SYSTEM_POLICY))
            request_bytes = _estimate_request_bytes(effective_system_policy, provider_prompt, effective_max_tokens, temperature)
            trace.update({
                "context_kind": context_kind,
                "prompt_tokens": token_est(provider_prompt),
                "prompt_chars": len(provider_prompt),
                "estimated_request_bytes": request_bytes,
                "remaining_budget_sec": round(remaining_budget_sec, 1),
                "attempt_timeout_sec": round(effective_attempt_timeout_sec, 1),
            })
            started = time.perf_counter()
            reply = _call_provider_once(
                provider,
                provider_prompt,
                max_tokens=effective_max_tokens,
                temperature=temperature,
                provider_profile=provider_profile,
                timeout_override=effective_attempt_timeout_sec,
            )
            trace["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
            trace["status"] = "ok"
            provider_trace.append(trace)
            if reply is not None:
                reply.meta = dict(reply.meta or {})
                continue_rounds = 0
                while continue_rounds < max_continue_rounds and _needs_auto_continue(reply, effective_max_tokens):
                    remaining_continue_budget_sec = route_timeout_sec - (time.perf_counter() - route_started)
                    if remaining_continue_budget_sec <= 5:
                        provider_trace.append({
                            "provider": provider,
                            "status": "skipped",
                            "context_kind": context_kind,
                            "prompt_tokens": token_est(reply.text or ""),
                            "prompt_chars": len(reply.text or ""),
                            "estimated_request_bytes": 0,
                            "elapsed_ms": round((time.perf_counter() - route_started) * 1000, 1),
                            "reason": f"continue_budget_exhausted:{round(max(0.0, remaining_continue_budget_sec), 1)}s_left",
                        })
                        break
                    continue_rounds += 1
                    continue_attempt_timeout_sec = min(float(per_attempt_timeout_sec), float(remaining_continue_budget_sec))
                    continue_prompt = _build_continue_prompt(provider_prompt, reply.text)
                    continue_trace: Dict[str, Any] = {
                        "provider": provider,
                        "status": "continuing",
                        "context_kind": context_kind,
                        "prompt_tokens": token_est(continue_prompt),
                        "prompt_chars": len(continue_prompt),
                        "estimated_request_bytes": _estimate_request_bytes(
                            effective_system_policy,
                            continue_prompt,
                            effective_max_tokens,
                            temperature,
                        ),
                        "remaining_budget_sec": round(remaining_continue_budget_sec, 1),
                        "attempt_timeout_sec": round(continue_attempt_timeout_sec, 1),
                        "reason": f"auto_continue_round_{continue_rounds}",
                    }
                    continue_started = time.perf_counter()
                    continuation = _call_provider_once(
                        provider,
                        continue_prompt,
                        max_tokens=effective_max_tokens,
                        temperature=temperature,
                        provider_profile=provider_profile,
                        timeout_override=continue_attempt_timeout_sec,
                    )
                    continue_trace["elapsed_ms"] = round((time.perf_counter() - continue_started) * 1000, 1)
                    continue_trace["status"] = "ok" if continuation and strip_redundant(continuation.text) else "error"
                    provider_trace.append(continue_trace)
                    if not continuation or not strip_redundant(continuation.text):
                        break
                    reply.text = strip_redundant((reply.text or "").rstrip() + "\n" + strip_redundant(continuation.text))
                    merged_meta = dict(continuation.meta or {})
                    base_usage = dict(reply.meta.get("usage") or {})
                    cont_usage = dict(merged_meta.get("usage") or {})
                    merged_usage: Dict[str, Any] = {}
                    usage_keys = set(base_usage.keys()) | set(cont_usage.keys())
                    for key in usage_keys:
                        left = base_usage.get(key, 0)
                        right = cont_usage.get(key, 0)
                        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                            merged_usage[key] = left + right
                        else:
                            merged_usage[key] = right or left
                    reply.meta["usage"] = merged_usage or cont_usage or base_usage
                reply.meta["continue_rounds"] = continue_rounds
                reply.meta["disable_auto_continue"] = disable_auto_continue
                reply.meta["provider_attempts"] = attempts[:]
                if provider_meta := prompt_meta:
                    reply.meta["prompt_meta"] = provider_meta
                if provider_profile:
                    reply.meta["provider_profile"] = provider_profile
                reply.meta["provider_trace"] = provider_trace[:]
                if errors:
                    reply.meta["failover_errors"] = [{"provider": p, "error": e} for p, e in errors]
                reply.meta["route_timeout_sec"] = route_timeout_sec
                best_reply = reply
                return reply
        except Exception as exc:
            errors.append((provider, str(exc)))
            trace["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1) if started is not None else 0
            trace["status"] = "error"
            trace["reason"] = str(exc)
            provider_trace.append(trace)
            debug_log(provider, exc)

    route_elapsed_ms = round((time.perf_counter() - route_started) * 1000, 1)
    if best_reply is not None:
        best_reply.meta = dict(best_reply.meta or {})
        best_reply.meta["provider_trace"] = provider_trace[:]
        best_reply.meta["failover_errors"] = [{"provider": p, "error": e} for p, e in errors]
        best_reply.meta["provider_attempts"] = attempts[:]
        best_reply.meta["route_timeout_sec"] = route_timeout_sec
        return best_reply

    timeout_text = (
        f"本輪對話在 {route_timeout_sec} 秒內未能取得可用回覆，系統已主動結束這次嘗試。"
        "請改用較快的回覆模式、縮短問題，或稍後再試。"
    )
    provider_error_text = (
        f"指定 provider 無法產生可用回覆。最後錯誤：{errors[-1][0]} -> {errors[-1][1]}"
        if errors else local_fallback_generate(prompt_bundle["heavy"])
    )
    return ModelReply(
        timeout_text if any("timeout" in str(err).lower() for _, err in errors) else provider_error_text,
        {
            "provider": "local_fallback",
            "errors": errors,
            "provider_attempts": attempts,
            "provider_trace": provider_trace,
            "route_timeout_sec": route_timeout_sec,
            "usage": {},
            "failover_errors": [{"provider": p, "error": e} for p, e in errors],
            "timeout_message": f"route_timeout_or_all_providers_failed:{route_timeout_sec}s",
            "route_elapsed_ms": route_elapsed_ms,
            "disable_auto_continue": disable_auto_continue,
        },
    )


def _build_context(
    adapter: MemoryAdapter,
    user_text: str,
    history_turns: int,
    use_history: bool,
    use_summary: bool,
    user_id: Optional[str] = None,
) -> str:
    blocks: List[Tuple[str, str]] = []

    global_facts = load_global_facts(user_id)
    if global_facts:
        lines = []
        for key in sorted(global_facts.keys()):
            value = global_facts[key]
            if isinstance(value, dict):
                lines.append(f"{key}={value.get('value', '')}")
            else:
                lines.append(f"{key}={value}")
        if lines:
            blocks.append(("global", _clip_text("[全域記憶]\n" + "\n".join(lines), MAX_GLOBAL_MEMORY_CHARS, "全域記憶")))

    if use_summary:
        summary = adapter.load_summary()
        if summary:
            blocks.append(("summary", "[對話摘要]\n" + _clip_text(summary, MAX_SUMMARY_CHARS, "對話摘要")))

    if use_history:
        turns = adapter.get_recent_turns(history_turns)
        if turns:
            lines = []
            for turn in turns:
                role = str(turn.get("role", "")).upper()
                content = _clip_text(str(turn.get("content", "")), MAX_HISTORY_TURN_CHARS, "單則歷史對話")
                if role and content:
                    lines.append(f"{role}: {content}")
            if lines:
                blocks.append(("history", "[最近對話]\n" + "\n".join(lines)))

    user_text = _clip_text(user_text, MAX_USER_MESSAGE_CHARS, "本輪使用者訊息")
    blocks.append(("user", "[本輪使用者問題]\n" + user_text))

    def _join(items: List[Tuple[str, str]]) -> str:
        return "\n\n".join(text for _, text in items if text.strip())

    prompt = _join(blocks)
    if token_est(prompt) <= MAX_PROMPT_TOKENS:
        return prompt

    compacted = [item for item in blocks if item[0] != "history"]
    prompt = _join(compacted)
    if token_est(prompt) <= MAX_PROMPT_TOKENS:
        return "[系統註記] 最近對話因 token 預算限制已省略，請優先回答本輪問題。\n\n" + prompt

    compacted = [item for item in compacted if item[0] != "summary"]
    prompt = _join(compacted)
    if token_est(prompt) <= MAX_PROMPT_TOKENS:
        return "[系統註記] 對話摘要與最近對話因 token 預算限制已省略，請優先回答本輪問題。\n\n" + prompt

    compacted = [item for item in compacted if item[0] != "global"]
    prompt = _join(compacted)
    if token_est(prompt) <= MAX_PROMPT_TOKENS:
        return "[系統註記] 全域記憶、摘要與最近對話因 token 預算限制已省略，請優先回答本輪問題。\n\n" + prompt

    max_user_chars = max(1000, int(MAX_PROMPT_TOKENS / max(TOKEN_EST_RATIO, 0.1)) - 800)
    return (
        "[系統註記] 使用者訊息過長，已保留前段內容並省略歷史脈絡；若需要完整分析，請分段貼上。\n\n"
        + "[本輪使用者問題]\n"
        + _clip_text(user_text, max_user_chars, "本輪使用者訊息")
    )


def _build_context_variant(
    adapter: MemoryAdapter,
    user_text: str,
    history_turns: int,
    use_history: bool,
    use_summary: bool,
    user_id: Optional[str] = None,
    external_context: Optional[str] = None,
    include_global: bool = True,
    include_summary: bool = True,
    include_history: bool = True,
    max_prompt_tokens: Optional[int] = None,
    max_summary_chars: Optional[int] = None,
    max_global_memory_chars: Optional[int] = None,
    max_history_turn_chars: Optional[int] = None,
    max_user_message_chars: Optional[int] = None,
    max_external_context_chars: Optional[int] = None,
) -> str:
    blocks: List[Tuple[str, str]] = []
    prompt_budget = int(max_prompt_tokens or MAX_PROMPT_TOKENS)
    summary_budget = int(max_summary_chars or MAX_SUMMARY_CHARS)
    global_budget = int(max_global_memory_chars or MAX_GLOBAL_MEMORY_CHARS)
    history_turn_budget = int(max_history_turn_chars or MAX_HISTORY_TURN_CHARS)
    user_budget = int(max_user_message_chars or MAX_USER_MESSAGE_CHARS)
    external_budget = int(max_external_context_chars or WEB_SEARCH_MAX_CONTEXT_CHARS)

    global_facts = load_global_facts(user_id) if include_global else {}
    if global_facts:
        lines = []
        for key in sorted(global_facts.keys()):
            value = global_facts[key]
            if isinstance(value, dict):
                lines.append(f"{key}={value.get('value', '')}")
            else:
                lines.append(f"{key}={value}")
        if lines:
            blocks.append(("global", _clip_text("[全域記憶]\n" + "\n".join(lines), global_budget, "全域記憶")))

    if use_summary and include_summary:
        summary = adapter.load_summary()
        if summary:
            blocks.append(("summary", "[對話摘要]\n" + _clip_text(summary, summary_budget, "對話摘要")))

    if use_history and include_history and history_turns > 0:
        turns = adapter.get_recent_turns(history_turns)
        if turns:
            lines = []
            for turn in turns:
                role = str(turn.get("role", "")).upper()
                content = _clip_text(str(turn.get("content", "")), history_turn_budget, "單則歷史對話")
                if role and content:
                    lines.append(f"{role}: {content}")
            if lines:
                blocks.append(("history", "[最近對話]\n" + "\n".join(lines)))

    if external_context:
        blocks.append(("web", _clip_text(external_context, external_budget, "即時網路搜尋結果")))

    user_text = _clip_text(user_text, user_budget, "本輪使用者訊息")
    blocks.append(("user", "[本輪使用者問題]\n" + user_text))

    def _join(items: List[Tuple[str, str]]) -> str:
        return "\n\n".join(text for _, text in items if text.strip())

    prompt = _join(blocks)
    if token_est(prompt) <= prompt_budget:
        return prompt

    compacted = [item for item in blocks if item[0] != "history"]
    prompt = _join(compacted)
    if token_est(prompt) <= prompt_budget:
        return "[系統註記] 最近對話因 token 預算限制已省略，請優先回答本輪問題。\n\n" + prompt

    compacted = [item for item in compacted if item[0] != "summary"]
    prompt = _join(compacted)
    if token_est(prompt) <= prompt_budget:
        return "[系統註記] 對話摘要與最近對話因 token 預算限制已省略，請優先回答本輪問題。\n\n" + prompt

    compacted = [item for item in compacted if item[0] != "global"]
    prompt = _join(compacted)
    if token_est(prompt) <= prompt_budget:
        return "[系統註記] 全域記憶、摘要與最近對話因 token 預算限制已省略，請優先回答本輪問題。\n\n" + prompt

    max_user_chars = max(1000, int(prompt_budget / max(TOKEN_EST_RATIO, 0.1)) - 800)
    return (
        "[系統註記] 使用者訊息過長，已保留前段內容並省略歷史脈絡；若需要完整分析，請分段貼上。\n\n"
        + "[本輪使用者問題]\n"
        + _clip_text(user_text, max_user_chars, "本輪使用者訊息")
    )


def _build_light_context(
    adapter: MemoryAdapter,
    user_text: str,
    use_summary: bool,
    user_id: Optional[str] = None,
    external_context: Optional[str] = None,
) -> str:
    return _build_context_variant(
        adapter,
        user_text,
        history_turns=LIGHT_HISTORY_TURNS,
        use_history=False,
        use_summary=use_summary,
        user_id=user_id,
        external_context=external_context,
        include_global=True,
        include_summary=True,
        include_history=False,
        max_prompt_tokens=LIGHT_PROVIDER_MAX_PROMPT_TOKENS,
        max_summary_chars=LIGHT_SUMMARY_CHARS,
        max_global_memory_chars=LIGHT_GLOBAL_MEMORY_CHARS,
        max_user_message_chars=LIGHT_USER_MESSAGE_CHARS,
        max_external_context_chars=LIGHT_WEB_SEARCH_MAX_CONTEXT_CHARS,
    )


def _update_summary(adapter: MemoryAdapter) -> None:
    rows = adapter.load_chat_raw()
    recent = rows[-8:]
    lines = [f"# {adapter.thread_id} 對話摘要", ""]
    for row in recent:
        role = row.get("role", "")
        content = strip_redundant(mask_pii(row.get("content", "")))[:220]
        lines.append(f"- {role}: {content}")
    lines.append("")
    lines.append(f"更新：{datetime.now().isoformat()}")
    adapter.save_summary("\n".join(lines))


def chat_once_detailed(
    memory: Any = None,
    THREAD_ID: str = "",
    user_text_or_prompt: str = "",
    print_reply: bool = True,
    min_tokens: Optional[int] = None,
    max_tokens_override: Optional[int] = None,
    temperature_override: Optional[float] = None,
    use_history: bool = True,
    history_turns: int = 8,
    use_summary: bool = True,
    summary_chars: int = 1800,
    user_id: Optional[str] = None,
    response_mode: str = "fast",
    disable_auto_continue: bool = False,
    web_search_provider_override: Optional[str] = None,
    force_web_search: bool = False,
    model_overrides: Optional[Dict[str, str]] = None,
    provider_order_override: Optional[str] = None,
) -> Dict[str, Any]:
    del min_tokens, summary_chars
    _CALL_CONTEXT.model_overrides = {k: v.strip() for k, v in (model_overrides or {}).items() if v and v.strip()}
    _CALL_CONTEXT.provider_order_override = (provider_order_override or "").strip() or None
    user_text = strip_redundant(user_text_or_prompt)
    if not user_text:
        return {"reply": "請輸入問題。", "meta": {"provider": "local"}}

    _MODE_HINTS: Dict[str, str] = {
        "fast": "（請簡潔回答，150字以內，重點條列）",
        "stable": "",
        "deep": "（請深入詳細分析，盡量完整，可分段說明）",
    }
    mode_hint = _MODE_HINTS.get(response_mode, "")
    prompt_user_text = f"{user_text}{mode_hint}" if mode_hint else user_text

    adapter = MemoryAdapter(memory=memory, THREAD_ID=THREAD_ID or "WEB_DEFAULT", USER_ID=user_id)
    search_decision = search_intent_classifier(user_text)
    if force_web_search and search_decision.get("priority") != "explicit":
        search_decision = {
            "must_search": True,
            "confidence": 1.0,
            "reason": "forced by UI override",
            "priority": "forced",
            "matched_keyword": "force_web_search",
        }
    web_search_meta: Dict[str, Any] = {
        "enabled": bool(WEB_SEARCH_ENABLED and (TAVILY_API_KEY or SERPER_API_KEY)),
        "used": False,
        "query": "",
        "results": [],
        "error": "",
        "decision": search_decision,
        "provider": "",
        "attempts": [],
        "fallback_errors": [],
        "provider_override": (web_search_provider_override or "").strip().lower() or "auto",
        "forced": bool(force_web_search),
    }
    external_context = ""
    if bool(search_decision.get("must_search")):
        web_search_meta["query"] = user_text
        try:
            raw_search = search_web(user_text, provider_override=web_search_provider_override)
            web_search_meta["used"] = bool(raw_search.get("results"))
            web_search_meta["results"] = raw_search.get("results") or []
            web_search_meta["response_time"] = raw_search.get("response_time")
            web_search_meta["provider"] = raw_search.get("provider") or ""
            web_search_meta["attempts"] = raw_search.get("attempts") or []
            web_search_meta["fallback_errors"] = raw_search.get("fallback_errors") or []
            external_context = _format_web_search_context(raw_search, WEB_SEARCH_MAX_CONTEXT_CHARS)
        except Exception as exc:
            web_search_meta["error"] = f"{type(exc).__name__}: {exc}"
            debug_log("web_search", web_search_meta["error"])
    heavy_prompt = _build_context_variant(
        adapter,
        prompt_user_text,
        history_turns=history_turns,
        use_history=use_history,
        use_summary=use_summary,
        user_id=user_id,
        external_context=external_context,
    )
    light_prompt = _build_light_context(
        adapter,
        prompt_user_text,
        use_summary=use_summary,
        user_id=user_id,
        external_context=external_context,
    )

    try:
        upsert_global_facts_from_text(user_text, user_id=user_id)
    except Exception as exc:
        debug_log("memory", f"failed to upsert global facts: {type(exc).__name__}: {exc}")
    try:
        adapter.add_turn("user", user_text)
    except Exception as exc:
        debug_log("memory", f"failed to persist user turn: {type(exc).__name__}: {exc}")
    t0 = time.perf_counter()
    try:
        reply = route_generate(
            {"heavy": mask_pii(heavy_prompt), "light": mask_pii(light_prompt)},
            max_tokens=int(max_tokens_override or MAX_OUTPUT_TOKENS),
            temperature=float(temperature_override if temperature_override is not None else 0.2),
            response_mode=response_mode,
            disable_auto_continue=disable_auto_continue,
        )
    finally:
        _CALL_CONTEXT.model_overrides = None
        _CALL_CONTEXT.provider_order_override = None
    latency_s = round(time.perf_counter() - t0, 3)
    assistant_text = strip_redundant(clean_model_output(reply.text))
    meta = dict(reply.meta or {})
    meta["latency_s"] = latency_s
    meta["response_mode"] = response_mode
    meta["disable_auto_continue"] = disable_auto_continue
    meta["web_search"] = web_search_meta
    meta["context_sizes"] = {
        "heavy_prompt_tokens": token_est(heavy_prompt),
        "heavy_prompt_chars": len(heavy_prompt),
        "light_prompt_tokens": token_est(light_prompt),
        "light_prompt_chars": len(light_prompt),
    }
    try:
        adapter.add_turn("assistant", assistant_text, meta=meta)
    except Exception as exc:
        debug_log("memory", f"failed to persist assistant turn: {type(exc).__name__}: {exc}")
    try:
        _update_summary(adapter)
    except Exception as exc:
        debug_log("memory", f"failed to update summary: {type(exc).__name__}: {exc}")

    if print_reply:
        print(assistant_text)
    return {"reply": assistant_text, "meta": meta}


def chat_once(
    memory: Any = None,
    THREAD_ID: str = "",
    user_text_or_prompt: str = "",
    print_reply: bool = True,
    min_tokens: Optional[int] = None,
    max_tokens_override: Optional[int] = None,
    temperature_override: Optional[float] = None,
    use_history: bool = True,
    history_turns: int = 8,
    use_summary: bool = True,
    summary_chars: int = 1800,
    user_id: Optional[str] = None,
    response_mode: str = "fast",
    disable_auto_continue: bool = False,
) -> str:
    result = chat_once_detailed(
        memory=memory,
        THREAD_ID=THREAD_ID,
        user_text_or_prompt=user_text_or_prompt,
        print_reply=print_reply,
        min_tokens=min_tokens,
        max_tokens_override=max_tokens_override,
        temperature_override=temperature_override,
        use_history=use_history,
        history_turns=history_turns,
        use_summary=use_summary,
        summary_chars=summary_chars,
        user_id=user_id,
        response_mode=response_mode,
        disable_auto_continue=disable_auto_continue,
    )
    assistant_text = str(result.get("reply", ""))
    return assistant_text


def provider_health_check(probe_prompt: str = "請用繁體中文簡短列出三點法金 AI 風險。") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for provider in resolve_provider_order():
        try:
            reply = route_generate(probe_prompt, max_tokens=512, temperature=0.2)
            rows.append({"provider": provider, "ok": True, "preview": reply.text[:160], "meta": reply.meta})
            break
        except Exception as exc:
            rows.append({"provider": provider, "ok": False, "error": str(exc)})
    return rows


if __name__ == "__main__":
    print("Finance/Analysis Agent deployment wrapper")
    print("MEMORY_ROOT =", MEMORY_ROOT)
    print("readiness =", json.dumps(provider_readiness(), ensure_ascii=False))
