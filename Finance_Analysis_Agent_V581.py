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

import json
import os
import re
import shutil
import tempfile
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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()

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

AGENT_PROVIDER_ORDER = os.getenv("AGENT_PROVIDER_ORDER", "openrouter,cloudflare,groq,gemini,aws,fireworks")
AGENT_PROVIDER_DISABLE = os.getenv("AGENT_PROVIDER_DISABLE", "").lower()
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))
REQUEST_TIMEOUT_SEC = int(os.getenv("REQUEST_TIMEOUT_SEC", "120"))
TOKEN_EST_RATIO = float(os.getenv("TOKEN_EST_RATIO", "0.5"))
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "9000"))
MAX_USER_MESSAGE_CHARS = int(os.getenv("MAX_USER_MESSAGE_CHARS", "20000"))
MAX_HISTORY_TURN_CHARS = int(os.getenv("MAX_HISTORY_TURN_CHARS", "800"))
MAX_SUMMARY_CHARS = int(os.getenv("MAX_SUMMARY_CHARS", "1200"))
MAX_GLOBAL_MEMORY_CHARS = int(os.getenv("MAX_GLOBAL_MEMORY_CHARS", "1600"))

SYSTEM_POLICY = (
    "你是 CathyChang AI，請用繁體中文回答。"
    "你可以協助一般知識、生活、娛樂、商業、財務、投資與資料分析問題。"
    "若問題涉及財務、投資、法律、醫療或其他高風險決策，請提醒使用者需要自行查證或諮詢專業人士。"
    "回答要清楚、友善、具體；資料不足時請明確說明，不要編造。"
    "不要輸出內部推理過程、<think> 標籤或隱藏思考內容。"
)


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


def _acquire_lock(path: str, ttl_sec: int = 120) -> str:
    lock = _lock_path(path)
    payload = {"pid": os.getpid(), "ts": time.time(), "path": path}
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
        facts[key] = {"value": value.strip(), "ts": now}
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
        rows = MemoryAdapter(memory=None, THREAD_ID=name, USER_ID=user_id).load_chat_raw()
        if not rows:
            continue
        preview = ""
        for row in rows:
            if row.get("role") == "user" and row.get("content"):
                preview = strip_redundant(str(row.get("content", "")))[:120]
                break
        updated_at = rows[-1].get("ts", "")
        try:
            mtime = os.path.getmtime(chat_path)
        except Exception:
            mtime = 0
        out.append({
            "thread_id": name,
            "updated_at": updated_at,
            "message_count": len(rows),
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
        "openrouter": bool(OPENROUTER_API_KEY),
        "gemini": bool(GEMINI_API_KEY),
        "cloudflare": bool(CF_API_TOKEN and CF_ACCOUNT_ID),
        "groq": bool(GROQ_API_KEY),
        "fireworks": bool(FIREWORKS_API_KEY),
        "aws": bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY),
    }


def resolve_provider_order() -> List[str]:
    ready = provider_readiness()
    provider_order = os.getenv("AGENT_PROVIDER_ORDER", AGENT_PROVIDER_ORDER)
    provider_disable = os.getenv("AGENT_PROVIDER_DISABLE", AGENT_PROVIDER_DISABLE).lower()
    disabled = {p.strip() for p in provider_disable.split(",") if p.strip()}
    out: List[str] = []
    for raw in provider_order.split(","):
        provider = raw.strip().lower()
        if provider and ready.get(provider) and provider not in disabled and provider not in out:
            out.append(provider)
    return out


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


def call_openrouter(prompt: str, max_tokens: int, temperature: float) -> ModelReply:
    model = os.getenv("OPENROUTER_MODEL", OPENROUTER_MODEL).strip() or OPENROUTER_MODEL
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
    resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SEC)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    text = _extract_openai_compatible_text(data)
    if not text:
        raise RuntimeError("OpenRouter returned empty content")
    return ModelReply(text=text, meta={"provider": "openrouter", "model": data.get("model", model), "usage": data.get("usage")})


def call_groq(prompt: str, max_tokens: int, temperature: float) -> ModelReply:
    model = os.getenv("GROQ_MODEL", GROQ_MODEL).strip() or GROQ_MODEL
    url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_POLICY},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": min(int(max_tokens), 4096),
        "temperature": float(temperature),
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SEC)
    if resp.status_code >= 400:
        raise RuntimeError(f"Groq error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    text = _extract_openai_compatible_text(data)
    if not text:
        raise RuntimeError("Groq returned empty content")
    return ModelReply(text=text, meta={"provider": "groq", "model": model, "usage": data.get("usage")})


def call_fireworks(prompt: str, max_tokens: int, temperature: float) -> ModelReply:
    model = os.getenv("FIREWORKS_MODEL", FIREWORKS_MODEL).strip() or FIREWORKS_MODEL
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
    resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SEC)
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
    resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SEC)
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


def call_cloudflare(prompt: str, max_tokens: int, temperature: float) -> ModelReply:
    model = os.getenv("CF_MODEL", CF_MODEL).strip() or CF_MODEL
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{model}"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_POLICY},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": min(int(max_tokens), 4096),
        "temperature": float(temperature),
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SEC)
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


def call_gemini(prompt: str, max_tokens: int, temperature: float) -> ModelReply:
    model = os.getenv("GEMINI_MODEL", GEMINI_MODEL).strip() or GEMINI_MODEL
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except Exception as exc:
        raise RuntimeError("google-genai is not installed") from exc

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_POLICY,
            max_output_tokens=int(max_tokens),
            temperature=float(temperature),
        ),
    )
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


def call_aws(prompt: str, max_tokens: int, temperature: float) -> ModelReply:
    model = os.getenv("AWS_BEDROCK_MODEL", AWS_BEDROCK_MODEL).strip() or AWS_BEDROCK_MODEL
    try:
        import boto3  # type: ignore
        import botocore  # type: ignore
    except Exception as exc:
        raise RuntimeError("boto3/botocore are not installed") from exc

    session = boto3.Session(region_name=AWS_REGION)
    client = session.client(
        "bedrock-runtime",
        config=botocore.config.Config(connect_timeout=10, read_timeout=REQUEST_TIMEOUT_SEC),
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


def route_generate(prompt: str, max_tokens: int = MAX_OUTPUT_TOKENS, temperature: float = 0.2) -> ModelReply:
    providers = resolve_provider_order()
    if not providers:
        return ModelReply(local_fallback_generate(prompt), {"provider": "local_fallback"})

    errors: List[Tuple[str, str]] = []
    attempts: List[str] = []
    for provider in providers:
        attempts.append(provider)
        try:
            reply: Optional[ModelReply] = None
            if provider == "openrouter":
                reply = call_openrouter(prompt, max_tokens=max_tokens, temperature=temperature)
            if provider == "gemini":
                reply = call_gemini(prompt, max_tokens=max_tokens, temperature=temperature)
            if provider == "cloudflare":
                reply = call_cloudflare(prompt, max_tokens=max_tokens, temperature=temperature)
            if provider == "groq":
                reply = call_groq(prompt, max_tokens=max_tokens, temperature=temperature)
            if provider == "fireworks":
                reply = call_fireworks(prompt, max_tokens=max_tokens, temperature=temperature)
            if provider == "aws":
                reply = call_aws(prompt, max_tokens=max_tokens, temperature=temperature)
            if reply is not None:
                reply.meta = dict(reply.meta or {})
                reply.meta["provider_attempts"] = attempts[:]
                if errors:
                    reply.meta["failover_errors"] = [{"provider": p, "error": e} for p, e in errors]
                return reply
        except Exception as exc:
            errors.append((provider, str(exc)))
            debug_log(provider, exc)

    return ModelReply(
        local_fallback_generate(prompt),
        {"provider": "local_fallback", "errors": errors},
    )


def _build_context_legacy(
    adapter: MemoryAdapter,
    user_text: str,
    history_turns: int,
    use_history: bool,
    use_summary: bool,
    user_id: Optional[str] = None,
) -> str:
    blocks: List[str] = []
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
            blocks.append("[全域記憶]\n" + "\n".join(lines))
    if use_summary:
        summary = adapter.load_summary()
        if summary:
            blocks.append("[對話摘要]\n" + summary[:2000])
    if use_history:
        turns = adapter.get_recent_turns(history_turns)
        if turns:
            lines = []
            for turn in turns:
                role = str(turn.get("role", "")).upper()
                content = str(turn.get("content", ""))[:1200]
                if role and content:
                    lines.append(f"{role}: {content}")
            if lines:
                blocks.append("[最近對話]\n" + "\n".join(lines))
    blocks.append("[本輪使用者問題]\n" + user_text)
    return "\n\n".join(blocks)


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
) -> str:
    del min_tokens, summary_chars
    user_text = strip_redundant(user_text_or_prompt)
    if not user_text:
        return {"reply": "請輸入問題。", "meta": {"provider": "local"}}

    adapter = MemoryAdapter(memory=memory, THREAD_ID=THREAD_ID or "WEB_DEFAULT", USER_ID=user_id)
    prompt = _build_context(
        adapter,
        user_text,
        history_turns=history_turns,
        use_history=use_history,
        use_summary=use_summary,
        user_id=user_id,
    )

    upsert_global_facts_from_text(user_text, user_id=user_id)
    adapter.add_turn("user", user_text)
    t0 = time.perf_counter()
    reply = route_generate(
        mask_pii(prompt),
        max_tokens=int(max_tokens_override or MAX_OUTPUT_TOKENS),
        temperature=float(temperature_override if temperature_override is not None else 0.2),
    )
    latency_s = round(time.perf_counter() - t0, 3)
    assistant_text = strip_redundant(clean_model_output(reply.text))
    meta = dict(reply.meta or {})
    meta["latency_s"] = latency_s
    adapter.add_turn("assistant", assistant_text, meta=meta)
    _update_summary(adapter)

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
