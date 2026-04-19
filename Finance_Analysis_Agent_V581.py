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

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()

CF_API_TOKEN = os.getenv("CF_API_TOKEN", os.getenv("CF_API_KEY", "")).strip()
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "").strip()
CF_MODEL = os.getenv("CF_MODEL", "@cf/meta/llama-3.1-8b-instruct").strip()

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "").strip()
FIREWORKS_MODEL = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/deepseek-v3p1").strip()
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1").rstrip("/")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")).strip()
AWS_BEDROCK_MODEL = os.getenv("AWS_BEDROCK_MODEL", "anthropic.claude-3-haiku-20240307-v1:0").strip()

AGENT_PROVIDER_ORDER = os.getenv("AGENT_PROVIDER_ORDER", "openrouter,fireworks,gemini,cloudflare,groq,aws")
AGENT_PROVIDER_DISABLE = os.getenv("AGENT_PROVIDER_DISABLE", "").lower()
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))
REQUEST_TIMEOUT_SEC = int(os.getenv("REQUEST_TIMEOUT_SEC", "120"))

SYSTEM_POLICY = (
    "你是 Finance/Analysis Agent，請用繁體中文回答。"
    "回覆要專業、具體、可操作；資料不足時要明確說明，不要編造。"
    "禁止外露個資或敏感資訊；若發現個資，改以欄位名稱描述。"
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


class MemoryAdapter:
    def __init__(self, memory: Any = None, THREAD_ID: str = "THREAD_DEFAULT"):
        del memory
        self.thread_id = _sanitize_thread_id(THREAD_ID)
        self.thread_dir = os.path.join(MEMORY_ROOT, self.thread_id)
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
        return "\n".join(parts).strip()
    return str(content).strip()


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


def _build_context(adapter: MemoryAdapter, user_text: str, history_turns: int, use_history: bool, use_summary: bool) -> str:
    blocks: List[str] = []
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
) -> str:
    del min_tokens, summary_chars
    user_text = strip_redundant(user_text_or_prompt)
    if not user_text:
        return {"reply": "請輸入問題。", "meta": {"provider": "local"}}

    adapter = MemoryAdapter(memory=memory, THREAD_ID=THREAD_ID or "WEB_DEFAULT")
    prompt = _build_context(adapter, user_text, history_turns=history_turns, use_history=use_history, use_summary=use_summary)

    adapter.add_turn("user", user_text)
    t0 = time.perf_counter()
    reply = route_generate(
        mask_pii(prompt),
        max_tokens=int(max_tokens_override or MAX_OUTPUT_TOKENS),
        temperature=float(temperature_override if temperature_override is not None else 0.2),
    )
    latency_s = round(time.perf_counter() - t0, 3)
    assistant_text = strip_redundant(reply.text)
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
