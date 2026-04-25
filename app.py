from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import hmac
import time
import traceback
from collections import defaultdict, deque
from threading import Lock
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from Finance_Analysis_Agent_V581 import (
    AWS_BEDROCK_MODEL,
    CF_MODEL,
    CEREBRAS_MODEL,
    FIREWORKS_MODEL,
    GEMINI_MODEL,
    GROQ_MODEL,
    MISTRAL_MODEL,
    NVIDIA_MODEL,
    MemoryAdapter,
    OPENROUTER_MODEL,
    call_fireworks,
    call_mistral,
    call_nvidia,
    chat_once_detailed,
    delete_global_fact,
    delete_thread_memory,
    first_fireworks_serverless_model,
    list_thread_summaries,
    list_fireworks_serverless_models,
    list_nvidia_free_models,
    load_global_facts,
    provider_readiness,
    provider_diagnostics,
    resolve_provider_order,
    upsert_global_fact,
)


APP_NAME = os.getenv("APP_NAME", "CathyChang AI")
STATIC_DIR = os.getenv("STATIC_DIR", "static")
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()
APP_USERS_RAW = os.getenv("APP_USERS", "").strip()
APP_BUILD_ID = os.getenv("APP_BUILD_ID", "2026-04-21-route-budget-v2")
SESSION_SECRET = (
    os.getenv("SESSION_SECRET", "").strip()
    or APP_PASSWORD
    or APP_USERS_RAW
    or APP_NAME
)
SESSION_TTL_SEC = max(300, int(os.getenv("SESSION_TTL_SEC", "28800")))
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_MAX_REQUESTS = max(1, int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "12")))
RATE_LIMIT_WINDOW_SEC = max(10, int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60")))
RATE_LIMIT_SELFTEST_MAX_REQUESTS = max(1, int(os.getenv("RATE_LIMIT_SELFTEST_MAX_REQUESTS", "6")))
MODE_DEFAULT_MAX_TOKENS = {
    "fast": int(os.getenv("FAST_MODE_MAX_TOKENS", "768")),
    "stable": int(os.getenv("STABLE_MODE_MAX_TOKENS", "2048")),
    "deep": int(os.getenv("DEEP_MODE_MAX_TOKENS", "4096")),
}
MODE_DEFAULT_PROVIDER_ORDER = {
    "fast": os.getenv("FAST_MODE_PROVIDER_ORDER", "cerebras,groq,mistral,gemini,openrouter,cloudflare,fireworks,aws,nvidia"),
    "stable": os.getenv("STABLE_MODE_PROVIDER_ORDER", "cerebras,groq,mistral,gemini,openrouter,cloudflare,fireworks,aws,nvidia"),
    "deep": os.getenv("DEEP_MODE_PROVIDER_ORDER", "cerebras,groq,mistral,gemini,openrouter,cloudflare,fireworks,aws,nvidia"),
}

app = FastAPI(title=APP_NAME)
_RATE_LIMIT_LOCK = Lock()
_RATE_LIMIT_BUCKETS: Dict[str, deque[float]] = defaultdict(deque)

origins = [item.strip() for item in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    thread_id: str = Field(default="WEB_DEFAULT", max_length=80)
    message: str = Field(min_length=1, max_length=200_000)
    max_tokens: Optional[int] = Field(default=None, ge=128, le=12000)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    use_history: bool = True
    history_turns: int = Field(default=8, ge=0, le=20)
    response_mode: str = Field(default="fast", pattern="^(fast|stable|deep)$")
    disable_auto_continue: bool = False
    provider_order: Optional[str] = Field(default=None, max_length=120)
    model_overrides: Dict[str, str] = Field(default_factory=dict)
    web_search_provider: str = Field(default="auto", pattern="^(auto|tavily|serper)$")
    force_web_search: bool = False


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    provider: Optional[str] = None
    model: Optional[str] = None
    usage: Dict[str, Any] = Field(default_factory=dict)
    latency_s: Optional[float] = None
    provider_attempts: List[str] = Field(default_factory=list)
    failover_errors: List[Dict[str, str]] = Field(default_factory=list)
    provider_trace: List[Dict[str, Any]] = Field(default_factory=list)
    context_sizes: Dict[str, Any] = Field(default_factory=dict)
    web_search: Dict[str, Any] = Field(default_factory=dict)
    continue_rounds: int = 0
    route_timeout_sec: Optional[int] = None


class MessageRecord(BaseModel):
    ts: str = ""
    role: str
    content: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class ThreadSummary(BaseModel):
    thread_id: str
    updated_at: str = ""
    message_count: int = 0
    preview: str = ""


class GlobalFactRequest(BaseModel):
    key: str = Field(min_length=1, max_length=2000)
    value: str = Field(default="", max_length=2000)


class SessionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=80)
    password: str = Field(default="", max_length=2000)


def _rate_limit_key(user_id: str, scope: str) -> str:
    return f"{scope}:{user_id}"


def enforce_rate_limit(user_id: str, scope: str = "chat") -> None:
    if not RATE_LIMIT_ENABLED:
        return

    limit = RATE_LIMIT_MAX_REQUESTS
    if scope == "selftest":
        limit = RATE_LIMIT_SELFTEST_MAX_REQUESTS

    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SEC
    bucket_key = _rate_limit_key(user_id, scope)

    with _RATE_LIMIT_LOCK:
        bucket = _RATE_LIMIT_BUCKETS[bucket_key]
        while bucket and bucket[0] <= window_start:
            bucket.popleft()

        if len(bucket) >= limit:
            retry_after = max(1, int(RATE_LIMIT_WINDOW_SEC - (now - bucket[0])))
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded for user '{user_id}' on scope '{scope}'. "
                    f"Limit={limit} requests per {RATE_LIMIT_WINDOW_SEC}s. "
                    f"Retry after about {retry_after}s."
                ),
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)


def _sanitize_user_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())[:80]


def _parse_app_users() -> Dict[str, str]:
    users: Dict[str, str] = {}
    for item in re.split(r"[\n,;]+", APP_USERS_RAW):
        raw = item.strip()
        if not raw:
            continue
        if "=" in raw:
            user, password = raw.split("=", 1)
        elif ":" in raw:
            user, password = raw.split(":", 1)
        else:
            continue
        user_id = _sanitize_user_id(user)
        password = password.strip()
        if user_id and password:
            users[user_id] = password
    return users


def configured_users() -> Dict[str, str]:
    return _parse_app_users()


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def _session_secret_bytes() -> bytes:
    return SESSION_SECRET.encode("utf-8")


def _validate_login_credentials(user_id: str, password: str) -> str:
    normalized_user_id = _sanitize_user_id(user_id)
    if not normalized_user_id:
        raise HTTPException(status_code=400, detail="Missing user id")

    users = configured_users()
    if users:
        expected = users.get(normalized_user_id)
        if not expected or not hmac.compare_digest(password or "", expected):
            raise HTTPException(status_code=401, detail="Invalid user id or password")
        return normalized_user_id

    if APP_PASSWORD and not hmac.compare_digest(password or "", APP_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid app password")
    return normalized_user_id


def create_session_token(user_id: str) -> str:
    now = int(time.time())
    payload = {
        "uid": _sanitize_user_id(user_id),
        "iat": now,
        "exp": now + SESSION_TTL_SEC,
        "ver": 1,
    }
    payload_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64url_encode(payload_bytes)
    signature = hmac.new(_session_secret_bytes(), payload_bytes, hashlib.sha256).digest()
    return f"{payload_b64}.{_b64url_encode(signature)}"


def verify_session_token(token: str, expected_user_id: str) -> str:
    raw = (token or "").strip()
    if "." not in raw:
        raise HTTPException(status_code=401, detail="Invalid session token")

    payload_b64, sig_b64 = raw.split(".", 1)
    try:
        payload_bytes = _b64url_decode(payload_b64)
        given_sig = _b64url_decode(sig_b64)
        expected_sig = hmac.new(_session_secret_bytes(), payload_bytes, hashlib.sha256).digest()
        if not hmac.compare_digest(given_sig, expected_sig):
            raise HTTPException(status_code=401, detail="Invalid session token")
        payload = json.loads(payload_bytes.decode("utf-8"))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid session token: {type(exc).__name__}") from exc

    token_user_id = _sanitize_user_id(str(payload.get("uid") or ""))
    if token_user_id != _sanitize_user_id(expected_user_id):
        raise HTTPException(status_code=401, detail="Session token user mismatch")
    if int(payload.get("exp") or 0) < int(time.time()):
        raise HTTPException(status_code=401, detail="Session token expired")
    return token_user_id


def require_auth(
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
    x_session_token: Optional[str] = Header(default=None),
) -> str:
    user_id = re.sub(r"[^A-Za-z0-9._-]+", "_", (x_user_id or "").strip())[:80]
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing X-User-Id")

    if x_session_token:
        return verify_session_token(x_session_token, user_id)

    return _validate_login_credentials(user_id, x_app_password or "")


@app.post("/api/session")
def create_session(req: SessionRequest) -> Dict[str, Any]:
    user_id = _validate_login_credentials(req.user_id, req.password)
    return {
        "ok": True,
        "user_id": user_id,
        "session_token": create_session_token(user_id),
        "expires_in_sec": SESSION_TTL_SEC,
    }


@app.get("/api/health")
def health(
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
    x_session_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    user_id = require_auth(x_app_password, x_user_id, x_session_token)
    users = configured_users()
    return {
        "ok": True,
        "app": APP_NAME,
        "backend_version": APP_BUILD_ID,
        "auth_mode": "per_user" if users else "shared_password",
        "password_required": bool(APP_PASSWORD or users),
        "session_auth_enabled": True,
        "session_ttl_sec": SESSION_TTL_SEC,
        "user_required": True,
        "user_id": user_id,
        "configured_user_count": len(users),
        "storage_scope": f"users/{user_id}",
        "rate_limit": {
            "enabled": RATE_LIMIT_ENABLED,
            "chat_max_requests": RATE_LIMIT_MAX_REQUESTS,
            "selftest_max_requests": RATE_LIMIT_SELFTEST_MAX_REQUESTS,
            "window_sec": RATE_LIMIT_WINDOW_SEC,
        },
        "providers": provider_readiness(),
        "provider_diagnostics": provider_diagnostics(),
        "provider_order": resolve_provider_order(),
        "model_defaults": {
            "nvidia": os.getenv("NVIDIA_MODEL", NVIDIA_MODEL),
            "cerebras": os.getenv("CEREBRAS_MODEL", CEREBRAS_MODEL),
            "mistral": os.getenv("MISTRAL_MODEL", MISTRAL_MODEL),
            "openrouter": os.getenv("OPENROUTER_MODEL", OPENROUTER_MODEL),
            "fireworks": os.getenv("FIREWORKS_MODEL", FIREWORKS_MODEL),
            "gemini": os.getenv("GEMINI_MODEL", GEMINI_MODEL),
            "cloudflare": os.getenv("CF_MODEL", CF_MODEL),
            "groq": os.getenv("GROQ_MODEL", GROQ_MODEL),
            "aws": os.getenv("AWS_BEDROCK_MODEL", AWS_BEDROCK_MODEL),
        },
    }


@app.get("/api/healthz")
def healthz() -> Dict[str, Any]:
    diagnostics = provider_diagnostics()
    return {
        "ok": True,
        "app": APP_NAME,
        "backend_version": APP_BUILD_ID,
        "rate_limit_enabled": RATE_LIMIT_ENABLED,
        "rate_limit_max_requests": RATE_LIMIT_MAX_REQUESTS,
        "rate_limit_selftest_max_requests": RATE_LIMIT_SELFTEST_MAX_REQUESTS,
        "rate_limit_window_sec": RATE_LIMIT_WINDOW_SEC,
        "session_auth_enabled": True,
        "session_ttl_sec": SESSION_TTL_SEC,
        "web_search_enabled": bool(diagnostics.get("web_search_enabled")),
        "tavily_key_present": bool(diagnostics.get("tavily_key_present")),
        "serper_key_present": bool(diagnostics.get("serper_key_present")),
        "web_search_provider_order": diagnostics.get("web_search_provider_order") or [],
        "web_search_max_results": int(diagnostics.get("web_search_max_results") or 0),
        "nvidia_key_present": bool(diagnostics.get("nvidia_key_present")),
        "nvidia_model": str(diagnostics.get("nvidia_model") or ""),
        "nvidia_raw_model": str(diagnostics.get("nvidia_raw_model") or ""),
        "nvidia_effective_model": str(diagnostics.get("nvidia_effective_model") or ""),
        "nvidia_base_url": str(diagnostics.get("nvidia_base_url") or ""),
        "cerebras_key_present": bool(diagnostics.get("cerebras_key_present")),
        "cerebras_model": str(diagnostics.get("cerebras_model") or ""),
        "cerebras_raw_model": str(diagnostics.get("cerebras_raw_model") or ""),
        "cerebras_effective_model": str(diagnostics.get("cerebras_effective_model") or ""),
        "nvidia_probe_path": f"/api/provider-probe/nvidia?model={diagnostics.get('nvidia_model') or ''}",
        "nvidia_model_list_path": "/api/provider-models/nvidia",
        "mistral_import_ok": bool(diagnostics.get("mistral_import_ok")),
        "mistral_import_error": str(diagnostics.get("mistral_import_error") or ""),
        "mistral_client_mode": str(diagnostics.get("mistral_client_mode") or ""),
        "mistral_key_present": bool(diagnostics.get("mistral_key_present")),
        "mistral_model": str(diagnostics.get("mistral_model") or ""),
        "mistral_base_url": str(diagnostics.get("mistral_base_url") or ""),
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
    x_session_token: Optional[str] = Header(default=None),
) -> ChatResponse:
    user_id = require_auth(x_app_password, x_user_id, x_session_token)
    enforce_rate_limit(user_id, scope="chat")
    return _chat_impl(req, user_id=user_id)


def _chat_impl(req: ChatRequest, user_id: str, enforce_min_tokens: bool = True) -> ChatResponse:
    effective_provider_order = req.provider_order or MODE_DEFAULT_PROVIDER_ORDER.get(req.response_mode, MODE_DEFAULT_PROVIDER_ORDER["fast"])
    effective_model_overrides = {
        str(provider).strip().lower(): str(model).strip()
        for provider, model in (req.model_overrides or {}).items()
        if str(provider).strip() and str(model).strip()
    }
    try:
        mode_default_tokens = MODE_DEFAULT_MAX_TOKENS.get(req.response_mode, MODE_DEFAULT_MAX_TOKENS["fast"])
        effective_max_tokens = int(req.max_tokens or mode_default_tokens)
        if enforce_min_tokens:
            effective_max_tokens = max(256, effective_max_tokens)
        result = chat_once_detailed(
            memory=None,
            THREAD_ID=req.thread_id,
            user_text_or_prompt=req.message,
            print_reply=False,
            max_tokens_override=effective_max_tokens,
            temperature_override=req.temperature,
            use_history=req.use_history,
            history_turns=req.history_turns,
            user_id=user_id,
            response_mode=req.response_mode,
            disable_auto_continue=req.disable_auto_continue,
            web_search_provider_override=req.web_search_provider,
            force_web_search=req.force_web_search,
            provider_order_override=effective_provider_order,
            provider_models_override=effective_model_overrides,
        )
    except Exception as exc:
        traceback.print_exc()
        error_context = {
            "phase": "chat_once_detailed",
            "user_id": user_id,
            "thread_id": req.thread_id,
            "provider_order": effective_provider_order if 'effective_provider_order' in locals() else resolve_provider_order(),
            "response_mode": req.response_mode,
            "disable_auto_continue": req.disable_auto_continue,
            "model_overrides": list(effective_model_overrides.keys()),
            "web_search_provider": req.web_search_provider,
            "force_web_search": req.force_web_search,
        }
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}; context={error_context}",
        ) from exc
    meta = result.get("meta", {}) if isinstance(result, dict) else {}
    return ChatResponse(
        thread_id=req.thread_id,
        reply=str(result.get("reply", "")) if isinstance(result, dict) else "",
        provider=meta.get("provider"),
        model=meta.get("model"),
        usage=meta.get("usage") or {},
        latency_s=meta.get("latency_s"),
        provider_attempts=meta.get("provider_attempts") or [],
        failover_errors=meta.get("failover_errors") or [],
        provider_trace=meta.get("provider_trace") or [],
        context_sizes=meta.get("context_sizes") or {},
        web_search=meta.get("web_search") or {},
        continue_rounds=int(meta.get("continue_rounds") or 0),
        route_timeout_sec=int(meta.get("route_timeout_sec")) if meta.get("route_timeout_sec") is not None else None,
    )


@app.post("/api/selftest", response_model=ChatResponse)
def selftest(
    req: ChatRequest,
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
    x_session_token: Optional[str] = Header(default=None),
) -> ChatResponse:
    user_id = require_auth(x_app_password, x_user_id, x_session_token)
    enforce_rate_limit(user_id, scope="selftest")
    req.message = "SelfTest：請只回覆 OK，並用繁體中文。"
    req.max_tokens = req.max_tokens or 256
    req.temperature = 0.0
    return _chat_impl(req, user_id=user_id, enforce_min_tokens=False)


@app.get("/api/threads/{thread_id}/messages", response_model=List[MessageRecord])
def messages(
    thread_id: str,
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
    x_session_token: Optional[str] = Header(default=None),
) -> List[Dict[str, Any]]:
    user_id = require_auth(x_app_password, x_user_id, x_session_token)
    adapter = MemoryAdapter(memory=None, THREAD_ID=thread_id, USER_ID=user_id)
    return adapter.load_chat_raw()


@app.get("/api/threads", response_model=List[ThreadSummary])
def threads(
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
    x_session_token: Optional[str] = Header(default=None),
) -> List[Dict[str, Any]]:
    user_id = require_auth(x_app_password, x_user_id, x_session_token)
    return list_thread_summaries(user_id=user_id)


@app.delete("/api/threads/{thread_id}")
def delete_thread(
    thread_id: str,
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
    x_session_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    user_id = require_auth(x_app_password, x_user_id, x_session_token)
    return {"ok": delete_thread_memory(thread_id, user_id=user_id), "thread_id": thread_id}


@app.get("/api/global-memory")
def global_memory(
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
    x_session_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    user_id = require_auth(x_app_password, x_user_id, x_session_token)
    return {"ok": True, "facts": load_global_facts(user_id=user_id)}


@app.put("/api/global-memory")
def save_global_memory_fact(
    req: GlobalFactRequest,
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
    x_session_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    user_id = require_auth(x_app_password, x_user_id, x_session_token)
    key = req.key.strip()
    value = req.value.strip()
    if "=" in key and not value:
        key, value = key.split("=", 1)
    return {"ok": True, "facts": upsert_global_fact(key, value, user_id=user_id)}


@app.delete("/api/global-memory/{key}")
def remove_global_memory_fact(
    key: str,
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
    x_session_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    user_id = require_auth(x_app_password, x_user_id, x_session_token)
    return {"ok": True, "facts": delete_global_fact(key, user_id=user_id)}


@app.get("/api/provider-probe/{provider}")
def provider_probe(
    provider: str,
    model: Optional[str] = None,
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
    x_session_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    require_auth(x_app_password, x_user_id, x_session_token)
    provider = provider.strip().lower()
    active_model = (model or "").strip()
    try:
        if provider == "nvidia":
            active_model = active_model or NVIDIA_MODEL
            reply = call_nvidia("請只回覆 OK。", max_tokens=64, temperature=0.0, model_override=active_model)
            return {
                "ok": True,
                "provider": "nvidia",
                "model": reply.meta.get("model"),
                "requested_model": reply.meta.get("requested_model", active_model),
                "reply": reply.text,
                "usage": reply.meta.get("usage") or {},
            }

        if provider == "fireworks":
            selected_model = active_model
            if selected_model == "auto":
                selected_model = first_fireworks_serverless_model()
                if not selected_model:
                    return {"ok": False, "provider": "fireworks", "model": "auto", "error": "No Fireworks serverless models returned by /api/provider-models/fireworks."}
            active_model = selected_model or FIREWORKS_MODEL
            reply = call_fireworks("請只回覆 OK。", max_tokens=64, temperature=0.0, model_override=active_model)
            return {"ok": True, "provider": "fireworks", "model": reply.meta.get("model"), "reply": reply.text, "usage": reply.meta.get("usage") or {}}

        if provider == "mistral":
            active_model = active_model or MISTRAL_MODEL
            reply = call_mistral("請只回覆 OK。", max_tokens=64, temperature=0.0, model_override=active_model)
            return {"ok": True, "provider": "mistral", "model": reply.meta.get("model"), "reply": reply.text, "usage": reply.meta.get("usage") or {}}

        raise HTTPException(status_code=400, detail="Only nvidia, fireworks and mistral probes are implemented.")
    except HTTPException:
        raise
    except Exception as exc:
        return {"ok": False, "provider": provider, "model": active_model, "error": f"{type(exc).__name__}: {exc}"}

@app.get("/api/provider-models/{provider}")
def provider_models(
    provider: str,
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
    x_session_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    require_auth(x_app_password, x_user_id, x_session_token)
    provider = provider.strip().lower()
    try:
        if provider == "fireworks":
            models = list_fireworks_serverless_models()
            return {"ok": True, "provider": "fireworks", "models": models}
        if provider == "nvidia":
            models = list_nvidia_free_models()
            return {"ok": True, "provider": "nvidia", "models": models}
        raise HTTPException(status_code=400, detail="Only fireworks and nvidia model listing is implemented.")
    except HTTPException:
        raise
    except Exception as exc:
        return {"ok": False, "provider": provider, "error": str(exc), "models": []}


if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
