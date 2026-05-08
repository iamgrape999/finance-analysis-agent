from __future__ import annotations

import os
import re
import hmac
import threading
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from Finance_Analysis_Agent_V581 import (
    AWS_BEDROCK_MODEL,
    CF_MODEL,
    CEREBRAS_MODEL,
    FIREWORKS_MODEL,
    GEMINI_API_KEY,
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
    fireworks_model_status,
    list_thread_summaries,
    list_fireworks_serverless_models,
    list_fireworks_catalog_models,
    list_nvidia_free_models,
    load_global_facts,
    provider_readiness,
    provider_diagnostics,
    resolve_provider_order,
    upsert_global_fact,
)


_ENV_LOCK = threading.Lock()

# ── Rate limiting (in-memory, single-process safe) ────────────────────────────
_RL_LOCK = threading.Lock()
_rl_chat: Dict[str, List[float]] = defaultdict(list)      # user_id → timestamps
_rl_image: Dict[str, List[float]] = defaultdict(list)     # user_id → timestamps
_rl_chat_ip: Dict[str, List[float]] = defaultdict(list)   # ip → timestamps (second layer)
_rl_image_ip: Dict[str, List[float]] = defaultdict(list)  # ip → timestamps

RATE_CHAT_MAX     = int(os.getenv("RATE_CHAT_MAX",     "30"))  # per user per window
RATE_IMAGE_MAX    = int(os.getenv("RATE_IMAGE_MAX",    "6"))   # per user per window (Gemini)
RATE_CHAT_IP_MAX  = int(os.getenv("RATE_CHAT_IP_MAX",  "60"))  # per IP per window (loose)
RATE_IMAGE_IP_MAX = int(os.getenv("RATE_IMAGE_IP_MAX", "12"))  # per IP per window (images)
RATE_WINDOW_SEC   = int(os.getenv("RATE_WINDOW_SEC",   "60"))

# ── Brute-force lockout ───────────────────────────────────────────────────────
_BF_LOCK = threading.Lock()
_bf_failures: Dict[str, List[float]] = defaultdict(list)  # ip → fail timestamps
BF_MAX_FAILURES  = int(os.getenv("BF_MAX_FAILURES",  "10"))
BF_WINDOW_SEC    = int(os.getenv("BF_WINDOW_SEC",    "300"))  # 5-minute window
BF_LOCKOUT_SEC   = int(os.getenv("BF_LOCKOUT_SEC",   "600"))  # 10-minute lockout

# ── Gemini concurrency cap ────────────────────────────────────────────────────
GEMINI_MAX_CONCURRENT = int(os.getenv("GEMINI_MAX_CONCURRENT", "2"))
_gemini_semaphore = threading.Semaphore(GEMINI_MAX_CONCURRENT)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    return (forwarded.split(",")[0].strip() or request.client.host or "unknown")[:64]


def _check_rate_limit(store: Dict[str, List[float]], key: str, max_req: int, window: int) -> bool:
    now = time.monotonic()
    with _RL_LOCK:
        times = store[key]
        times[:] = [t for t in times if now - t < window]
        if len(times) >= max_req:
            return False
        times.append(now)
        return True


def _record_auth_failure(ip: str) -> None:
    now = time.monotonic()
    with _BF_LOCK:
        times = _bf_failures[ip]
        times[:] = [t for t in times if now - t < BF_WINDOW_SEC]
        times.append(now)


def _is_locked_out(ip: str) -> bool:
    now = time.monotonic()
    with _BF_LOCK:
        times = _bf_failures[ip]
        times[:] = [t for t in times if now - t < BF_WINDOW_SEC]
        if len(times) >= BF_MAX_FAILURES:
            # still within lockout if last failure is recent enough
            return (now - times[-1]) < BF_LOCKOUT_SEC
        return False


APP_NAME = os.getenv("APP_NAME", "CathyChang AI")
STATIC_DIR = os.getenv("STATIC_DIR", "static")
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()
APP_USERS_RAW = os.getenv("APP_USERS", "").strip()
APP_BUILD_ID = os.getenv("APP_BUILD_ID", "2026-04-21-route-budget-v2")
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

origins = [item.strip() for item in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup_warning() -> None:
    import warnings
    if not APP_PASSWORD and not APP_USERS_RAW:
        warnings.warn(
            "SECURITY: Neither APP_PASSWORD nor APP_USERS is set. "
            "The API is accessible without authentication.",
            stacklevel=1,
        )


@app.middleware("http")
async def add_security_headers(request: Request, call_next) -> Response:
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


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
    image_base64: Optional[str] = Field(default=None, max_length=4_000_000)
    image_mime_type: Optional[str] = Field(default=None, pattern="^image/(jpeg|png|webp|gif)$")


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


def require_auth(
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
    request: Optional[Request] = None,
) -> str:
    ip = _client_ip(request) if request else "unknown"
    if _is_locked_out(ip):
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again later.")

    user_id = re.sub(r"[^A-Za-z0-9._-]+", "_", (x_user_id or "").strip())[:80]
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing X-User-Id")

    users = configured_users()
    password = x_app_password or ""
    if users:
        expected = users.get(user_id)
        if not expected or not hmac.compare_digest(password, expected):
            _record_auth_failure(ip)
            raise HTTPException(status_code=401, detail="Invalid user id or password")
        return user_id

    if APP_PASSWORD and not hmac.compare_digest(password, APP_PASSWORD):
        _record_auth_failure(ip)
        raise HTTPException(status_code=401, detail="Invalid app password")
    return user_id


@app.get("/api/health")
def health(
    request: Request,
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    user_id = require_auth(x_app_password, x_user_id, request)
    users = configured_users()
    return {
        "ok": True,
        "app": APP_NAME,
        "backend_version": APP_BUILD_ID,
        "auth_mode": "per_user" if users else "shared_password",
        "password_required": bool(APP_PASSWORD or users),
        "user_required": True,
        "user_id": user_id,
        "configured_user_count": len(users),
        "storage_scope": f"users/{user_id}",
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
def healthz(
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    require_auth(x_app_password, x_user_id)
    diagnostics = provider_diagnostics()
    return {
        "ok": True,
        "app": APP_NAME,
        "backend_version": APP_BUILD_ID,
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
    request: Request,
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
) -> ChatResponse:
    user_id = require_auth(x_app_password, x_user_id, request)
    ip = _client_ip(request)
    has_image = bool(req.image_base64)
    # dual-layer rate limiting: per-user and per-IP (defeats user_id rotation)
    if not _check_rate_limit(_rl_chat, user_id, RATE_CHAT_MAX, RATE_WINDOW_SEC) or \
       not _check_rate_limit(_rl_chat_ip, ip, RATE_CHAT_IP_MAX, RATE_WINDOW_SEC):
        raise HTTPException(status_code=429, detail=f"請求過於頻繁，請稍後再試（每 {RATE_WINDOW_SEC} 秒限 {RATE_CHAT_MAX} 次）。")
    if has_image and (not _check_rate_limit(_rl_image, user_id, RATE_IMAGE_MAX, RATE_WINDOW_SEC) or
                      not _check_rate_limit(_rl_image_ip, ip, RATE_IMAGE_IP_MAX, RATE_WINDOW_SEC)):
        raise HTTPException(status_code=429, detail=f"圖片請求過於頻繁，請稍後再試（每 {RATE_WINDOW_SEC} 秒限 {RATE_IMAGE_MAX} 次）。")
    return _chat_impl(req, user_id=user_id)


def _chat_impl(req: ChatRequest, user_id: str, enforce_min_tokens: bool = True) -> ChatResponse:
    has_image = bool(req.image_base64)
    if has_image:
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=400, detail="圖片分析需要 Gemini API 金鑰，請聯絡管理員設定。")
        effective_provider_order = "gemini"
    else:
        effective_provider_order = req.provider_order or MODE_DEFAULT_PROVIDER_ORDER.get(req.response_mode, MODE_DEFAULT_PROVIDER_ORDER["fast"])
    mode_default_tokens = MODE_DEFAULT_MAX_TOKENS.get(req.response_mode, MODE_DEFAULT_MAX_TOKENS["fast"])
    effective_max_tokens = int(req.max_tokens or mode_default_tokens)
    if enforce_min_tokens:
        effective_max_tokens = max(256, effective_max_tokens)
    gemini_lock_acquired = False
    if has_image:
        gemini_lock_acquired = _gemini_semaphore.acquire(blocking=True, timeout=10)
        if not gemini_lock_acquired:
            raise HTTPException(status_code=503, detail="伺服器正忙，Gemini 圖片分析請求已達上限，請稍後再試。")
    try:
        result = chat_once_detailed(
            memory=None,
            THREAD_ID=req.thread_id,
            user_text_or_prompt=req.message,
            print_reply=False,
            max_tokens_override=min(effective_max_tokens, 4096 if has_image else effective_max_tokens),
            temperature_override=req.temperature,
            use_history=req.use_history,
            history_turns=req.history_turns,
            user_id=user_id,
            response_mode=req.response_mode,
            disable_auto_continue=req.disable_auto_continue,
            web_search_provider_override=req.web_search_provider,
            force_web_search=req.force_web_search,
            model_overrides=req.model_overrides,
            provider_order_override=effective_provider_order,
            image_base64=req.image_base64 or None,
            image_mime_type=req.image_mime_type or None,
        )
    except Exception as exc:
        traceback.print_exc()
        error_context = {
            "phase": "chat_once_detailed",
            "user_id": user_id,
            "thread_id": req.thread_id,
            "provider_order": effective_provider_order or resolve_provider_order(),
            "response_mode": req.response_mode,
            "disable_auto_continue": req.disable_auto_continue,
            "model_overrides": list((req.model_overrides or {}).keys()),
            "web_search_provider": req.web_search_provider,
            "force_web_search": req.force_web_search,
        }
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}; context={error_context}",
        ) from exc
    finally:
        if gemini_lock_acquired:
            _gemini_semaphore.release()
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
) -> ChatResponse:
    user_id = require_auth(x_app_password, x_user_id)
    test_req = ChatRequest(
        thread_id=req.thread_id,
        message="SelfTest：請只回覆 OK，並用繁體中文。",
        max_tokens=req.max_tokens or 256,
        temperature=0.0,
        use_history=False,
        history_turns=0,
        response_mode=req.response_mode,
        disable_auto_continue=True,
        provider_order=req.provider_order,
        model_overrides=req.model_overrides,
        web_search_provider="auto",
        force_web_search=False,
    )
    return _chat_impl(test_req, user_id=user_id, enforce_min_tokens=False)


@app.get("/api/threads/{thread_id}/messages", response_model=List[MessageRecord])
def messages(
    thread_id: str,
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
) -> List[Dict[str, Any]]:
    user_id = require_auth(x_app_password, x_user_id)
    adapter = MemoryAdapter(memory=None, THREAD_ID=thread_id, USER_ID=user_id)
    return adapter.load_chat_raw()


@app.get("/api/threads", response_model=List[ThreadSummary])
def threads(
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
) -> List[Dict[str, Any]]:
    user_id = require_auth(x_app_password, x_user_id)
    return list_thread_summaries(user_id=user_id)


@app.delete("/api/threads/{thread_id}")
def delete_thread(
    thread_id: str,
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    user_id = require_auth(x_app_password, x_user_id)
    return {"ok": delete_thread_memory(thread_id, user_id=user_id), "thread_id": thread_id}


@app.get("/api/global-memory")
def global_memory(
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    user_id = require_auth(x_app_password, x_user_id)
    return {"ok": True, "facts": load_global_facts(user_id=user_id)}


@app.put("/api/global-memory")
def save_global_memory_fact(
    req: GlobalFactRequest,
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    user_id = require_auth(x_app_password, x_user_id)
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
) -> Dict[str, Any]:
    user_id = require_auth(x_app_password, x_user_id)
    return {"ok": True, "facts": delete_global_fact(key, user_id=user_id)}


@app.get("/api/provider-probe/{provider}")
def provider_probe(
    provider: str,
    model: Optional[str] = None,
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    require_auth(x_app_password, x_user_id)
    provider = provider.strip().lower()
    with _ENV_LOCK:
        previous_nvidia_model = os.environ.get("NVIDIA_MODEL")
        previous_fireworks_model = os.environ.get("FIREWORKS_MODEL")
        previous_mistral_model = os.environ.get("MISTRAL_MODEL")
        try:
            if provider == "nvidia":
                selected_model = (model or os.environ.get("NVIDIA_MODEL") or NVIDIA_MODEL).strip()
                if selected_model:
                    os.environ["NVIDIA_MODEL"] = selected_model
                reply = call_nvidia("請只回覆 OK。", max_tokens=64, temperature=0.0)
                return {"ok": True, "provider": "nvidia", "model": reply.meta.get("model"), "reply": reply.text, "usage": reply.meta.get("usage") or {}}

            if provider == "fireworks":
                selected_model = model
                if selected_model == "auto":
                    selected_model = first_fireworks_serverless_model()
                    if not selected_model:
                        return {"ok": False, "provider": "fireworks", "model": "auto", "error": "No Fireworks serverless models returned by /api/provider-models/fireworks."}
                if selected_model:
                    os.environ["FIREWORKS_MODEL"] = selected_model
                reply = call_fireworks("請只回覆 OK。", max_tokens=64, temperature=0.0)
                return {"ok": True, "provider": "fireworks", "model": reply.meta.get("model"), "reply": reply.text, "usage": reply.meta.get("usage") or {}}

            if provider == "mistral":
                selected_model = (model or os.environ.get("MISTRAL_MODEL") or MISTRAL_MODEL).strip()
                if selected_model:
                    os.environ["MISTRAL_MODEL"] = selected_model
                reply = call_mistral("請只回覆 OK。", max_tokens=64, temperature=0.0)
                return {"ok": True, "provider": "mistral", "model": reply.meta.get("model"), "reply": reply.text, "usage": reply.meta.get("usage") or {}}

            raise HTTPException(status_code=400, detail="Only nvidia, fireworks and mistral probes are implemented.")
        except HTTPException:
            raise
        except Exception as exc:
            active_model = (
                model
                or (os.environ.get("NVIDIA_MODEL") if provider == "nvidia" else None)
                or (os.environ.get("FIREWORKS_MODEL") if provider == "fireworks" else None)
                or (os.environ.get("MISTRAL_MODEL") if provider == "mistral" else None)
            )
            return {"ok": False, "provider": provider, "model": active_model, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            if previous_nvidia_model is None:
                os.environ.pop("NVIDIA_MODEL", None)
            else:
                os.environ["NVIDIA_MODEL"] = previous_nvidia_model
            if previous_fireworks_model is None:
                os.environ.pop("FIREWORKS_MODEL", None)
            else:
                os.environ["FIREWORKS_MODEL"] = previous_fireworks_model
            if previous_mistral_model is None:
                os.environ.pop("MISTRAL_MODEL", None)
            else:
                os.environ["MISTRAL_MODEL"] = previous_mistral_model


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
            catalog_models = list_fireworks_catalog_models()
            serverless_models = [item for item in catalog_models if bool(item.get("serverless_supported"))]
            current_model = fireworks_model_status(os.getenv("FIREWORKS_MODEL", FIREWORKS_MODEL))
            return {
                "ok": True,
                "provider": "fireworks",
                "models": serverless_models,
                "catalog_models": catalog_models,
                "current_model": current_model,
            }
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
