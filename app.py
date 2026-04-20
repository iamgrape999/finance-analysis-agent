from __future__ import annotations

import os
import re
import hmac
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from Finance_Analysis_Agent_V581 import (
    AWS_BEDROCK_MODEL,
    CF_MODEL,
    FIREWORKS_MODEL,
    GEMINI_MODEL,
    GROQ_MODEL,
    MemoryAdapter,
    OPENROUTER_MODEL,
    call_fireworks,
    chat_once_detailed,
    delete_global_fact,
    delete_thread_memory,
    first_fireworks_serverless_model,
    list_thread_summaries,
    list_fireworks_serverless_models,
    load_global_facts,
    provider_readiness,
    resolve_provider_order,
    upsert_global_fact,
)


APP_NAME = os.getenv("APP_NAME", "CathyChang AI")
STATIC_DIR = os.getenv("STATIC_DIR", "static")
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()
APP_USERS_RAW = os.getenv("APP_USERS", "").strip()

app = FastAPI(title=APP_NAME)

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
    provider_order: Optional[str] = Field(default=None, max_length=120)
    model_overrides: Dict[str, str] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    provider: Optional[str] = None
    model: Optional[str] = None
    usage: Dict[str, Any] = Field(default_factory=dict)
    latency_s: Optional[float] = None
    provider_attempts: List[str] = Field(default_factory=list)
    failover_errors: List[Dict[str, str]] = Field(default_factory=list)


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
) -> str:
    user_id = re.sub(r"[^A-Za-z0-9._-]+", "_", (x_user_id or "").strip())[:80]
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing X-User-Id")

    users = configured_users()
    password = x_app_password or ""
    if users:
        expected = users.get(user_id)
        if not expected or not hmac.compare_digest(password, expected):
            raise HTTPException(status_code=401, detail="Invalid user id or password")
        return user_id

    if APP_PASSWORD and not hmac.compare_digest(password, APP_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid app password")
    return user_id


@app.get("/api/health")
def health(
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    user_id = require_auth(x_app_password, x_user_id)
    users = configured_users()
    return {
        "ok": True,
        "app": APP_NAME,
        "auth_mode": "per_user" if users else "shared_password",
        "password_required": bool(APP_PASSWORD or users),
        "user_required": True,
        "user_id": user_id,
        "configured_user_count": len(users),
        "storage_scope": f"users/{user_id}",
        "providers": provider_readiness(),
        "provider_order": resolve_provider_order(),
        "model_defaults": {
            "openrouter": os.getenv("OPENROUTER_MODEL", OPENROUTER_MODEL),
            "fireworks": os.getenv("FIREWORKS_MODEL", FIREWORKS_MODEL),
            "gemini": os.getenv("GEMINI_MODEL", GEMINI_MODEL),
            "cloudflare": os.getenv("CF_MODEL", CF_MODEL),
            "groq": os.getenv("GROQ_MODEL", GROQ_MODEL),
            "aws": os.getenv("AWS_BEDROCK_MODEL", AWS_BEDROCK_MODEL),
        },
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
) -> ChatResponse:
    user_id = require_auth(x_app_password, x_user_id)
    return _chat_impl(req, user_id=user_id)


def _chat_impl(req: ChatRequest, user_id: str) -> ChatResponse:
    previous_provider_order = os.environ.get("AGENT_PROVIDER_ORDER")
    model_env_keys = {
        "openrouter": "OPENROUTER_MODEL",
        "fireworks": "FIREWORKS_MODEL",
        "gemini": "GEMINI_MODEL",
        "cloudflare": "CF_MODEL",
        "groq": "GROQ_MODEL",
        "aws": "AWS_BEDROCK_MODEL",
    }
    previous_models = {env_key: os.environ.get(env_key) for env_key in model_env_keys.values()}
    try:
        if req.provider_order:
            os.environ["AGENT_PROVIDER_ORDER"] = req.provider_order
        for provider, model in (req.model_overrides or {}).items():
            env_key = model_env_keys.get(provider)
            if env_key and model.strip():
                os.environ[env_key] = model.strip()
        result = chat_once_detailed(
            memory=None,
            THREAD_ID=req.thread_id,
            user_text_or_prompt=req.message,
            print_reply=False,
            max_tokens_override=req.max_tokens,
            temperature_override=req.temperature,
            use_history=req.use_history,
            history_turns=req.history_turns,
            user_id=user_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if req.provider_order:
            if previous_provider_order is None:
                os.environ.pop("AGENT_PROVIDER_ORDER", None)
            else:
                os.environ["AGENT_PROVIDER_ORDER"] = previous_provider_order
        for env_key, previous_value in previous_models.items():
            if previous_value is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = previous_value
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
    )


@app.post("/api/selftest", response_model=ChatResponse)
def selftest(
    req: ChatRequest,
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
) -> ChatResponse:
    user_id = require_auth(x_app_password, x_user_id)
    req.message = "SelfTest：請只回覆 OK，並用繁體中文。"
    req.max_tokens = req.max_tokens or 256
    req.temperature = 0.0
    return _chat_impl(req, user_id=user_id)


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
    previous_model = os.environ.get("FIREWORKS_MODEL")
    try:
        if provider != "fireworks":
            raise HTTPException(status_code=400, detail="Only fireworks probe is implemented.")
        selected_model = model
        if selected_model == "auto":
            selected_model = first_fireworks_serverless_model()
            if not selected_model:
                return {"ok": False, "provider": "fireworks", "model": "auto", "error": "No Fireworks serverless models returned by /api/provider-models/fireworks."}
        if selected_model:
            os.environ["FIREWORKS_MODEL"] = selected_model
        reply = call_fireworks("請只回覆 OK。", max_tokens=64, temperature=0.0)
        return {"ok": True, "provider": "fireworks", "model": reply.meta.get("model"), "reply": reply.text, "usage": reply.meta.get("usage") or {}}
    except HTTPException:
        raise
    except Exception as exc:
        return {"ok": False, "provider": provider, "model": model or os.environ.get("FIREWORKS_MODEL"), "error": str(exc)}
    finally:
        if previous_model is None:
            os.environ.pop("FIREWORKS_MODEL", None)
        else:
            os.environ["FIREWORKS_MODEL"] = previous_model


@app.get("/api/provider-models/{provider}")
def provider_models(
    provider: str,
    x_app_password: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    require_auth(x_app_password, x_user_id)
    provider = provider.strip().lower()
    try:
        if provider != "fireworks":
            raise HTTPException(status_code=400, detail="Only fireworks model listing is implemented.")
        models = list_fireworks_serverless_models()
        return {"ok": True, "provider": "fireworks", "models": models}
    except HTTPException:
        raise
    except Exception as exc:
        return {"ok": False, "provider": provider, "error": str(exc), "models": []}


if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
