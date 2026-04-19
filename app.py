from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from Finance_Analysis_Agent_V581 import (
    MemoryAdapter,
    call_fireworks,
    chat_once_detailed,
    first_fireworks_serverless_model,
    list_fireworks_serverless_models,
    provider_readiness,
    resolve_provider_order,
)


APP_NAME = os.getenv("APP_NAME", "Finance Analysis Agent")
STATIC_DIR = os.getenv("STATIC_DIR", "static")
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()

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


def require_password(x_app_password: Optional[str] = Header(default=None)) -> None:
    if not APP_PASSWORD:
        return
    if (x_app_password or "") != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid app password")


@app.get("/api/health")
def health(x_app_password: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    require_password(x_app_password)
    return {
        "ok": True,
        "app": APP_NAME,
        "password_required": bool(APP_PASSWORD),
        "providers": provider_readiness(),
        "provider_order": resolve_provider_order(),
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, x_app_password: Optional[str] = Header(default=None)) -> ChatResponse:
    require_password(x_app_password)
    return _chat_impl(req)


def _chat_impl(req: ChatRequest) -> ChatResponse:
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
def selftest(req: ChatRequest, x_app_password: Optional[str] = Header(default=None)) -> ChatResponse:
    require_password(x_app_password)
    req.message = "SelfTest：請只回覆 OK，並用繁體中文。"
    req.max_tokens = req.max_tokens or 256
    req.temperature = 0.0
    return chat(req)


@app.get("/api/threads/{thread_id}/messages", response_model=List[MessageRecord])
def messages(thread_id: str, x_app_password: Optional[str] = Header(default=None)) -> List[Dict[str, Any]]:
    require_password(x_app_password)
    adapter = MemoryAdapter(memory=None, THREAD_ID=thread_id)
    return adapter.load_chat_raw()


@app.get("/api/provider-probe/{provider}")
def provider_probe(provider: str, model: Optional[str] = None, x_app_password: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    require_password(x_app_password)
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
def provider_models(provider: str, x_app_password: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    require_password(x_app_password)
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
