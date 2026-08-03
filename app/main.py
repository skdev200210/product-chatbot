"""FastAPI application exposing the chat endpoint."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from app.agents.chat_agent import InvalidInputVariables, chat_agent, run_chat
from app.core.config import get_settings
from app.core.prompt import MissingPromptKeys, placeholders
from app.core.schemas import ChatRequest, ChatResponse, ErrorResponse
from app.core.system_prompt import SYSTEM_PROMPT_TEMPLATE
from app.core.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "system prompt expects payload keys: %s",
        ", ".join(sorted(placeholders(SYSTEM_PROMPT_TEMPLATE))) or "none",
    )
    # Entering the agent opens the MCP sessions once and keeps them for the
    # lifetime of the process, instead of reconnecting on every request. If a
    # configured server is unreachable, startup fails here rather than the app
    # silently serving a model with no tools.
    async with chat_agent:
        yield


app = FastAPI(
    title="Product Chatbot",
    version="0.1.0",
    summary="Chat endpoint backed by a pydantic-ai agent with JSON-only output.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_allow_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": get_settings().model_name}


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    settings = get_settings()

    if len(request.conversation) > settings.max_conversation_messages:
        raise HTTPException(
            422,
            detail=(
                f"conversation has {len(request.conversation)} messages, "
                f"the limit is {settings.max_conversation_messages}"
            ),
        )

    try:
        template, usage = await asyncio.wait_for(
            run_chat(request.conversation, request.payload),
            timeout=settings.request_timeout_seconds,
        )
    except MissingPromptKeys as exc:
        raise HTTPException(
            422, detail={"detail": str(exc), "missing_payload_keys": exc.keys}
        ) from exc
    except InvalidInputVariables as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        logger.warning("chat timed out after %ss", settings.request_timeout_seconds)
        raise HTTPException(504, detail="the model did not respond in time") from exc
    except ModelHTTPError as exc:
        logger.error("model error %s: %s", exc.status_code, exc)
        raise HTTPException(502, detail=f"model request failed: {exc}") from exc
    except UnexpectedModelBehavior as exc:
        logger.error("model returned unusable output: %s", exc)
        raise HTTPException(
            502, detail="the model did not return a valid response for the required schema"
        ) from exc

    return ChatResponse(output=template, model=settings.model_name, usage=usage)
