"""Talk-to-data endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.talk_to_data.memory import MEMORY
from src.talk_to_data.nl_to_sql import answer
from src.talk_to_data.prompt_templates import CURRENT_VERSION, available_versions
from src.talk_to_data.schema_context import approximate_tokens, build_schema_context
from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["talk-to-data"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    # Conversation state lives server-side, keyed by this. The client holds an
    # identifier, never the history, so replacing the UI does not mean
    # reimplementing memory.
    session_id: str = Field(default="default", min_length=1, max_length=128)
    prompt_version: str = Field(default=CURRENT_VERSION)
    use_memory: bool = True


@router.post("/chat")
def chat(request: ChatRequest) -> dict:
    settings = get_settings()
    if not settings.llm_configured:
        raise HTTPException(
            status_code=503,
            detail=(
                "ANTHROPIC_API_KEY is not set on the API service. The chatbot "
                "requires it; every other module works without it."
            ),
        )
    if request.prompt_version not in available_versions():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown prompt version. Available: {available_versions()}",
        )

    result = answer(
        request.question,
        session_id=request.session_id,
        version=request.prompt_version,
        use_memory=request.use_memory,
    )
    result["session_id"] = request.session_id
    result["turns_remembered"] = MEMORY.turn_count(request.session_id)
    return result


@router.delete("/chat/{session_id}")
def clear_session(session_id: str) -> dict:
    return {"session_id": session_id, "cleared": MEMORY.clear(session_id)}


@router.get("/chat/schema")
def schema() -> dict:
    """What the chatbot can see.

    Exposed so the UI can show users which fields are available, which heads
    off a class of question the bot would otherwise have to refuse.
    """
    context = build_schema_context()
    return {
        "schema_context": context,
        "approximate_tokens": approximate_tokens(context),
        "prompt_versions": available_versions(),
        "current_version": CURRENT_VERSION,
        "active_sessions": MEMORY.active_sessions(),
    }
