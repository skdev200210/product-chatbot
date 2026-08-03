"""Request and response models for the chat endpoint."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Role = Literal["user", "assistant"]


class Message(BaseModel):
    """One turn of the conversation."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation: list[Message] = Field(
        min_length=1,
        description="Full conversation so far, oldest first. The last message must be from the user.",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary context injected into the prompt. `input_variables` is special: "
            "it declares the {{variable}} markers the generated template may use, as "
            "either a list of names or a mapping of name -> description."
        ),
    )

    @field_validator("conversation")
    @classmethod
    def _last_message_is_from_user(cls, messages: list[Message]) -> list[Message]:
        if messages[-1].role != "user":
            raise ValueError("the last message in `conversation` must have role 'user'")
        return messages


class AgentTemplate(BaseModel):
    """The call-agent template the model is forced to produce.

    This is the agent's output schema, so the response is always valid JSON.

    Every field except ``summary_prompt`` may contain ``{{variable}}`` markers,
    drawn only from the ``input_variables`` supplied in the request payload.
    ``summary_prompt`` runs after the call, when those variables no longer mean
    anything, so it must be plain text.
    """

    model_config = ConfigDict(extra="forbid")

    # No `min_length` / `min_items` here on purpose: those render as `minLength`
    # and `minItems`, which Anthropic's structured outputs reject. Emptiness is
    # enforced by the output validator in app/agents/chat_agent.py instead.
    start_message: str = Field(
        description=(
            "The first thing the agent says when the call connects. One or two "
            "spoken sentences."
        ),
    )
    end_message: str = Field(
        description="What the agent says to close the call, once the objective is met or refused.",
    )
    objective: str = Field(
        description="What the agent must achieve on the call, stated as an outcome.",
    )
    instructions: str = Field(
        description=(
            "How the agent should behave: tone, pacing, how to handle the "
            "expected turns of the conversation."
        ),
    )
    rules: list[str] = Field(
        description="Hard constraints — do's and don'ts. One rule per item, imperative.",
    )
    summary_prompt: str = Field(
        description=(
            "A prompt used after the call to summarise the transcript. "
            "Plain text only — no {{variable}} markers."
        ),
    )


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ChatResponse(BaseModel):
    output: AgentTemplate
    model: str
    usage: Usage


class ErrorResponse(BaseModel):
    detail: str
    missing_payload_keys: list[str] | None = None
