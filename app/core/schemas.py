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
        description="Conversation so far, oldest first. The last message must be from the user.",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Context for the turn. `input_variables` declares the {{variables}} the template "
            "may use; `agent_config` holds the ids the create tool needs."
        ),
    )

    @field_validator("conversation")
    @classmethod
    def _last_message_is_from_user(cls, messages: list[Message]) -> list[Message]:
        if messages[-1].role != "user":
            raise ValueError("the last message in `conversation` must have role 'user'")
        return messages


class AgentTemplate(BaseModel):
    """The voice-agent template. Every field but `summary_prompt` may use {{variables}}."""

    model_config = ConfigDict(extra="forbid")

    start_message: str = Field(description="The first thing the agent says when the call connects.")
    end_message: str = Field(description="What the agent says to close the call.")
    objective: str = Field(description="What the agent must achieve, as an outcome.")
    instructions: str = Field(description="How the agent behaves: tone, pacing, handling the call.")
    rules: list[str] = Field(description="Hard constraints. One imperative sentence per item.")
    summary_prompt: str = Field(
        description="Prompt used after the call to summarise it. Plain text, no variables."
    )


class AgentCreated(BaseModel):
    """Returned instead of a template on a turn where the agent was actually created."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str = Field(description="Name of the agent that was created.")
    message: str = Field(description="One sentence for the user confirming what was created.")


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ChatResponse(BaseModel):
    """`kind` tells the client which shape `output` is, without inspecting fields."""

    kind: Literal["template", "created"]
    output: AgentTemplate | AgentCreated
    model: str
    usage: Usage


class ErrorResponse(BaseModel):
    detail: str
    missing_payload_keys: list[str] | None = None
