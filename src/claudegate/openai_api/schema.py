"""The subset of the OpenAI wire format this server speaks.

Parsing is deliberately permissive (``extra="allow"``): clients send fields we
have no use for, and rejecting a request because of an unknown key is a worse
failure than ignoring it. Fields that would silently change the meaning of a
request are the exception — those are validated and rejected loudly.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Role = Literal["system", "developer", "user", "assistant", "tool", "function"]


class _Permissive(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class ImageURL(_Permissive):
    url: str
    detail: str | None = None


class FileData(_Permissive):
    file_data: str | None = None
    file_id: str | None = None
    filename: str | None = None


class ContentPart(_Permissive):
    type: str
    text: str | None = None
    image_url: ImageURL | None = None
    file: FileData | None = None


class FunctionCall(_Permissive):
    name: str = ""
    arguments: str = ""


class ToolCall(_Permissive):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(_Permissive):
    role: Role
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def text(self) -> str:
        """Flatten the content to plain text, dropping non-text parts."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(p.text for p in self.content if p.type == "text" and p.text)


class FunctionDef(_Permissive):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolDef(_Permissive):
    type: str = "function"
    function: FunctionDef


class StreamOptions(_Permissive):
    include_usage: bool = False


class ChatCompletionRequest(_Permissive):
    messages: list[Message]
    model: str | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    tools: list[ToolDef] | None = None
    tool_choice: str | dict[str, Any] | None = None
    n: int | None = None
    user: str | None = None

    # Accepted for compatibility. The Claude Code CLI does not expose sampling
    # parameters, so these are recorded and ignored rather than rejected —
    # documented in docs/COMPATIBILITY.md.
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop: str | list[str] | None = None
    seed: int | None = None

    # Mapped onto the agent's thinking budget.
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None

    @field_validator("n")
    @classmethod
    def _single_choice(cls, v: int | None) -> int | None:
        if v is not None and v != 1:
            raise ValueError("only n=1 is supported")
        return v

    @property
    def wants_usage_chunk(self) -> bool:
        return bool(self.stream_options and self.stream_options.include_usage)

    def named_tool_choice(self) -> str | None:
        """The function name when ``tool_choice`` pins one, else ``None``."""
        if isinstance(self.tool_choice, dict):
            fn = self.tool_choice.get("function")
            name = fn.get("name") if isinstance(fn, dict) else None
            if isinstance(name, str):
                return name
        return None

    @property
    def tools_disabled(self) -> bool:
        return self.tool_choice == "none"


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "anthropic"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


Usage = Annotated[dict[str, Any], Field(description="OpenAI usage block")]
