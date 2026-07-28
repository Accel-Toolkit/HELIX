"""Provider-neutral conversation model.

The agent loop owns ONE internal transcript format; the provider
adapters translate to/from the Anthropic Messages API and the
OpenAI-compatible chat/completions wire formats.  Structural invariant
both APIs require: every assistant tool call receives exactly one tool
result, and all results of one assistant turn travel together in the
next message (``ToolResultsMsg`` enforces this by construction).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field


class StopReason(enum.Enum):
    END = "end"              # normal end of turn
    TOOL_USE = "tool_use"    # the model requested tool calls
    LENGTH = "length"        # ran out of max_tokens
    REFUSED = "refused"      # model-level refusal (relay verbatim)
    ERROR = "error"


@dataclass(frozen=True)
class ToolCall:
    id: str                  # provider call id (synthesized when absent)
    name: str
    args: dict


@dataclass(frozen=True)
class ToolOutcome:
    call_id: str
    name: str
    content: str             # JSON-serialized ToolResult envelope
    is_error: bool = False   # error/refused/denied


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


@dataclass(frozen=True)
class AssistantTurn:
    text: str
    tool_calls: tuple = ()
    stop_reason: StopReason = StopReason.END
    usage: Usage = field(default_factory=Usage)


# ---- transcript entries ---------------------------------------------------
@dataclass(frozen=True)
class UserMsg:
    text: str


@dataclass(frozen=True)
class SystemNote:
    """Runtime-injected note (job completions …), rendered to the model
    as a user-role message prefixed ``[system-note]``."""
    text: str


@dataclass(frozen=True)
class AssistantMsg:
    turn: AssistantTurn


@dataclass(frozen=True)
class ToolResultsMsg:
    outcomes: tuple            # tuple[ToolOutcome, ...]


# Transcript = list[UserMsg | SystemNote | AssistantMsg | ToolResultsMsg]
