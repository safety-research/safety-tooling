from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union

import openai
import pydantic

from safetytooling.data_models.utils import GeminiBlockReason, GeminiStopReason, SafetyRatings

from .hashable import HashableBaseModel


class LLMParams(HashableBaseModel):
    model: str | tuple[str, ...]
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    num_candidates_per_completion: int = 1
    insufficient_valids_behaviour: Literal["error", "continue", "pad_invalids", "retry"] = "retry"
    max_tokens: int | None = None  # For OpenAI models this is converted to max_completion_tokens in the API layer
    max_completion_tokens: int | None = None  # Used by OpenAI models
    thinking: dict[str, Any] | None = None  # Used by Anthropic reasoning models
    logprobs: int | None = None
    seed: int | None = None
    return_logprobs: bool = False  # ?
    top_k_logprobs: int | None = None  # ?
    logit_bias: dict[int, float] | None = None
    stop: list[str] | None = None
    response_format: Any = pydantic.Field(
        default=None,
        # TODO: Stopgap.
        # Currently we send a pydantic.BaseModel class type in this param, which is not serializable.
        # Caching will be agnostic to the value of this field
        # Remove exlude=True once you figure out how to serialise pydantic.BaseModel class types
        exclude=True,
    )
    model_config = pydantic.ConfigDict(extra="forbid")


class StopReason(Enum):
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    CONTENT_FILTER = "content_filter"
    API_ERROR = "api_error"
    PROMPT_BLOCKED = "prompt_blocked"
    UNKNOWN = "unknown"

    def __str__(self):
        return self.value


class LLMResponse(pydantic.BaseModel):
    model_id: str
    completion: str
    stop_reason: StopReason | GeminiStopReason | GeminiBlockReason
    cost: float = 0
    audio_out: str | Path | None = None
    duration: float | None = None
    api_duration: float | None = None
    logprobs: list[dict[str, float]] | None = None
    safety_ratings: Union[dict, SafetyRatings] | None = None
    recitation_retries: int | None = None
    api_failures: int | None = 0
    batch_custom_id: str | None = None
    reasoning_content: str | None = None

    model_config = pydantic.ConfigDict(protected_namespaces=())

    @pydantic.field_validator("stop_reason", mode="before")
    def parse_stop_reason(cls, v: str):
        if v in ["length", "max_tokens"]:
            return StopReason.MAX_TOKENS
        elif v in ["stop", "stop_sequence", "end_turn", "eos"]:
            # end_turn is for Anthropic
            return StopReason.STOP_SEQUENCE
        elif v in ["content_filter"]:
            return StopReason.CONTENT_FILTER
        elif v in ["prompt_blocked"]:
            return StopReason.PROMPT_BLOCKED
        elif v in ["api_error"]:
            return StopReason.API_ERROR
        elif v in ["recitation"]:
            return GeminiStopReason.RECITATION
        elif v in ["safety"]:
            return GeminiStopReason.SAFETY
        elif "unspecified" in v:
            return GeminiStopReason.FINISH_REASON_UNSPECIFIED
        elif v in "other":
            return GeminiStopReason.OTHER
        raise ValueError(f"Invalid stop reason: {v}")

    def to_dict(self):
        return {**self.model_dump(), "stop_reason": str(self.stop_reason)}

    def __setitem__(self, key, value):
        setattr(self, key, value)


class TaggedModeration(openai._models.BaseModel):
    # The model used to generate the moderation results.
    model_id: str

    # Actual moderation results
    moderation: openai.types.Moderation

    model_config = pydantic.ConfigDict(protected_namespaces=())
