"""Request and response types for the System One endpoint.

Modelled on TypeSafe's Jev API (``POST /v1/systemone``): one *state* plus any
number of typed *questions*, all answered in a single call. The three primitives
are ``noul`` (boolean), ``choice`` (one of a set) and ``score`` (an ordered
scale).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

# A canvas slot holds one token, so every option has to be a distinct
# single-token label. Latin capitals give 26, far past what a calibrated read
# can separate anyway.
OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Question(BaseModel):
    type: Literal["noul", "choice", "score"]
    instructions: Union[str, Dict[str, Any], List[Any]]
    criteria: Optional[Union[Dict[str, Optional[str]], List[str]]] = None

    @field_validator("criteria")
    @classmethod
    def _check_criteria(cls, value, info):
        kind = info.data.get("type")
        if kind == "score":
            if not isinstance(value, list) or len(value) < 2:
                raise ValueError("score criteria must be a list of >= 2 levels")
        elif kind == "choice":
            if not isinstance(value, dict) or len(value) < 2:
                raise ValueError("choice criteria must be a map of >= 2 options")
        return value


class SystemOneRequest(BaseModel):
    state: Union[str, Dict[str, Any], List[Any]]
    questions: Dict[str, Question]
    model: Optional[str] = None
    # Extension over Jev: reads average independent canvases, and their spread
    # is reported as stderr. One read is fast; more is steadier on close calls.
    reads: int = Field(default=4, ge=1, le=64)

    @field_validator("questions")
    @classmethod
    def _non_empty(cls, value):
        if not value:
            raise ValueError("at least one question is required")
        return value


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    forward_passes: int = 0


class SystemOneResponse(BaseModel):
    model: str
    answers: Dict[str, Dict[str, Any]]
    usage: Usage
