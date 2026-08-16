"""Structured output schemas.

Every field is required and non-null: providers omit whatever is not marked
required, and a missing field is worse than a wrong one because it silently
changes what a run means.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1"

Importance = Literal["critical", "high", "normal", "low"]
SuggestedAction = Literal["inbox", "archive", "delete", "unsubscribe"]


class Classification(BaseModel):
    """One LLM opinion about one email."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="One short factual sentence about the important content.")
    type: str = Field(description="What kind of email this is, snake_case, e.g. receipt.")
    category: str = Field(description="Broader area of life, e.g. Travel.")
    importance: Importance
    needs_action: bool
    suggested_action: SuggestedAction
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_short: str = Field(description="Very brief justification, for debugging.")


class LabelMapping(BaseModel):
    """One entry of a refined taxonomy: source label -> canonical label."""

    model_config = ConfigDict(extra="forbid")

    source_label: str
    canonical_label: str


class LabelMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mappings: list[LabelMapping]


def json_schema(model: type[BaseModel]) -> dict:
    """A strict JSON schema suitable for provider structured-output modes."""
    schema = model.model_json_schema()
    _strictify(schema)
    return schema


def _strictify(node: object) -> None:
    """Mark every object node required and closed, as strict modes demand."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["required"] = list(node["properties"])
            node["additionalProperties"] = False
        for value in node.values():
            _strictify(value)
    elif isinstance(node, list):
        for item in node:
            _strictify(item)
