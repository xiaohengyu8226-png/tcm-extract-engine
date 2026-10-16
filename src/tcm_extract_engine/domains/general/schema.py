from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class KnowledgeFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["definition", "relation", "quantity", "rule", "process", "event"] = "relation"
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = ""
    conditions: list[str] = Field(default_factory=list)
    scope: list[str] = Field(default_factory=list)
    negated: bool = False
    modality: str | None = None
    exceptions: list[str] = Field(default_factory=list)
    quantities: list[dict[str, Any]] = Field(default_factory=list)
    time: str | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def meaningful(self) -> KnowledgeFact:
        if not self.subject.strip() or not self.predicate.strip():
            raise ValueError("subject/predicate 不可空白")
        if not (self.object.strip() or self.quantities or self.steps):
            raise ValueError("事实必须有 object、quantities 或 steps")
        return self
