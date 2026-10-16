"""Validated visual observations and diagram structure, preserving source labels."""
from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class VisualFact(BaseModel):
    model_config = ConfigDict(extra='allow')
    content: str = Field(min_length=1)
    evidence: str = ''
    conditions: list[str] = Field(default_factory=list)
    region: list[float] | None = None


class VisualStep(BaseModel):
    model_config = ConfigDict(extra='allow', populate_by_name=True)
    step_id: str = Field(validation_alias=AliasChoices('step_id', 'id'))
    description: str = Field(min_length=1, validation_alias=AliasChoices('description', 'label'))
    next_steps: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def normalize_next(cls, value: Any) -> Any:
        if isinstance(value, dict) and 'next_steps' not in value:
            value = dict(value)
            next_value = value.get('next')
            value['next_steps'] = [next_value] if isinstance(next_value, str) else next_value or []
        return value


class VisualRelation(BaseModel):
    model_config = ConfigDict(extra='allow', populate_by_name=True)
    source: str = Field(min_length=1, validation_alias=AliasChoices('from', 'source'), serialization_alias='from')
    target: str = Field(min_length=1, validation_alias=AliasChoices('to', 'target'), serialization_alias='to')
    relation_type: str = ''
    condition: str | None = None


class VisualData(BaseModel):
    # Figure-specific measurements remain losslessly retained. Facts/steps/edges
    # have a fixed runtime schema and can be handed off as individually identified items.
    model_config = ConfigDict(extra='allow')
    facts: list[VisualFact] = Field(default_factory=list)
    steps: list[VisualStep] = Field(default_factory=list)
    relations: list[VisualRelation] = Field(default_factory=list)
    decision_points: list[dict[str, Any]] = Field(default_factory=list)
    data_points: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def string_steps(cls, value: Any) -> Any:
        if isinstance(value, dict) and isinstance(value.get('steps'), list):
            value = dict(value)
            value['steps'] = [({'step_id': f'step-{i+1}', 'description': step} if isinstance(step, str) else step)
                              for i, step in enumerate(value['steps'])]
        return value
