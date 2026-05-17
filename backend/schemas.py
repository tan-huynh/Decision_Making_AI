from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Scenario(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    probability: float = Field(ge=0, le=1)
    utility: float = Field(ge=-1000, le=1000)
    evidence: str | None = Field(default="", max_length=500)


class DecisionOption(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=160)
    cost: float = Field(ge=0, le=1000)
    reversibility: float = Field(ge=0, le=1)
    scenarios: list[Scenario] = Field(min_length=1, max_length=12)


class DecisionRequest(BaseModel):
    question: str = Field(min_length=4, max_length=1200)
    domain: str = Field(default="life", max_length=80)
    objective: str = Field(min_length=4, max_length=1200)
    context: str = Field(default="", max_length=5000)
    riskTolerance: float = Field(default=0.5, ge=0, le=1)
    timeHorizon: str = Field(default="", max_length=160)
    model: str = Field(default="llama3.1", max_length=120)
    webRealtime: bool = True
    options: list[DecisionOption] = Field(min_length=2, max_length=8)

    @field_validator("domain")
    @classmethod
    def normalize_domain(cls, value: str) -> str:
        return value.strip().lower() or "life"


class LearnRequest(BaseModel):
    decision: DecisionRequest
    result: dict
    outcome: Literal["success", "mixed", "failure"]
    chosen_option: str


class GenerateModelRequest(BaseModel):
    question: str = Field(min_length=4, max_length=1200)
    domain: str = Field(default="life", max_length=80)
    objective: str = Field(default="Tối ưu quyết định theo utility, rủi ro và khả năng đảo ngược.", max_length=1200)
    context: str = Field(default="", max_length=5000)
    riskTolerance: float = Field(default=0.5, ge=0, le=1)
    timeHorizon: str = Field(default="", max_length=160)
    model: str = Field(default="llama3.1", max_length=120)
    webRealtime: bool = True
