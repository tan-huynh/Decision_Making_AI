from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EDSSContext(BaseModel):
    title: str = Field(default="Engineering decision case", max_length=240)
    domain: str = Field(default="engineering", max_length=80)
    decision_maker: str = Field(default="engineer", max_length=120)
    objective_direction: Literal["maximize", "minimize"] = "maximize"
    unit: str = Field(default="", max_length=80)
    time_horizon: str = Field(default="", max_length=120)
    description: str = Field(default="", max_length=5000)


class EDSSVariable(BaseModel):
    name: str
    description: str = ""
    variable_type: Literal["continuous", "integer", "binary"] = "continuous"
    lower_bound: float = 0
    upper_bound: float | None = None
    unit: str = ""


class EDSSObjective(BaseModel):
    sense: Literal["maximize", "minimize"] = "maximize"
    coefficients: dict[str, float] = Field(default_factory=dict)
    constant: float = 0
    expression: str = ""


class EDSSConstraint(BaseModel):
    name: str
    coefficients: dict[str, float]
    operator: Literal["<=", ">=", "="] = "<="
    rhs: float
    resource: str = ""
    unit: str = ""


class EDSSAlternative(BaseModel):
    name: str
    description: str = ""
    attributes: dict[str, float] = Field(default_factory=dict)


class EDSSState(BaseModel):
    name: str
    probability: float = Field(ge=0)


class EDSSPayoffCell(BaseModel):
    alternative: str
    state: str
    payoff: float
    cost: float = 0
    loss: float | None = None


class EDSSDecisionNode(BaseModel):
    id: str
    node_type: Literal["decision", "chance", "outcome"]
    label: str
    value: float | None = None
    probability: float | None = None
    children: list[str] = Field(default_factory=list)


class EDSSProblem(BaseModel):
    context: EDSSContext
    problem_type: str | None = None
    variables: list[EDSSVariable] = Field(default_factory=list)
    objective: EDSSObjective | None = None
    constraints: list[EDSSConstraint] = Field(default_factory=list)
    alternatives: list[EDSSAlternative] = Field(default_factory=list)
    states: list[EDSSState] = Field(default_factory=list)
    payoff_matrix: list[EDSSPayoffCell] = Field(default_factory=list)
    decision_tree: list[EDSSDecisionNode] = Field(default_factory=list)
    graph: dict[str, Any] = Field(default_factory=dict)
    assignment_costs: list[list[float]] = Field(default_factory=list)
    stages: list[dict[str, Any]] = Field(default_factory=list)
    objectives: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class NaturalProblemRequest(BaseModel):
    description: str = Field(min_length=4, max_length=8000)
    domain: str = Field(default="engineering", max_length=80)

