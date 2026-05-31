from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EDSSContext(BaseModel):
    title: str = Field(default="Engineering decision case", max_length=240)
    domain: str = Field(default="engineering", max_length=80)
    decision_maker: str = Field(default="engineer", max_length=120)
    objective_direction: str = Field(default="maximize", max_length=40)
    unit: str = Field(default="", max_length=80)
    time_horizon: str = Field(default="", max_length=120)
    description: str = Field(default="", max_length=5000)


class EDSSVariable(BaseModel):
    name: str
    description: str = ""
    variable_type: str = Field(default="continuous", max_length=40)
    lower_bound: float = 0
    upper_bound: float | None = None
    unit: str = ""


class EDSSObjective(BaseModel):
    sense: str = Field(default="maximize", max_length=40)
    coefficients: dict[str, float] = Field(default_factory=dict)
    constant: float = 0
    expression: str = ""


class EDSSConstraint(BaseModel):
    name: str
    coefficients: dict[str, float]
    operator: str = Field(default="<=", max_length=10)
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
    node_type: str = Field(default="chance", max_length=40)
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
    game: dict[str, Any] | None = None
    assignment_costs: list[list[float]] = Field(default_factory=list)
    stages: list[dict[str, Any]] = Field(default_factory=list)
    objectives: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    probability_tree: dict[str, Any] | None = None
    bayes: dict[str, Any] | None = None
    diagnostic_decision: dict[str, Any] | None = None
    imperfect_information_decision: dict[str, Any] | None = None
    forklift_decision: dict[str, Any] | None = None
    independent_probabilities: list[float] | None = None
    resource_allocation: dict[str, Any] | None = None
    inventory_dp: dict[str, Any] | None = None
    periods: int | None = None
    demands: list[float] | None = None
    max_inventory: int | None = None
    initial_inventory: int | None = None
    max_order: int | None = None
    fixed_order_cost: float | None = None
    queueing: dict[str, Any] | None = None
    arrival_rate: float | None = None
    service_rate: float | None = None
    servers: int | None = None
    queue_capacity: int | None = None
    population_size: int | None = None
    waiting_cost: float | None = None
    service_cost: float | None = None
    optimize_servers: bool | None = None
    max_servers: int | None = None
    annual_demand: float | None = None
    weekly_demand: float | None = None
    order_cost: float | None = None
    holding_cost: float | None = None
    holding_cost_rate: float | None = None
    shortage_cost: float | None = None
    purchase_cost: float | None = None
    gross_profit_per_unit: float | None = None
    lead_time: float | None = None
    price_breaks: list[dict[str, float]] | None = None
    ip: dict[str, Any] | None = None
    c: list[float] | None = None
    A_ub: list[list[float]] | None = None
    b_ub: list[float] | None = None
    A_eq: list[list[float]] | None = None
    b_eq: list[float] | None = None
    integrality: list[int] | None = None
    nlp: dict[str, Any] | None = None
    variable_names: list[str] | None = None
    initial: list[float] | None = None
    bounds: list[tuple[float | None, float | None]] | None = None
    sense: str | None = None
    markov: dict[str, Any] | None = None
    markov_chain: dict[str, Any] | None = None
    markov_states: list[Any] | None = None
    transition_matrix: list[list[float]] | None = None
    initial_distribution: list[float] | None = None
    requested_outputs: list[str] | None = None
    absorbing_states: list[Any] | None = None
    state_costs: list[Any] | None = None
    target_state_for_first_passage: str | int | None = None
    n_steps: int | None = None
    time_step: str | None = None


class NaturalProblemRequest(BaseModel):
    description: str = Field(min_length=4, max_length=8000)
    domain: str = Field(default="engineering", max_length=80)
