from __future__ import annotations

from typing import Any


def solve_resource_allocation_dp(total_resource: int, stage_returns: list[list[float]], resource_name: str = "resource") -> dict[str, Any]:
    if total_resource < 0 or not stage_returns:
        raise ValueError("Resource allocation DP requires total resource and stage return tables.")
    stages = len(stage_returns)
    dp = [[float("-inf")] * (total_resource + 1) for _ in range(stages + 1)]
    choice = [[0] * (total_resource + 1) for _ in range(stages + 1)]
    dp[0][0] = 0.0
    for stage in range(1, stages + 1):
        returns = stage_returns[stage - 1]
        for used in range(total_resource + 1):
            best_value = float("-inf")
            best_amount = 0
            for amount, profit in enumerate(returns):
                if amount <= used and dp[stage - 1][used - amount] != float("-inf"):
                    value = dp[stage - 1][used - amount] + float(profit)
                    if value > best_value:
                        best_value = value
                        best_amount = amount
            dp[stage][used] = best_value
            choice[stage][used] = best_amount

    allocation = []
    remaining = total_resource
    for stage in range(stages, 0, -1):
        amount = choice[stage][remaining]
        allocation.append(
            {
                "stage": stage,
                resource_name: amount,
                "profit": stage_returns[stage - 1][amount],
            }
        )
        remaining -= amount
    allocation.reverse()
    tables = [
        {
            "stage": stage,
            "values": {resource: dp[stage][resource] for resource in range(total_resource + 1) if dp[stage][resource] != float("-inf")},
            "choice": {resource: choice[stage][resource] for resource in range(total_resource + 1) if dp[stage][resource] != float("-inf")},
        }
        for stage in range(1, stages + 1)
    ]
    return {
        "status": "optimal",
        "solver": "resource_allocation_dynamic_programming",
        "objective_value": dp[stages][total_resource],
        "allocation": allocation,
        "dp_tables": tables,
        "recommendation": f"Allocate {resource_name} by stage as {[item[resource_name] for item in allocation]} to maximize total return {dp[stages][total_resource]:.3f}.",
    }


def solve_finite_horizon_dp(problem: dict[str, Any]) -> dict[str, Any]:
    stages = problem.get("stages", [])
    if not stages:
        raise ValueError("DP requires stages with states/actions/transitions.")
    value_next: dict[str, float] = {}
    policy: list[dict[str, Any]] = []
    for stage_index in range(len(stages) - 1, -1, -1):
        stage = stages[stage_index]
        value_current: dict[str, float] = {}
        stage_policy: dict[str, str] = {}
        for state in stage.get("states", []):
            best_value = float("-inf")
            best_action = None
            for action in stage.get("actions", []):
                transitions = stage.get("transitions", {}).get(state, {}).get(action, [])
                expected = float(stage.get("rewards", {}).get(state, {}).get(action, 0))
                for transition in transitions:
                    expected += float(transition.get("probability", 1)) * value_next.get(transition.get("to"), 0)
                if expected > best_value:
                    best_value = expected
                    best_action = action
            value_current[state] = best_value if best_action is not None else 0.0
            stage_policy[state] = best_action or "none"
        value_next = value_current
        policy.append({"stage": stage_index, "value": value_current, "policy": stage_policy})
    policy.reverse()
    return {"status": "computed", "solver": "finite_horizon_backward_induction", "policy": policy, "initial_value": policy[0]["value"] if policy else {}}
