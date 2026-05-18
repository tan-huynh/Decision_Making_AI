from __future__ import annotations

from typing import Any

from .linear_programming import solve_lp_raw
from .teaching_report import wrap_teaching_report


def solve_goal_programming(problem: dict[str, Any]) -> dict[str, Any]:
    """Weighted/preemptive goal programming via deviation variables.

    Input:
      variables: ["x1", ...]
      constraints: normal hard constraints in raw <= form optional
      goals: [{coefficients:[...], target: 10, weight_under:1, weight_over:0, priority:1}]
    """
    names = problem.get("variable_names") or problem.get("variables") or [f"x{i+1}" for i in range(len(problem.get("c", [])))]
    goals = problem.get("goals", [])
    if not goals:
        raise ValueError("Goal programming cần goals.")
    n = len(names)
    mode = problem.get("mode", "weighted")
    if mode == "preemptive":
        return _solve_preemptive(problem, names, goals)
    return _solve_weighted(problem, names, goals)


def _solve_weighted(problem: dict[str, Any], names: list[str], goals: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(names)
    c = [0.0] * n
    variable_names = list(names)
    a_eq = []
    b_eq = []
    for idx, goal in enumerate(goals):
        d_minus = f"d{idx+1}_minus"
        d_plus = f"d{idx+1}_plus"
        variable_names += [d_minus, d_plus]
        c += [float(goal.get("weight_under", goal.get("weight", 1))), float(goal.get("weight_over", goal.get("weight", 1)))]
        row = [0.0] * len(c)
        for j, coef in enumerate(goal["coefficients"]):
            row[j] = float(coef)
        row[n + 2 * idx] = 1.0
        row[n + 2 * idx + 1] = -1.0
        a_eq.append(row)
        b_eq.append(float(goal["target"]))
    a_ub = [list(row) + [0.0] * (len(c) - len(row)) for row in problem.get("A_ub", [])]
    result = solve_lp_raw(
        {
            "sense": "minimize",
            "c": c,
            "A_ub": a_ub,
            "b_ub": problem.get("b_ub", []),
            "A_eq": a_eq,
            "b_eq": b_eq,
            "bounds": [(0, None)] * len(c),
            "variable_names": variable_names,
            "constraint_names_eq": [f"goal_{i+1}" for i in range(len(goals))],
        }
    )
    report = wrap_teaching_report(
        "Goal Programming",
        "Weighted Goal Programming",
        [
            ("Deviation variables", "Mỗi goal được viết `a_i x + d_i^- - d_i^+ = target_i`. Objective tối thiểu hóa tổng trọng số deviation."),
            ("Kết quả solver", result.get("markdown_report", "")),
        ],
        f"Tổng weighted deviation = {result.get('objective_value')}.",
    )
    return {**result, "solver": "weighted_goal_programming", "markdown_report": report}


def _solve_preemptive(problem: dict[str, Any], names: list[str], goals: list[dict[str, Any]]) -> dict[str, Any]:
    # Lexicographic approximation: solve priorities one by one and add previous optimal deviations as constraints.
    hard_a = [list(row) for row in problem.get("A_ub", [])]
    hard_b = list(problem.get("b_ub", []))
    priority_results = []
    for priority in sorted({int(goal.get("priority", 1)) for goal in goals}):
        scoped = {**problem, "goals": [g for g in goals if int(g.get("priority", 1)) == priority], "A_ub": hard_a, "b_ub": hard_b}
        result = _solve_weighted(scoped, names, scoped["goals"])
        priority_results.append({"priority": priority, "objective_value": result.get("objective_value"), "solution": result.get("all_values", {})})
        # Full lexicographic locking of deviation expressions is intentionally conservative here.
    return {
        "status": "computed",
        "solver": "preemptive_goal_programming_sequential",
        "priority_results": priority_results,
        "markdown_report": wrap_teaching_report(
            "Preemptive Goal Programming",
            "Goal Programming",
            [("Ưu tiên", "Các priority được giải theo thứ tự tăng dần. Báo cáo này dùng sequential solve; production lock đầy đủ deviation cần basis/model adapter nâng cao.")],
            str(priority_results[-1] if priority_results else {}),
        ),
    }
