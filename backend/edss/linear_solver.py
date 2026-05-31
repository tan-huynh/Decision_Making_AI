from __future__ import annotations

import re
from typing import Any

import numpy as np

from .linear_programming import solve_lp


def recognize_linear_programming(problem: dict[str, Any]) -> dict[str, Any]:
    description = problem.get("context", {}).get("description", "") or ""
    lowered = description.lower()
    variables = _variable_names(problem)
    objective = problem.get("objective") or {}
    constraints = _constraints_summary(problem)
    bounds = problem.get("bounds")
    variable_type = _variable_type(problem)
    objective_expr = _objective_expression(problem)
    nonlinear_evidence = _nonlinear_evidence(objective_expr, constraints)
    evidence: list[str] = []
    missing: list[str] = []

    if problem.get("problem_type") == "linear_programming" or problem.get("c") or objective:
        evidence.append("Structured LP data/objective present.")
    if any(token in lowered for token in ["linear programming", "simplex", "product mix", "resource", "capacity", "shadow price", "slack"]):
        evidence.append("Description contains LP/simplex/resource keywords.")
    if variables:
        evidence.append("Decision variables are available.")
    if constraints:
        evidence.append("Linear constraint candidates are available.")

    if not variables:
        missing.append("decision_variables")
    if not (problem.get("c") or objective):
        missing.append("objective")
    if not constraints:
        missing.append("constraints")
    if not _has_nonnegativity(problem, bounds):
        missing.append("non_negativity_or_bounds")

    direction = problem.get("sense") or objective.get("sense") or problem.get("context", {}).get("objective_direction") or ""
    if str(direction).lower() not in {"maximize", "max", "minimize", "min"}:
        missing.append("objective_direction")

    all_linear = not nonlinear_evidence
    obj_linear = not _nonlinear_evidence(objective_expr, [])
    subtype = _subtype(problem, variables, constraints, lowered)
    method = _method(subtype)
    confidence = 0.25 + 0.2 * bool(variables) + 0.2 * bool(problem.get("c") or objective) + 0.2 * bool(constraints) + 0.15 * all_linear
    if problem.get("problem_type") == "linear_programming":
        confidence += 0.1
    if variable_type in {"integer", "binary"} or not all_linear:
        confidence = min(confidence, 0.84)

    return {
        "problem_type": "Linear Programming",
        "subtype": subtype,
        "confidence": round(min(confidence, 0.99), 2),
        "evidence": evidence + nonlinear_evidence,
        "decision_variables": variables,
        "objective": {
            "direction": "maximize" if str(direction).lower().startswith("max") else "minimize",
            "expression": objective_expr,
            "is_linear": obj_linear,
        },
        "constraints": constraints,
        "all_constraints_linear": all_linear,
        "non_negativity_constraints": _nonnegativity_constraints(problem, variables, bounds),
        "variable_type": variable_type,
        "solution_method": method,
        "missing_information": missing,
        "can_solve": confidence >= 0.85 and not missing and all_linear and variable_type == "continuous",
    }


def solve_linear_programming_problem(problem: dict[str, Any]) -> dict[str, Any]:
    recognition = recognize_linear_programming(problem)
    gate_md = _recognition_markdown(recognition)
    if recognition["variable_type"] in {"integer", "binary"}:
        return {
            "status": "redirect_required",
            "solver": "linear_programming_recognition_gate",
            "target_agent": "integer_programming",
            "recognition": recognition,
            "missing_data": recognition["missing_information"],
            "markdown_report": gate_md + "\nBài này có biến nguyên/binary, cần chuyển sang Integer Programming Agent.\n",
        }
    if not recognition["objective"]["is_linear"] or not recognition["all_constraints_linear"]:
        return {
            "status": "redirect_required",
            "solver": "linear_programming_recognition_gate",
            "target_agent": "nonlinear_programming",
            "recognition": recognition,
            "missing_data": recognition["missing_information"],
            "markdown_report": gate_md + "\nObjective hoặc constraints phi tuyến, cần chuyển sang Nonlinear Programming Agent.\n",
        }
    if not recognition["can_solve"]:
        return {
            "status": "needs_clarification",
            "solver": "linear_programming_recognition_gate",
            "recognition": recognition,
            "missing_data": recognition["missing_information"],
            "markdown_report": gate_md + "\nChưa đủ dữ liệu để lập/giải LP và kết luận nghiệm tối ưu.\n",
        }

    result = solve_lp(problem)
    verification = verify_linear_solution(recognition, result, problem)
    result["recognition"] = recognition
    result["verification"] = verification
    result["markdown_report"] = gate_md + "\n" + result.get("markdown_report", "") + _verification_markdown(verification)
    return result


def verify_linear_solution(recognition: dict[str, Any], result: dict[str, Any], problem: dict[str, Any]) -> dict[str, Any]:
    checks: list[str] = []
    passed = result.get("status") == "optimal"
    if not recognition["objective"]["is_linear"]:
        passed = False
        checks.append("Objective is nonlinear; LP solution is not valid.")
    if not recognition["all_constraints_linear"]:
        passed = False
        checks.append("At least one constraint is nonlinear; LP solution is not valid.")
    if recognition["variable_type"] != "continuous":
        passed = False
        checks.append("Variable type is not continuous.")

    names = _variable_names(problem)
    values = result.get("all_values") or result.get("solution") or {}
    x = np.asarray([float(values.get(name, 0.0)) for name in names], dtype=float) if names else np.asarray([])
    if names:
        for name, value in zip(names, x):
            if value < -1e-6:
                passed = False
                checks.append(f"{name} violates non-negativity.")
        checks.append("Non-negativity checked.")

    if problem.get("c") and names:
        objective = float(np.asarray(problem["c"], dtype=float) @ x)
        if abs(objective - float(result.get("objective_value", objective))) > 1e-5:
            passed = False
            checks.append("Objective value does not match c^T x.")
        else:
            checks.append("Objective recomputed from c^T x.")
    if problem.get("A_ub") and names:
        lhs = np.asarray(problem["A_ub"], dtype=float) @ x
        rhs = np.asarray(problem["b_ub"], dtype=float)
        if np.any(lhs - rhs > 1e-6):
            passed = False
            checks.append("One or more <= constraints are violated.")
        else:
            checks.append("All <= constraints satisfy feasibility tolerance.")
    if problem.get("A_eq") and names:
        lhs = np.asarray(problem["A_eq"], dtype=float) @ x
        rhs = np.asarray(problem["b_eq"], dtype=float)
        if np.any(np.abs(lhs - rhs) > 1e-6):
            passed = False
            checks.append("One or more equality constraints are violated.")
        else:
            checks.append("All equality constraints satisfy feasibility tolerance.")
    if result.get("binding_constraints") is not None:
        checks.append(f"Binding constraints identified: {result.get('binding_constraints')}.")
    if result.get("slacks") is not None:
        checks.append("Slack/surplus values reported by solver.")
    return {"passed": passed, "checks": checks}


def _variable_names(problem: dict[str, Any]) -> list[str]:
    if problem.get("variable_names"):
        return list(problem["variable_names"])
    if problem.get("variables"):
        return [item.get("name") for item in problem.get("variables", [])]
    if problem.get("c"):
        return [f"x{i + 1}" for i in range(len(problem["c"]))]
    return []


def _constraints_summary(problem: dict[str, Any]) -> list[Any]:
    constraints: list[Any] = []
    constraints.extend(problem.get("constraints", []))
    for idx, row in enumerate(problem.get("A_ub") or [], start=1):
        rhs = (problem.get("b_ub") or [None])[idx - 1]
        constraints.append({"name": f"ub_{idx}", "coefficients": row, "operator": "<=", "rhs": rhs})
    for idx, row in enumerate(problem.get("A_eq") or [], start=1):
        rhs = (problem.get("b_eq") or [None])[idx - 1]
        constraints.append({"name": f"eq_{idx}", "coefficients": row, "operator": "=", "rhs": rhs})
    return constraints


def _variable_type(problem: dict[str, Any]) -> str:
    if problem.get("integrality") or problem.get("ip"):
        return "binary" if any(_is_binary_bound(b) for b in problem.get("bounds") or []) else "integer"
    types = {item.get("variable_type", "continuous") for item in problem.get("variables", [])}
    if "binary" in types or "zero-one" in types:
        return "binary"
    if "integer" in types:
        return "integer"
    if not types or types == {"continuous"}:
        return "continuous"
    return "unknown"


def _is_binary_bound(bound: Any) -> bool:
    try:
        return bound[0] == 0 and bound[1] == 1
    except Exception:
        return False


def _objective_expression(problem: dict[str, Any]) -> str:
    objective = problem.get("objective") or {}
    if objective.get("expression"):
        return objective["expression"]
    names = _variable_names(problem)
    if problem.get("c") and names:
        return " + ".join(f"{coef}*{name}" for coef, name in zip(problem["c"], names))
    if objective.get("coefficients"):
        return " + ".join(f"{coef}*{name}" for name, coef in objective["coefficients"].items())
    return str(objective)


def _nonlinear_evidence(objective_expression: str, constraints: list[Any]) -> list[str]:
    evidence: list[str] = []
    expressions = [objective_expression] + [str(item.get("expression", "")) for item in constraints if isinstance(item, dict)]
    for expr in expressions:
        cleaned = expr.lower()
        if not cleaned:
            continue
        if re.search(r"\b(sqrt|log|exp|sin|cos)\b|\^|[a-z]\w*\s*\*\s*[a-z]\w*", cleaned):
            evidence.append(f"Nonlinear expression detected: {expr}")
    return evidence


def _has_nonnegativity(problem: dict[str, Any], bounds: Any) -> bool:
    if problem.get("variables"):
        return all(item.get("lower_bound", 0) is not None for item in problem.get("variables", []))
    if bounds:
        return all(bound[0] is not None for bound in bounds)
    if problem.get("c"):
        return True
    return False


def _nonnegativity_constraints(problem: dict[str, Any], variables: list[str], bounds: Any) -> list[str]:
    if problem.get("variables"):
        return [f"{item.get('name')} >= {item.get('lower_bound', 0)}" for item in problem.get("variables", [])]
    if bounds:
        return [f"{name} >= {bound[0] if bound[0] is not None else 0}" for name, bound in zip(variables, bounds)]
    return [f"{name} >= 0" for name in variables]


def _subtype(problem: dict[str, Any], variables: list[str], constraints: list[Any], lowered: str) -> str:
    if "dual" in lowered or "shadow price" in lowered:
        return "LP_DUALITY"
    if "sensitivity" in lowered or "allowable" in lowered or "slack" in lowered:
        return "LP_SENSITIVITY_ANALYSIS"
    if problem.get("formulation_only") or "formulate" in lowered:
        return "LP_FORMULATION_ONLY"
    if "blend" in lowered or "mixture" in lowered:
        return "LP_BLENDING_OR_MIXTURE"
    if "transport" in lowered or "distribution" in lowered:
        return "LP_TRANSPORT_DISTRIBUTION_FORMULATION"
    if "crop" in lowered or "acre" in lowered:
        return "LP_CROP_PLANNING"
    if len(variables) == 2:
        return "LP_GRAPHICAL_2D"
    if any(item.get("operator") in {">=", "="} for item in constraints if isinstance(item, dict)):
        return "LP_BIG_M_OR_TWO_PHASE"
    return "LP_SIMPLEX_STANDARD"


def _method(subtype: str) -> str:
    return {
        "LP_FORMULATION_ONLY": "formulation_only",
        "LP_GRAPHICAL_2D": "graphical",
        "LP_DUALITY": "dual",
        "LP_SENSITIVITY_ANALYSIS": "sensitivity",
    }.get(subtype, "simplex")


def _recognition_markdown(r: dict[str, Any]) -> str:
    evidence = "\n".join(f"- {item}" for item in r["evidence"]) or "- Chưa có dấu hiệu đủ mạnh."
    missing = "\n".join(f"- {item}" for item in r["missing_information"]) or "- Không thiếu dữ liệu bắt buộc."
    return (
        "# Lời giải\n\n"
        "## 1. Nhận dạng dạng toán\n\n"
        f"- Dạng toán chính: {r['problem_type']}\n"
        f"- Dạng toán phụ: {r['subtype']}\n"
        f"- Vì sao là LP: objective linear = {r['objective']['is_linear']}, constraints linear = {r['all_constraints_linear']}\n"
        f"- Objective: {r['objective']['direction']} `{r['objective']['expression']}`\n"
        f"- Biến liên tục: {r['variable_type'] == 'continuous'}\n"
        f"- Method: {r['solution_method']}\n"
        f"- Mức tin cậy: {r['confidence']:.2f}\n\n"
        "Evidence:\n"
        f"{evidence}\n\n"
        "Missing information:\n"
        f"{missing}\n\n"
        "## 3. Biến quyết định\n\n"
        f"{', '.join(r['decision_variables'])}\n\n"
        "## 5. Ràng buộc\n\n"
        f"`{r['constraints']}`\n\n"
    )


def _verification_markdown(v: dict[str, Any]) -> str:
    checks = "\n".join(f"- {item}" for item in v.get("checks", []))
    return f"\n## 8. Kiểm tra nghiệm\n\n- Trạng thái kiểm tra: {'passed' if v.get('passed') else 'failed'}\n{checks}\n"
