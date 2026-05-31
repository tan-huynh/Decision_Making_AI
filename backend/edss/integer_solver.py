from __future__ import annotations

from typing import Any

import numpy as np

from .mip import solve_mip


def recognize_integer_programming(problem: dict[str, Any]) -> dict[str, Any]:
    spec = _ip_spec(problem)
    description = problem.get("context", {}).get("description", "") or ""
    lowered = description.lower()
    variables = _variable_names(problem, spec)
    c = spec.get("c")
    integrality = spec.get("integrality")
    bounds = spec.get("bounds") or [(0, None)] * len(variables)
    constraints = _constraints_summary(problem, spec)
    logical_constraints = list(spec.get("logical_constraints") or problem.get("logical_constraints") or [])
    evidence: list[str] = []
    missing: list[str] = []

    if problem.get("problem_type") == "integer_programming" or spec:
        evidence.append("Structured Integer Programming/MIP data present.")
    if any(token in lowered for token in ["integer", "binary", "yes/no", "open", "close", "fixed charge", "facility", "shift", "covering"]):
        evidence.append("Description contains integer/binary/logical decision keywords.")
    if variables:
        evidence.append("Decision variables are available.")
    if integrality:
        evidence.append("Integrality vector is available.")
    if constraints:
        evidence.append("Linear constraints are available.")

    if not variables:
        missing.append("decision_variables")
    if not c and not problem.get("objective"):
        missing.append("objective")
    if not constraints and not problem.get("assignment_costs"):
        missing.append("constraints")
    if integrality is None and not _variables_have_integer_types(problem):
        missing.append("integrality_requirements")

    integer_variables, binary_variables, continuous_variables = _classify_variables(variables, integrality, bounds, problem)
    subtype = _subtype(problem, spec, integer_variables, binary_variables, continuous_variables, lowered)
    confidence = 0.3 + 0.2 * bool(variables) + 0.2 * bool(c or problem.get("objective")) + 0.15 * bool(constraints) + 0.2 * bool(integer_variables or binary_variables)
    if problem.get("problem_type") == "integer_programming":
        confidence += 0.1
    direction = spec.get("sense") or problem.get("sense") or (problem.get("objective") or {}).get("sense") or "minimize"

    return {
        "problem_type": "Integer Programming",
        "subtype": subtype,
        "confidence": round(min(confidence, 0.99), 2),
        "evidence": evidence,
        "decision_variables": variables,
        "integer_variables": integer_variables,
        "binary_variables": binary_variables,
        "continuous_variables": continuous_variables,
        "objective": {
            "direction": "maximize" if str(direction).lower().startswith("max") else "minimize",
            "expression": _objective_expression(problem, spec),
        },
        "constraints": constraints,
        "logical_constraints": logical_constraints,
        "big_m_constraints": list(spec.get("big_m_constraints") or problem.get("big_m_constraints") or []),
        "integrality_requirements": _integrality_requirements(integer_variables, binary_variables, continuous_variables),
        "missing_information": missing,
        "can_solve": confidence >= 0.85 and not missing,
    }


def solve_integer_programming_problem(problem: dict[str, Any]) -> dict[str, Any]:
    recognition = recognize_integer_programming(problem)
    gate_md = _recognition_markdown(recognition)
    if not recognition["can_solve"]:
        return {
            "status": "needs_clarification",
            "solver": "integer_programming_recognition_gate",
            "recognition": recognition,
            "missing_data": recognition["missing_information"],
            "markdown_report": gate_md + "\nChưa đủ dữ liệu để lập/giải IP và kết luận nghiệm tối ưu.\n",
        }

    spec = _ip_spec(problem)
    result = solve_mip(spec)
    verification = verify_integer_solution(recognition, result, spec)
    result["recognition"] = recognition
    result["verification"] = verification
    result["markdown_report"] = gate_md + "\n" + result.get("markdown_report", "") + _verification_markdown(verification)
    return result


def verify_integer_solution(recognition: dict[str, Any], result: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    checks: list[str] = []
    passed = result.get("status") == "optimal"
    solution = result.get("solution", {})
    names = spec.get("variable_names") or recognition.get("decision_variables") or []
    x = np.asarray([solution.get(name, 0.0) for name in names], dtype=float)
    integrality = spec.get("integrality") or [1] * len(names)
    bounds = spec.get("bounds") or [(0, None)] * len(names)

    for idx, (name, value) in enumerate(zip(names, x)):
        code = int(integrality[idx]) if idx < len(integrality) else 1
        lb, ub = bounds[idx] if idx < len(bounds) else (0, None)
        if lb is not None and value < float(lb) - 1e-6:
            passed = False
            checks.append(f"{name} violates lower bound.")
        if ub is not None and value > float(ub) + 1e-6:
            passed = False
            checks.append(f"{name} violates upper bound.")
        if code != 0 and abs(value - round(value)) > 1e-6:
            passed = False
            checks.append(f"{name} is not integer.")
        if code != 0 and lb == 0 and ub == 1 and value not in {0.0, 1.0} and abs(value - round(value)) <= 1e-6:
            checks.append(f"{name} satisfies binary bounds.")

    if spec.get("A_ub"):
        lhs = np.asarray(spec["A_ub"], dtype=float) @ x
        rhs = np.asarray(spec["b_ub"], dtype=float)
        if np.any(lhs - rhs > 1e-6):
            passed = False
            checks.append("One or more <= constraints are violated.")
        else:
            checks.append("All <= constraints satisfy feasibility tolerance.")
    if spec.get("A_eq"):
        lhs = np.asarray(spec["A_eq"], dtype=float) @ x
        rhs = np.asarray(spec["b_eq"], dtype=float)
        if np.any(np.abs(lhs - rhs) > 1e-6):
            passed = False
            checks.append("One or more equality constraints are violated.")
        else:
            checks.append("All equality constraints satisfy feasibility tolerance.")
    if spec.get("c") and names:
        objective = float(np.asarray(spec["c"], dtype=float) @ x)
        if abs(objective - float(result.get("objective_value", objective))) > 1e-5:
            passed = False
            checks.append("Objective value does not match c^T x.")
        else:
            checks.append("Objective value recomputed from c^T x.")
    if not checks:
        checks.append("Solver returned an optimal integer solution.")
    return {"passed": passed, "checks": checks}


def _ip_spec(problem: dict[str, Any]) -> dict[str, Any]:
    spec = dict(problem.get("ip") or {})
    for key in ["sense", "c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds", "integrality", "variable_names"]:
        if key not in spec and problem.get(key) is not None:
            spec[key] = problem[key]
    if "variable_names" not in spec and problem.get("variables"):
        spec["variable_names"] = [item.get("name") for item in problem.get("variables", [])]
    if "bounds" not in spec and problem.get("variables"):
        spec["bounds"] = [(item.get("lower_bound", 0), item.get("upper_bound")) for item in problem.get("variables", [])]
    if "integrality" not in spec and problem.get("variables"):
        spec["integrality"] = [0 if item.get("variable_type", "continuous") == "continuous" else 1 for item in problem.get("variables", [])]
    if "c" not in spec and problem.get("objective") and spec.get("variable_names"):
        coeffs = problem["objective"].get("coefficients", {})
        spec["c"] = [coeffs.get(name, 0.0) for name in spec["variable_names"]]
        spec["sense"] = problem["objective"].get("sense", spec.get("sense", "minimize"))
    return spec


def _variable_names(problem: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    return list(spec.get("variable_names") or problem.get("variable_names") or [item.get("name") for item in problem.get("variables", [])])


def _variables_have_integer_types(problem: dict[str, Any]) -> bool:
    return any(item.get("variable_type") in {"integer", "binary", "zero-one"} for item in problem.get("variables", []))


def _classify_variables(
    names: list[str],
    integrality: list[int] | None,
    bounds: list[Any],
    problem: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    integer_variables: list[str] = []
    binary_variables: list[str] = []
    continuous_variables: list[str] = []
    variable_types = {item.get("name"): item.get("variable_type", "continuous") for item in problem.get("variables", [])}
    for idx, name in enumerate(names):
        vtype = variable_types.get(name)
        code = int(integrality[idx]) if integrality is not None and idx < len(integrality) else (0 if vtype == "continuous" else 1)
        lb, ub = bounds[idx] if idx < len(bounds) else (0, None)
        if vtype in {"binary", "zero-one"} or (code != 0 and lb == 0 and ub == 1):
            binary_variables.append(name)
        elif code != 0:
            integer_variables.append(name)
        else:
            continuous_variables.append(name)
    return integer_variables, binary_variables, continuous_variables


def _constraints_summary(problem: dict[str, Any], spec: dict[str, Any]) -> list[Any]:
    constraints: list[Any] = []
    constraints.extend(problem.get("constraints", []))
    for idx, row in enumerate(spec.get("A_ub") or [], start=1):
        rhs = (spec.get("b_ub") or [None])[idx - 1]
        constraints.append({"name": f"ub_{idx}", "coefficients": row, "operator": "<=", "rhs": rhs})
    for idx, row in enumerate(spec.get("A_eq") or [], start=1):
        rhs = (spec.get("b_eq") or [None])[idx - 1]
        constraints.append({"name": f"eq_{idx}", "coefficients": row, "operator": "=", "rhs": rhs})
    return constraints


def _objective_expression(problem: dict[str, Any], spec: dict[str, Any]) -> str:
    if problem.get("objective", {}).get("expression"):
        return problem["objective"]["expression"]
    names = spec.get("variable_names") or []
    c = spec.get("c") or []
    if names and c:
        return " + ".join(f"{coef}*{name}" for coef, name in zip(c, names))
    return str(problem.get("objective") or "")


def _subtype(
    problem: dict[str, Any],
    spec: dict[str, Any],
    integer_variables: list[str],
    binary_variables: list[str],
    continuous_variables: list[str],
    lowered: str,
) -> str:
    if problem.get("assignment_costs") or "assignment" in lowered:
        return "IP_ASSIGNMENT"
    if any(token in lowered for token in ["shift", "workforce", "covering", "schedule"]):
        return "IP_SHIFT_COVERING"
    if any(token in lowered for token in ["facility", "plant", "warehouse", "open", "close"]):
        return "IP_FACILITY_LOCATION"
    if any(token in lowered for token in ["fixed charge", "setup"]) or spec.get("logical_constraints"):
        return "IP_FIXED_CHARGE_PRODUCTION"
    if continuous_variables and (integer_variables or binary_variables):
        return "IP_MIXED_INTEGER_LINEAR_PROGRAMMING"
    if binary_variables and not integer_variables:
        return "IP_BINARY_SELECTION"
    return "IP_GENERAL_INTEGER"


def _integrality_requirements(integer_variables: list[str], binary_variables: list[str], continuous_variables: list[str]) -> list[str]:
    items = [f"{name} ∈ {{0,1}}" for name in binary_variables]
    items.extend(f"{name} ∈ Z_+" for name in integer_variables)
    items.extend(f"{name} continuous" for name in continuous_variables)
    return items


def _recognition_markdown(r: dict[str, Any]) -> str:
    evidence = "\n".join(f"- {item}" for item in r["evidence"]) or "- Chưa có dấu hiệu đủ mạnh."
    missing = "\n".join(f"- {item}" for item in r["missing_information"]) or "- Không thiếu dữ liệu bắt buộc."
    return (
        "# Lời giải\n\n"
        "## 1. Nhận dạng dạng toán\n\n"
        f"- Dạng toán chính: {r['problem_type']}\n"
        f"- Dạng toán phụ: {r['subtype']}\n"
        f"- Vì sao cần biến nguyên/binary: {', '.join(r['integrality_requirements']) or 'chưa rõ'}\n"
        f"- Objective: {r['objective']['direction']} `{r['objective']['expression']}`\n"
        "- Method: MILP solver / branch-and-bound; không làm tròn nghiệm LP relaxation.\n"
        f"- Mức tin cậy: {r['confidence']:.2f}\n\n"
        "Evidence:\n"
        f"{evidence}\n\n"
        "Missing information:\n"
        f"{missing}\n\n"
        "## 3. Biến quyết định\n\n"
        f"- Integer variables: {', '.join(r['integer_variables']) or 'none'}\n"
        f"- Binary variables: {', '.join(r['binary_variables']) or 'none'}\n"
        f"- Continuous variables: {', '.join(r['continuous_variables']) or 'none'}\n\n"
        "## 5. Ràng buộc\n\n"
        f"`{r['constraints']}`\n\n"
    )


def _verification_markdown(v: dict[str, Any]) -> str:
    checks = "\n".join(f"- {item}" for item in v.get("checks", []))
    return f"\n## 7. Kiểm tra nghiệm\n\n- Trạng thái kiểm tra: {'passed' if v.get('passed') else 'failed'}\n{checks}\n"
