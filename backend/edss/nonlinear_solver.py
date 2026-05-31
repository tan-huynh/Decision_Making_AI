from __future__ import annotations

from typing import Any

import numpy as np

from .nonlinear import solve_circle_packing_box, solve_nonlinear


def recognize_nonlinear_programming(problem: dict[str, Any]) -> dict[str, Any]:
    spec = problem.get("nlp", {})
    description = problem.get("context", {}).get("description", "") or ""
    lowered = description.lower()
    variable_names = spec.get("variable_names") or problem.get("variable_names") or [v.get("name") for v in problem.get("variables", [])]
    objective = spec.get("objective") or problem.get("objective") or {}
    constraints = spec.get("constraints") or problem.get("constraints", [])
    bounds = spec.get("bounds") or problem.get("bounds")
    evidence: list[str] = []
    missing: list[str] = []
    nonlinear_elements: list[str] = []

    obj_expr = objective.get("expression", "") if isinstance(objective, dict) else ""
    obj_type = objective.get("type") if isinstance(objective, dict) else None
    if obj_type in {"quadratic", "nonlinear"}:
        nonlinear_elements.append(str(obj_type))
    if any(token in obj_expr.lower() for token in ["^2", "**2", "sqrt", "log", "exp", "*", "xy"]):
        nonlinear_elements.append(obj_expr)
    for con in constraints:
        expr = str(con.get("expression", ""))
        if any(token in expr.lower() for token in ["^2", "**2", "sqrt", "log", "exp", "*"]):
            nonlinear_elements.append(expr)
    if any(token in lowered for token in ["nonlinear", "non-linear", "kkt", "kuhn", "lagrange", "quadratic", "circle", "packing", "utility"]):
        evidence.append("Có keyword NLP/KKT/Lagrange/phi tuyến.")
    if nonlinear_elements:
        evidence.append("Có objective hoặc constraint phi tuyến.")
    if variable_names:
        evidence.append("Có decision variables.")
    if constraints:
        evidence.append("Có constraints.")

    if problem.get("circle_packing") or problem.get("radii") or problem.get("problem_type") == "circle_packing_box":
        subtype = "NLP_GEOMETRIC_PACKING"
        method = "numerical_solver"
        nonlinear_elements.append("non-overlap distance constraints")
    elif not constraints:
        subtype = "NLP_UNCONSTRAINED"
        method = "calculus" if obj_expr else "numerical_solver"
    elif any(con.get("type") == "eq" or con.get("operator") == "=" for con in constraints):
        subtype = "NLP_EQUALITY_CONSTRAINED_LAGRANGE"
        method = "Lagrange"
    elif any(con.get("type", "ineq") == "ineq" or con.get("operator") in {"<=", ">="} for con in constraints):
        subtype = "NLP_INEQUALITY_CONSTRAINED_KKT"
        method = "KKT" if obj_expr else "numerical_solver"
    else:
        subtype = "NLP_FORMULATION_ONLY"
        method = "formulation_only"

    direction = spec.get("sense") or problem.get("sense") or (objective.get("sense") if isinstance(objective, dict) else None)
    if not direction:
        direction = problem.get("context", {}).get("objective_direction", "")
    if not variable_names:
        missing.append("decision_variables")
    if not objective and not (problem.get("radii") or problem.get("circle_packing")):
        missing.append("objective_function")
    if not direction:
        missing.append("objective_direction")
    if bounds is None and not problem.get("variables"):
        evidence.append("Domain constraints not explicit; default numerical solver may use unbounded variables.")

    is_nonlinear = bool(nonlinear_elements) or obj_type == "quadratic"
    confidence = 0.35 + 0.2 * bool(variable_names) + 0.2 * bool(objective) + 0.15 * bool(constraints) + 0.2 * is_nonlinear
    if problem.get("problem_type") in {"nonlinear_programming", "circle_packing_box"}:
        confidence += 0.1
    return {
        "problem_type": "Nonlinear Programming",
        "subtype": subtype,
        "confidence": round(min(confidence, 0.99), 2),
        "evidence": evidence,
        "decision_variables": variable_names or [],
        "objective": {
            "direction": "minimize" if str(direction).lower().startswith("min") else "maximize",
            "expression": obj_expr or str(objective),
            "is_nonlinear": is_nonlinear,
        },
        "constraints": constraints,
        "nonlinear_elements": nonlinear_elements,
        "domain_constraints": _domain_constraints(problem, spec),
        "differentiability_status": "unknown" if not is_nonlinear else "differentiable",
        "convexity_status": _convexity_status(objective),
        "solution_method": method,
        "missing_information": missing,
        "can_solve": confidence >= 0.85 and not missing,
    }


def solve_nonlinear_programming_problem(problem: dict[str, Any]) -> dict[str, Any]:
    recognition = recognize_nonlinear_programming(problem)
    gate_md = _recognition_markdown(recognition)
    if not recognition["can_solve"]:
        return {
            "status": "needs_clarification",
            "solver": "nonlinear_recognition_gate",
            "recognition": recognition,
            "missing_data": recognition["missing_information"],
            "markdown_report": gate_md + "\nChưa đủ dữ liệu để lập KKT/giải NLP và kết luận nghiệm tối ưu.\n",
        }
    if problem.get("problem_type") == "circle_packing_box" or problem.get("radii"):
        result = solve_circle_packing_box(problem.get("radii", []))
    elif problem.get("nlp"):
        nlp = dict(problem["nlp"])
        result = solve_nonlinear(nlp)
    else:
        result = solve_nonlinear(problem)
    verification = verify_nlp_solution(recognition, result)
    result["recognition"] = recognition
    result["verification"] = verification
    result["markdown_report"] = gate_md + "\n" + result.get("markdown_report", "") + _verification_markdown(verification)
    return result


def verify_nlp_solution(recognition: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    checks: list[str] = []
    passed = result.get("status") in {"optimal", "optimal_local"}
    if result.get("solution") or result.get("box"):
        checks.append("Candidate solution returned by NLP solver.")
    if result.get("constraint_checks"):
        bad = [item for item in result["constraint_checks"] if float(item.get("slack", 0)) < -1e-5]
        if bad:
            passed = False
            checks.append(f"{len(bad)} constraints violate feasibility tolerance.")
        else:
            checks.append("All reported nonlinear constraints satisfy feasibility tolerance.")
    if recognition["convexity_status"] in {"convex", "concave"}:
        checks.append(f"Convexity/concavity status: {recognition['convexity_status']}.")
    else:
        checks.append("Global optimality not guaranteed; report as local/KKT candidate unless separately proven.")
    return {"passed": passed, "checks": checks}


def _convexity_status(objective: Any) -> str:
    if isinstance(objective, dict) and objective.get("type") == "quadratic" and objective.get("Q") is not None:
        try:
            eig = np.linalg.eigvals(np.asarray(objective["Q"], dtype=float))
            if all(float(v.real) >= -1e-9 for v in eig):
                return "convex"
            if all(float(v.real) <= 1e-9 for v in eig):
                return "concave"
            return "nonconvex"
        except Exception:
            return "unknown"
    return "unknown"


def _domain_constraints(problem: dict[str, Any], spec: dict[str, Any]) -> list[Any]:
    domains = []
    if spec.get("bounds") or problem.get("bounds"):
        domains.append({"bounds": spec.get("bounds") or problem.get("bounds")})
    for var in problem.get("variables", []):
        domains.append({"variable": var.get("name"), "lower_bound": var.get("lower_bound"), "upper_bound": var.get("upper_bound")})
    return domains


def _recognition_markdown(r: dict[str, Any]) -> str:
    evidence = "\n".join(f"- {item}" for item in r["evidence"]) or "- Chưa có dấu hiệu đủ mạnh."
    missing = "\n".join(f"- {item}" for item in r["missing_information"]) or "- Không thiếu dữ liệu bắt buộc."
    return (
        "# Lời giải\n\n"
        "## 1. Nhận dạng dạng toán\n\n"
        f"- Dạng toán chính: {r['problem_type']}\n"
        f"- Dạng toán phụ: {r['subtype']}\n"
        f"- Vì sao phi tuyến: {', '.join(r['nonlinear_elements']) or 'chưa rõ'}\n"
        "- Biến liên tục: giả định continuous nếu không nêu integer/binary\n"
        f"- Objective: {r['objective']['direction']} {r['objective']['expression']}\n"
        f"- Method: {r['solution_method']}\n"
        f"- Mức tin cậy: {r['confidence']:.2f}\n\n"
        "Evidence:\n"
        f"{evidence}\n\n"
        "Missing information:\n"
        f"{missing}\n\n"
        "## 3. Mô hình toán học\n\n"
        f"- Decision variables: {', '.join(r['decision_variables'])}\n"
        f"- Objective function: `{r['objective']['expression']}`\n"
        f"- Constraints: `{r['constraints']}`\n"
        f"- Domain constraints: `{r['domain_constraints']}`\n\n"
        "## 4. Chuẩn hóa bài toán\n\n"
        "Nếu bài là maximize và dùng KKT chuẩn, chuyển thành minimize `-f(x)` với ràng buộc `g_i(x) <= 0`.\n\n"
        "## 6. Kuhn-Tucker / KKT conditions\n\n"
        "- Stationarity: `∇f_min(x*) + Σ λ_i ∇g_i(x*) + Σ μ_j ∇h_j(x*) = 0`\n"
        "- Primal feasibility: mọi `g_i(x*) <= 0`, `h_j(x*) = 0`\n"
        "- Dual feasibility: `λ_i >= 0`\n"
        "- Complementary slackness: `λ_i g_i(x*) = 0`\n\n"
    )


def _verification_markdown(v: dict[str, Any]) -> str:
    checks = "\n".join(f"- {item}" for item in v.get("checks", []))
    return f"\n## 8. Kiểm tra nghiệm\n\n- Trạng thái kiểm tra: {'passed' if v.get('passed') else 'failed'}\n{checks}\n"
