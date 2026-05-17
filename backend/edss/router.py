"""Solver Router — Classifies problems and dispatches to the correct solver.

Integrates all solver modules, model validation, sensitivity analysis,
VOI computation, and audit trail into a unified pipeline.
"""

from __future__ import annotations

from typing import Any

from .assignment import solve_assignment
from .audit import create_audit_trail, log_step
from .classifier import classify_problem, missing_data_questions
from .dynamic_programming import solve_finite_horizon_dp, solve_resource_allocation_dp
from .linear_programming import solve_lp
from .model_validator import validate_model
from .multiobjective import pareto_frontier, weighted_score
from .network import solve_shortest_path, solve_transportation, solve_max_flow, solve_min_cost_flow
from .risk import risk_from_simulation, value_at_risk
from .sensitivity_engine import sensitivity_analysis
from .uncertainty import expected_payoff, rollback_decision_tree, simulate_payoffs, solve_binary_event_tree, value_of_information
from .voi_engine import compute_voi_from_problem


def build_mathematical_model(problem: dict[str, Any]) -> dict[str, Any]:
    """Generate a mathematical formulation from structured problem data."""
    kind = problem.get("problem_type") or classify_problem(problem.get("context", {}).get("description", ""), problem)["problem_type"]
    model: dict[str, Any] = {"problem_type": kind, "formulation": "", "assumptions": list(problem.get("assumptions", []))}

    if kind == "linear_programming":
        variables = [item["name"] for item in problem.get("variables", [])]
        obj = problem.get("objective") or {}
        terms = [f"{coef}*{name}" for name, coef in obj.get("coefficients", {}).items()]
        model["formulation"] = (
            f"{obj.get('sense', 'maximize')} Z = {' + '.join(terms) or obj.get('expression', '')}\n"
            + "\n".join(
                f"{c['name']}: "
                + " + ".join(f"{coef}*{name}" for name, coef in c.get("coefficients", {}).items())
                + f" {c.get('operator', '<=')} {c.get('rhs')}"
                for c in problem.get("constraints", [])
            )
            + "\n"
            + ", ".join(f"{name} >= 0" for name in variables)
        )
    elif kind == "transportation":
        model["formulation"] = (
            "minimize Z = Σ_i Σ_j c_ij x_ij\n"
            "subject to:\n"
            "  Σ_j x_ij = supply_i  ∀ source i\n"
            "  Σ_i x_ij = demand_j  ∀ destination j\n"
            "  x_ij ≥ 0"
        )
    elif kind == "assignment":
        model["formulation"] = (
            "minimize Z = Σ_i Σ_j c_ij x_ij\n"
            "subject to:\n"
            "  Σ_j x_ij = 1  ∀ agent i\n"
            "  Σ_i x_ij = 1  ∀ task j\n"
            "  x_ij ∈ {0,1}"
        )
    elif kind == "decision_tree":
        model["formulation"] = (
            "EV(action_i) = Σ_j P(state_j) × payoff(i, j)\n"
            "Rollback: outcome → chance EV → decision max.\n"
            "EVPI = E[max payoff per state] - max E[payoff per action]"
        )
    elif kind == "dynamic_programming":
        model["formulation"] = (
            "F_k(s) = max_x { reward_k(s,x) + F_{k+1}(T(s,x)) }\n"
            "Bellman backward induction over stages and states."
        )
    elif kind == "multi_objective":
        model["formulation"] = "Score(a)=Σ_k w_k × f_k(a); Pareto frontier for non-dominated."
    else:
        model["formulation"] = "Model builder needs structured variables, constraints, graph, stages, or payoff matrix."

    model["missing_data"] = missing_data_questions(problem)
    return model


def solve_problem(problem: dict[str, Any]) -> dict[str, Any]:
    """Main entry point: classify → validate → solve → sensitivity → VOI."""
    trail = create_audit_trail(problem.get("problem_id"))

    # 1. Classify
    kind = problem.get("problem_type") or classify_problem(
        problem.get("context", {}).get("description", ""), problem
    )["problem_type"]
    log_step(trail, "classification", kind)

    # 2. Validate
    validation = validate_model({**problem, "problem_type": kind})
    log_step(trail, "model_validation", validation)

    # 3. Solve
    result: dict[str, Any]
    if kind == "linear_programming":
        result = solve_lp(problem)
    elif kind == "assignment":
        result = solve_assignment(problem.get("assignment_costs", []), maximize=False)
    elif kind == "shortest_path":
        result = solve_shortest_path(problem.get("graph", {}))
    elif kind == "transportation":
        result = solve_transportation(problem)
    elif kind == "max_flow":
        result = solve_max_flow(problem.get("graph", {}))
    elif kind in ("min_cost_flow", "transshipment", "network_flow"):
        result = solve_min_cost_flow(problem.get("graph", {}))
    elif kind == "dynamic_programming":
        if problem.get("resource_allocation"):
            spec = problem["resource_allocation"]
            result = solve_resource_allocation_dp(
                int(spec["total_resource"]),
                spec["stage_returns"],
                spec.get("resource_name", "resource"),
            )
        else:
            result = solve_finite_horizon_dp(problem)
    elif kind == "decision_tree":
        if problem.get("probability_tree"):
            result = solve_binary_event_tree(problem)
        else:
            result = rollback_decision_tree(problem) if problem.get("decision_tree") else expected_payoff(problem)
        # Add VOI if payoff matrix available
        if problem.get("payoff_matrix"):
            try:
                result["voi"] = compute_voi_from_problem(problem)
            except Exception:
                result["voi"] = value_of_information(problem)
    elif kind == "simulation_risk":
        simulation = simulate_payoffs(problem)
        risk = risk_from_simulation(simulation)
        result = {"status": "computed", "simulation": simulation, "risk": risk}
    elif kind == "multi_objective":
        weighted = weighted_score(problem)
        result = {**weighted, "pareto": pareto_frontier(problem)}
    else:
        result = {"status": "needs_clarification", "missing_data": missing_data_questions(problem)}

    log_step(trail, "solver_execution", result.get("status"))

    # 4. Sensitivity analysis (for LP and decision_tree)
    sensitivity: dict[str, Any] = {}
    if kind == "linear_programming" and result.get("status") == "optimal":
        try:
            sensitivity = sensitivity_analysis(problem, result, solve_fn=solve_lp)
            log_step(trail, "sensitivity_analysis", "completed")
        except Exception:
            sensitivity = {"error": "Sensitivity analysis failed."}
    elif kind in ("decision_tree", "expected_value") and result.get("status") == "computed":
        try:
            sensitivity = sensitivity_analysis(problem, result)
            log_step(trail, "sensitivity_analysis", "completed")
        except Exception:
            sensitivity = {}

    # 5. Risk analysis for simulation
    risk_result: dict[str, Any] = {}
    if kind == "simulation_risk":
        log_step(trail, "risk_analysis", "from simulation")
        risk_result = result.get("risk", {})

    output = {
        "problem_type": kind,
        "model": build_mathematical_model(problem),
        "validation": validation,
        "result": result,
        "sensitivity": sensitivity,
        "risk": risk_result,
        "recommendation_explanation": explain_recommendation(kind, result),
        "audit_steps": len(trail["steps"]),
    }
    return output


def explain_recommendation(kind: str, result: dict[str, Any]) -> str:
    """Generate human-readable recommendation explanation."""
    if result.get("status") == "needs_clarification":
        return "Không đủ dữ liệu để khuyến nghị; cần bổ sung dữ liệu thiếu trước khi chạy solver."
    if kind == "linear_programming" and result.get("status") == "optimal":
        solver = result.get("solver", "LP")
        return (
            f"Nghiệm tối ưu (solver: {solver}) có objective = {result['objective_value']:.3f}. "
            f"Binding constraints: {', '.join(result.get('binding_constraints', [])) or 'không có'}. "
            "Shadow price cho biết giá trị cận biên khi tăng thêm tài nguyên."
        )
    if kind == "transportation" and result.get("status") in ("optimal", "feasible_heuristic"):
        return (
            f"Tổng chi phí vận chuyển tối thiểu = {result.get('objective_value', 0):.0f}. "
            "Kiểm tra reduced costs để xác nhận tối ưu."
        )
    if kind == "assignment":
        return f"Assignment tối ưu có tổng chi phí = {result.get('objective_value', 0):.3f}."
    if kind == "shortest_path":
        path = result.get("path", [])
        return f"Đường đi ngắn nhất: {' → '.join(path)}, chi phí = {result.get('objective_value', 0):.0f}."
    if kind in {"decision_tree", "simulation_risk"}:
        return (
            "Khuyến nghị dựa trên expected value/risk distribution. "
            "Lưu ý: quyết định tốt ≠ outcome tốt. Kết quả may mắn không chứng minh quyết định đúng."
        )
    if kind == "dynamic_programming":
        return (
            "Bellman recursion tối ưu toàn cục qua các giai đoạn. "
            "Greedy/myopic có thể sai vì không xét tương lai."
        )
    if kind == "multi_objective":
        return "Weighted score ranking; kiểm tra Pareto frontier để tránh chọn phương án bị dominated."
    return "Khuyến nghị được sinh từ solver phù hợp với loại bài toán."
