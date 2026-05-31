"""Solver Router — Classifies problems and dispatches to the correct solver.

Integrates all solver modules, model validation, sensitivity analysis,
VOI computation, and audit trail into a unified pipeline.
"""

from __future__ import annotations

from typing import Any

from .assignment import solve_assignment
from .audit import create_audit_trail, log_step
from .classifier import classify_problem, missing_data_questions
from .decision_theory import recognize_decision_theory, solve_decision_theory_problem
from .dp_solver import recognize_dynamic_programming, solve_dynamic_programming_problem
from .inventory_solver import recognize_inventory_theory, solve_inventory_theory_problem
from .integer_solver import recognize_integer_programming, solve_integer_programming_problem
from .linear_programming import solve_lp
from .linear_solver import recognize_linear_programming, solve_linear_programming_problem
from .markov_solver import recognize_markov_processes, solve_markov_processes_problem
from .mermaid_visualization import attach_mermaid_if_needed
from .game_theory import solve_game_theory
from .model_validator import validate_model
from .multiobjective import pareto_frontier, weighted_score
from .network import solve_shortest_path, solve_transportation, solve_max_flow, solve_min_cost_flow
from .network_solver import NETWORK_TYPES, recognize_network_modelling, solve_network_modelling_problem
from .nonlinear_solver import recognize_nonlinear_programming, solve_nonlinear_programming_problem
from .queueing_solver import recognize_queueing_theory, solve_queueing_problem
from .risk import risk_from_simulation, value_at_risk
from .sensitivity_engine import sensitivity_analysis
from .uncertainty import expected_payoff, rollback_decision_tree, simulate_payoffs, solve_bayes_problem, solve_binary_event_tree, solve_diagnostic_decision_tree, solve_forklift_decision_tree, solve_imperfect_information_decision_tree, solve_independent_probability, value_of_information
from .voi_engine import compute_voi_from_problem
from .or_pipeline import classify_with_taxonomy, recognition_gate



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
        if problem.get("probability_tree"):
            model["formulation"] = "Binary event tree: P(path)=Π branch probabilities; P(at least one success)=1-(1-p)^n."
        elif problem.get("bayes"):
            model["formulation"] = "Bayes: P(H|E)=P(E|H)P(H)/[P(E|H)P(H)+P(E|¬H)P(¬H)]."
        elif problem.get("diagnostic_decision"):
            model["formulation"] = "Diagnostic decision tree: prior → imperfect test → posterior probabilities → optional follow-up information → rollback expected payoff."
        elif problem.get("imperfect_information_decision"):
            model["formulation"] = "Imperfect sample information: prior states → pilot signal likelihoods → Bayes posteriors → choose best action for each signal → rollback EMV and EVSI."
        elif problem.get("forklift_decision"):
            model["formulation"] = "Forklift decision tree: choose new/used/test; use Bayes posteriors for test results; rollback minimum expected cost and information value."
        elif problem.get("independent_probabilities"):
            model["formulation"] = "Independent events: P(all)=Πp_i; P(at least one)=1-Π(1-p_i)."
        else:
            model["formulation"] = (
                "EV(action_i) = Σ_j P(state_j) × payoff(i, j)\n"
                "Rollback: outcome → chance EV → decision max.\n"
                "EVPI = E[max payoff per state] - max E[payoff per action]"
            )
    elif kind == "dynamic_programming":
        model["formulation"] = (
            "Recognition gate: identify stages, states, decisions, transition, reward/cost, boundary.\n"
            "Max: f_n(s)=max_x { r_n(s,x)+f_{n+1}(T_n(s,x)) }\n"
            "Min: f_n(s)=min_x { c_n(s,x)+f_{n+1}(T_n(s,x)) }\n"
            "Solve by Bellman recursion, DP tables, traceback, and verification."
        )
    elif kind == "integer_programming":
        model["formulation"] = (
            "Integer Programming recognition gate → identify integer/binary variables, objective, constraints, "
            "and logical links → formulate MILP with integrality restrictions → solve by branch-and-bound/MILP "
            "and verify integer feasibility. Do not round LP relaxation as an optimal IP solution."
        )
    elif kind == "multi_objective":
        model["formulation"] = "Score(a)=Σ_k w_k × f_k(a); Pareto frontier for non-dominated."
    elif kind == "game_theory":
        model["formulation"] = (
            "Game Theory recognition gate → payoff matrix → maximin/minimax saddle-point check → "
            "dominance reduction → mixed strategy by 2x2 algebra, graph method, or LP."
        )
    elif kind == "queueing_theory":
        model["formulation"] = (
            "Queueing recognition gate → normalize λ and μ → identify M/M/1, M/M/s, finite capacity, "
            "finite source, or network → check ρ < 1 → compute P0, Lq, L, Wq, W and cost if required."
        )
    elif kind in ("inventory", "inventory_theory"):
        model["formulation"] = (
            "Inventory recognition gate → normalize annual demand D, ordering cost K, holding cost h, "
            "unit cost c, lead time, discount/backorder flags → choose EOQ, reorder point, quantity discount, "
            "or planned shortage formula → verify costs and feasibility."
        )
    elif kind == "nonlinear_programming":
        model["formulation"] = (
            "NLP recognition gate → identify continuous variables, nonlinear objective/constraints, domain, "
            "convexity, and method → formulate Lagrangian/KKT or solve numerically with feasibility/globality checks."
        )
    elif kind == "markov_processes":
        model["formulation"] = (
            "Markov recognition gate → identify states, one-step transition matrix P, time step, initial distribution "
            "and requested output → validate row-stochastic probabilities → compute P^n, stationary distribution, "
            "absorbing metrics, first passage time, or long-run mean cost."
        )
    else:
        model["formulation"] = "Model builder needs structured variables, constraints, graph, stages, or payoff matrix."

    model["missing_data"] = missing_data_questions(problem)
    return model


def solve_problem(problem: dict[str, Any]) -> dict[str, Any]:
    """Main entry point: classify → validate → solve → sensitivity → VOI."""
    from .models import EDSSProblem
    
    # Phase 2: Pydantic Data Validation
    try:
        validated = EDSSProblem(**problem)
        # We keep the dict for downstream functions but ensure it's validated
        problem = validated.model_dump(exclude_none=True)
    except Exception as exc:
        raise ValueError(f"Dữ liệu bài toán không hợp lệ hoặc thiếu thông tin: {exc}")

    trail = create_audit_trail(problem.get("problem_id"))

    # 1. Classify & Gate
    classification = classify_with_taxonomy(problem.get("context", {}).get("description", ""), problem)
    kind = problem.get("problem_type") or classification["primary_type"]
    log_step(trail, "classification", kind)

    if kind == "dynamic_programming":
        dp_recognition = recognize_dynamic_programming(problem)
        if not dp_recognition["can_solve"]:
            dp_result = solve_dynamic_programming_problem(problem)
            return {
                "problem_type": kind,
                "status": "needs_clarification",
                "missing_data": dp_recognition["missing_information"],
                "model": build_mathematical_model(problem),
                "validation": {"is_valid": False, "errors": dp_recognition["missing_information"], "warnings": [], "info": []},
                "result": dp_result,
                "recognition_gate": dp_recognition,
                "recommendation_explanation": "Bài toán có dấu hiệu Dynamic Programming nhưng chưa đủ stage/state/decision/transition/reward để giải.",
                "audit_steps": len(trail["steps"]),
            }
    if kind == "linear_programming":
        lp_recognition = recognize_linear_programming(problem)
        if not lp_recognition["can_solve"]:
            lp_result = solve_linear_programming_problem(problem)
            return {
                "problem_type": kind,
                "status": lp_result.get("status", "needs_clarification"),
                "missing_data": lp_recognition["missing_information"],
                "model": build_mathematical_model(problem),
                "validation": {"is_valid": False, "errors": lp_recognition["missing_information"], "warnings": [], "info": []},
                "result": lp_result,
                "recognition_gate": lp_recognition,
                "recommendation_explanation": "Bài toán có dấu hiệu LP nhưng chưa đủ điều kiện LP: cần objective/constraints tuyến tính, biến continuous và bounds rõ ràng.",
                "audit_steps": len(trail["steps"]),
            }
    if kind == "decision_tree":
        dt_recognition = recognize_decision_theory(problem)
        if not dt_recognition["can_solve"]:
            dt_result = solve_decision_theory_problem(problem)
            return {
                "problem_type": kind,
                "status": "needs_clarification",
                "missing_data": dt_recognition["missing_information"],
                "model": build_mathematical_model(problem),
                "validation": {"is_valid": False, "errors": dt_recognition["missing_information"], "warnings": [], "info": []},
                "result": dt_result,
                "recognition_gate": dt_recognition,
                "recommendation_explanation": "Bài toán có dấu hiệu Decision Theory nhưng chưa đủ alternatives/states/probabilities/payoffs để kết luận.",
                "audit_steps": len(trail["steps"]),
            }
    if kind == "queueing_theory":
        qt_recognition = recognize_queueing_theory(problem)
        if not qt_recognition["can_solve"]:
            qt_result = solve_queueing_problem(problem)
            return {
                "problem_type": kind,
                "status": "needs_clarification",
                "missing_data": qt_recognition["missing_information"],
                "model": build_mathematical_model(problem),
                "validation": {"is_valid": False, "errors": qt_recognition["missing_information"], "warnings": [], "info": []},
                "result": qt_result,
                "recognition_gate": qt_recognition,
                "recommendation_explanation": "Bài toán có dấu hiệu Queueing Theory nhưng chưa đủ λ/μ/server/population/objective để giải.",
                "audit_steps": len(trail["steps"]),
            }
    if kind in ("inventory", "inventory_theory"):
        inv_recognition = recognize_inventory_theory(problem)
        if not inv_recognition["can_solve"]:
            inv_result = solve_inventory_theory_problem(problem)
            return {
                "problem_type": kind,
                "status": "needs_clarification",
                "missing_data": inv_recognition["missing_information"],
                "model": build_mathematical_model(problem),
                "validation": {"is_valid": False, "errors": inv_recognition["missing_information"], "warnings": [], "info": []},
                "result": inv_result,
                "recognition_gate": inv_recognition,
                "recommendation_explanation": "Bài toán có dấu hiệu Inventory Theory nhưng chưa đủ D/K/h/c/discount/backorder data để giải.",
                "audit_steps": len(trail["steps"]),
            }
    if kind in NETWORK_TYPES:
        net_recognition = recognize_network_modelling(problem)
        if not net_recognition["can_solve"]:
            net_result = solve_network_modelling_problem(problem)
            return {
                "problem_type": kind,
                "status": "needs_clarification",
                "missing_data": net_recognition["missing_information"],
                "model": build_mathematical_model(problem),
                "validation": {"is_valid": False, "errors": net_recognition["missing_information"], "warnings": [], "info": []},
                "result": net_result,
                "recognition_gate": net_recognition,
                "recommendation_explanation": "Bài toán có dấu hiệu Network Modelling nhưng chưa đủ nodes/arcs/weights/source/sink/capacity để giải.",
                "audit_steps": len(trail["steps"]),
            }
    if kind == "nonlinear_programming":
        nlp_recognition = recognize_nonlinear_programming(problem)
        if not nlp_recognition["can_solve"]:
            nlp_result = solve_nonlinear_programming_problem(problem)
            return {
                "problem_type": kind,
                "status": "needs_clarification",
                "missing_data": nlp_recognition["missing_information"],
                "model": build_mathematical_model(problem),
                "validation": {"is_valid": False, "errors": nlp_recognition["missing_information"], "warnings": [], "info": []},
                "result": nlp_result,
                "recognition_gate": nlp_recognition,
                "recommendation_explanation": "Bài toán có dấu hiệu Nonlinear Programming nhưng chưa đủ variables/objective/constraints/domain để giải.",
                "audit_steps": len(trail["steps"]),
            }
    if kind == "integer_programming":
        ip_recognition = recognize_integer_programming(problem)
        if not ip_recognition["can_solve"]:
            ip_result = solve_integer_programming_problem(problem)
            return {
                "problem_type": kind,
                "status": "needs_clarification",
                "missing_data": ip_recognition["missing_information"],
                "model": build_mathematical_model(problem),
                "validation": {"is_valid": False, "errors": ip_recognition["missing_information"], "warnings": [], "info": []},
                "result": ip_result,
                "recognition_gate": ip_recognition,
                "recommendation_explanation": "Bài toán có dấu hiệu Integer Programming nhưng chưa đủ variables/objective/constraints/integrality để giải.",
                "audit_steps": len(trail["steps"]),
            }
    if kind == "markov_processes":
        mc_recognition = recognize_markov_processes(problem)
        if not mc_recognition["can_solve"]:
            mc_result = solve_markov_processes_problem(problem)
            return {
                "problem_type": kind,
                "status": "needs_clarification",
                "missing_data": mc_recognition["missing_information"],
                "model": build_mathematical_model(problem),
                "validation": {"is_valid": False, "errors": mc_recognition["missing_information"], "warnings": [], "info": []},
                "result": mc_result,
                "recognition_gate": mc_recognition,
                "recommendation_explanation": "Bài toán có dấu hiệu Markov Processes nhưng chưa đủ states/P/time step/requested output để giải.",
                "audit_steps": len(trail["steps"]),
            }

    gate = recognition_gate(classification, problem)
    if gate["decision_to_solve"] == "ask_clarification":
        return {
            "problem_type": kind,
            "status": "needs_clarification",
            "missing_data": missing_data_questions(problem),
            "recognition_gate": gate,
            "audit_steps": len(trail["steps"]),
            "recommendation_explanation": "Bài toán chưa đủ dữ liệu hoặc mức độ tin cậy thấp. Vui lòng làm rõ các thông tin bị thiếu."
        }

    # 2. Validate
    validation = validate_model({**problem, "problem_type": kind})
    log_step(trail, "model_validation", validation)

    # 3. Solve
    result: dict[str, Any]
    if kind == "linear_programming":
        result = solve_linear_programming_problem(problem)
    elif kind == "assignment":
        result = solve_assignment(problem.get("assignment_costs", []), maximize=False)
    elif kind == "shortest_path":
        result = solve_network_modelling_problem(problem)
    elif kind == "transportation":
        result = solve_transportation(problem)
    elif kind == "max_flow":
        result = solve_network_modelling_problem(problem)
    elif kind in ("min_cost_flow", "transshipment", "network_flow"):
        result = solve_network_modelling_problem(problem)
    elif kind == "inventory":
        result = solve_inventory_theory_problem(problem)
    elif kind == "dynamic_programming":
        result = solve_dynamic_programming_problem(problem)
    elif kind == "decision_tree":
        result = solve_decision_theory_problem(problem)
    elif kind == "simulation_risk":
        simulation = simulate_payoffs(problem)
        risk = risk_from_simulation(simulation)
        result = {"status": "computed", "simulation": simulation, "risk": risk}
    elif kind == "multi_objective":
        weighted = weighted_score(problem)
        result = {**weighted, "pareto": pareto_frontier(problem)}
    elif kind == "game_theory":
        result = solve_game_theory(problem)
    elif kind == "queueing_theory":
        result = solve_queueing_problem(problem)
    elif kind == "network_modelling":
        result = solve_network_modelling_problem(problem)
    elif kind == "inventory_theory":
        result = solve_inventory_theory_problem(problem)
    elif kind == "nonlinear_programming":
        result = solve_nonlinear_programming_problem(problem)
    elif kind == "integer_programming":
        result = solve_integer_programming_problem(problem)
    elif kind == "markov_processes":
        result = solve_markov_processes_problem(problem)
    else:
        result = {"status": "needs_clarification", "missing_data": missing_data_questions(problem)}
    result = attach_mermaid_if_needed(problem, result, kind)

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
        "recognition_gate": gate,
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
    if kind == "inventory_theory" and result.get("status") in ("optimal", "computed"):
        return f"Chính sách Inventory: Đặt Q = {result.get('order_quantity', 0):.2f}. Tổng chi phí = {result.get('annual_inventory_cost') or result.get('total_relevant_cost', 0):.2f}."
    return "Khuyến nghị được sinh từ solver phù hợp với loại bài toán."
