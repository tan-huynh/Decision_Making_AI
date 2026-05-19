"""Model Validator — Pre-solve validation of decision problem models."""

from __future__ import annotations
from typing import Any


def validate_model(problem: dict[str, Any]) -> dict[str, Any]:
    """Run all validators and return aggregated report."""
    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []
    kind = problem.get("problem_type", "")

    _check_probabilities(problem, errors, warnings)
    _check_variables(problem, errors, warnings)
    _check_constraints(problem, errors, warnings, info)
    _check_objective(problem, kind, errors, warnings)
    _check_data_completeness(problem, kind, errors, warnings, info)

    if kind == "linear_programming":
        _validate_lp(problem, errors, warnings, info)
    elif kind == "transportation":
        _validate_transportation(problem, errors, warnings, info)
    elif kind == "assignment":
        _validate_assignment(problem, errors, warnings, info)
    elif kind in ("shortest_path", "max_flow", "min_cost_flow", "network_flow"):
        _validate_network(problem, kind, errors, warnings, info)
    elif kind == "dynamic_programming":
        _validate_dynamic_programming(problem, errors, warnings, info)
    elif kind in ("decision_tree", "expected_value"):
        _validate_decision_uncertainty(problem, errors, warnings)

    return {"is_valid": len(errors) == 0, "errors": errors, "warnings": warnings, "info": info, "problem_type": kind}


def _check_probabilities(problem: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    if problem.get("probability_tree"):
        spec = problem["probability_tree"]
        p = float(spec.get("success_probability", 0))
        if p < 0 or p > 1:
            errors.append(f"Xác suất success_probability phải nằm trong [0,1], hiện là {p}.")
        if int(spec.get("trials", 0) or 0) < 1:
            errors.append("Probability tree cần số lần thử trials >= 1.")
    if problem.get("bayes"):
        spec = problem["bayes"]
        for key in ("prior", "sensitivity", "false_positive_rate"):
            value = float(spec.get(key, 0))
            if value < 0 or value > 1:
                errors.append(f"Bayes parameter '{key}' phải nằm trong [0,1], hiện là {value}.")
    if problem.get("independent_probabilities"):
        for i, p in enumerate(problem["independent_probabilities"]):
            value = float(p)
            if value < 0 or value > 1:
                errors.append(f"Xác suất biến cố độc lập thứ {i + 1} phải nằm trong [0,1], hiện là {value}.")

    states = problem.get("states", [])
    if not states:
        return
    probs = [float(s.get("probability", 0)) for s in states]
    for i, p in enumerate(probs):
        if p < 0:
            errors.append(f"Xác suất trạng thái '{states[i].get('name', i)}' âm ({p}).")
        if p > 1:
            errors.append(f"Xác suất trạng thái '{states[i].get('name', i)}' > 1 ({p}).")
    total = sum(probs)
    if abs(total - 1.0) > 0.05:
        warnings.append(f"Tổng xác suất = {total:.4f}, khác 1.0 đáng kể.")
    elif abs(total - 1.0) > 0.001:
        warnings.append(f"Tổng xác suất = {total:.4f}, sẽ được chuẩn hóa về 1.0.")
    if len(probs) > 1 and all(abs(p - probs[0]) < 1e-8 for p in probs):
        warnings.append("Tất cả trạng thái có xác suất bằng nhau — có thể là giả định chưa xác nhận.")


def _check_variables(problem: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    names_seen: set[str] = set()
    for var in problem.get("variables", []):
        name = var.get("name", "")
        if not name:
            errors.append("Biến quyết định thiếu tên.")
            continue
        if name in names_seen:
            errors.append(f"Biến '{name}' bị trùng tên.")
        names_seen.add(name)
        lb, ub = var.get("lower_bound"), var.get("upper_bound")
        if lb is not None and ub is not None and float(lb) > float(ub):
            errors.append(f"Biến '{name}': lower_bound ({lb}) > upper_bound ({ub}).")


def _check_constraints(problem: dict[str, Any], errors: list[str], warnings: list[str], info: list[str]) -> None:
    var_names = {v.get("name") for v in problem.get("variables", [])}
    for i, con in enumerate(problem.get("constraints", [])):
        name = con.get("name", f"c{i}")
        coeffs = con.get("coefficients", {})
        if var_names:
            for vn in coeffs:
                if vn not in var_names:
                    warnings.append(f"Ràng buộc '{name}' tham chiếu biến '{vn}' không tồn tại.")
        op = con.get("operator", "<=")
        if op not in ("<=", ">=", "="):
            errors.append(f"Ràng buộc '{name}': operator '{op}' không hợp lệ.")


def _check_objective(problem: dict[str, Any], kind: str, errors: list[str], warnings: list[str]) -> None:
    obj = problem.get("objective")
    if not obj and kind == "linear_programming":
        errors.append("LP cần hàm mục tiêu (objective).")
    if obj and obj.get("sense") not in ("maximize", "minimize", None):
        errors.append(f"Objective sense '{obj.get('sense')}' không hợp lệ.")


def _check_data_completeness(problem: dict[str, Any], kind: str, errors: list[str], warnings: list[str], info: list[str]) -> None:
    ctx = problem.get("context", {})
    if not ctx.get("unit"):
        warnings.append("Chưa xác định đơn vị đo hàm mục tiêu.")
    if not problem.get("assumptions"):
        info.append("Chưa liệt kê giả định; khuyến nghị thêm assumptions.")


def _validate_lp(problem: dict[str, Any], errors: list[str], warnings: list[str], info: list[str]) -> None:
    if not problem.get("variables"):
        errors.append("LP cần ít nhất một biến quyết định.")
    if not problem.get("constraints"):
        warnings.append("LP không có ràng buộc; nghiệm có thể unbounded.")
    if len(problem.get("variables", [])) > 20:
        info.append("LP có nhiều biến; nên dùng scipy/OR-Tools thay vì vertex enumeration.")


def _validate_transportation(problem: dict[str, Any], errors: list[str], warnings: list[str], info: list[str]) -> None:
    graph = problem.get("graph", {})
    supplies, demands = graph.get("supplies", {}), graph.get("demands", {})
    if not supplies:
        errors.append("Transportation cần dữ liệu supply.")
    if not demands:
        errors.append("Transportation cần dữ liệu demand.")
    if supplies and demands:
        ts, td = sum(float(v) for v in supplies.values()), sum(float(v) for v in demands.values())
        if abs(ts - td) > 1e-6:
            warnings.append(f"Tổng cung ({ts}) ≠ tổng cầu ({td}). Sẽ thêm dummy.")


def _validate_assignment(problem: dict[str, Any], errors: list[str], warnings: list[str], info: list[str]) -> None:
    costs = problem.get("assignment_costs", [])
    if not costs:
        errors.append("Assignment cần ma trận chi phí.")
        return
    rows, cols = len(costs), len(costs[0]) if costs else 0
    if rows != cols:
        info.append(f"Ma trận không vuông ({rows}×{cols}); sẽ pad dummy.")


def _validate_network(problem: dict[str, Any], kind: str, errors: list[str], warnings: list[str], info: list[str]) -> None:
    graph = problem.get("graph", {})
    edges = graph.get("edges", [])
    if not edges:
        errors.append("Network problem cần danh sách cạnh.")
        return
    for edge in edges:
        if kind == "shortest_path" and float(edge.get("cost", 0)) < 0:
            errors.append("Shortest path bằng Dijkstra không hỗ trợ edge cost âm.")
        if kind in ("max_flow", "min_cost_flow") and float(edge.get("capacity", 0)) < 0:
            errors.append("Network flow không hỗ trợ capacity âm.")
    if kind == "max_flow":
        if not graph.get("source"):
            errors.append("Max flow cần source.")
        if not (graph.get("sink") or graph.get("target")):
            errors.append("Max flow cần sink/target.")
    if kind == "min_cost_flow":
        nodes = graph.get("nodes", [])
        if not nodes:
            errors.append("Min-cost flow cần nodes có supply/demand.")
        balance = sum(float(n.get("supply", 0)) - float(n.get("demand", 0)) for n in nodes)
        if abs(balance) > 1e-8:
            warnings.append(f"Tổng supply-demand = {balance:.4f}; min-cost flow có thể infeasible nếu không thêm dummy.")


def _validate_dynamic_programming(problem: dict[str, Any], errors: list[str], warnings: list[str], info: list[str]) -> None:
    spec = problem.get("resource_allocation")
    if not spec:
        return
    total = int(spec.get("total_resource", -1))
    rows = spec.get("stage_returns", [])
    if total < 0:
        errors.append("DP resource allocation cần total_resource >= 0.")
    if not rows:
        errors.append("DP resource allocation cần stage_returns.")
    for idx, row in enumerate(rows):
        if len(row) <= total:
            warnings.append(f"Stage {idx + 1} chỉ có {len(row)} mức, có thể không đủ để xét total_resource={total}.")


def _validate_decision_uncertainty(problem: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    if problem.get("probability_tree") or problem.get("bayes") or problem.get("diagnostic_decision") or problem.get("forklift_decision") or problem.get("independent_probabilities"):
        return
    if not problem.get("alternatives"):
        errors.append("Cần ít nhất 2 alternatives.")
    if not problem.get("states"):
        errors.append("Cần ít nhất 2 states of nature.")
    if not problem.get("payoff_matrix"):
        errors.append("Cần payoff matrix.")
