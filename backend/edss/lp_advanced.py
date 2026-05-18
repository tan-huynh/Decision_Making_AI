from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from .linear_programming import solve_lp_raw
from .teaching_report import wrap_teaching_report

EPS = 1e-9


def simplex_step_by_step(problem: dict[str, Any]) -> dict[str, Any]:
    """Primal simplex tableau for max c^T x, A x <= b, x >= 0."""
    c = np.asarray(problem["c"], dtype=float)
    a = np.asarray(problem.get("A_ub", []), dtype=float)
    b = np.asarray(problem.get("b_ub", []), dtype=float)
    sense = problem.get("sense", "maximize")
    if sense != "maximize":
        return {"status": "unsupported", "message": "Step-by-step simplex hiện hỗ trợ dạng maximize với <= và b >= 0."}
    if a.ndim != 2 or len(b) != a.shape[0] or a.shape[1] != len(c):
        raise ValueError("Cần c, A_ub, b_ub đúng kích thước.")
    if np.any(b < -EPS):
        return {"status": "needs_two_phase", "message": "Có RHS âm; cần two-phase/Big-M."}
    m, n = a.shape
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:m, :n] = a
    tableau[:m, n:n + m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[-1, :n] = -c
    basis = [n + i for i in range(m)]
    var_names = problem.get("variable_names", [f"x{i+1}" for i in range(n)]) + [f"s{i+1}" for i in range(m)]
    steps: list[dict[str, Any]] = [_snapshot(tableau, basis, var_names, "Initial tableau")]
    status = "optimal"
    for iteration in range(1, 100):
        obj = tableau[-1, :-1]
        entering = int(np.argmin(obj))
        if obj[entering] >= -EPS:
            break
        column = tableau[:m, entering]
        if np.all(column <= EPS):
            status = "unbounded"
            break
        ratios = [tableau[i, -1] / column[i] if column[i] > EPS else float("inf") for i in range(m)]
        leaving_row = int(np.argmin(ratios))
        pivot = tableau[leaving_row, entering]
        tableau[leaving_row, :] /= pivot
        for i in range(m + 1):
            if i != leaving_row:
                tableau[i, :] -= tableau[i, entering] * tableau[leaving_row, :]
        basis[leaving_row] = entering
        steps.append(_snapshot(tableau, basis, var_names, f"Iteration {iteration}: enter {var_names[entering]}, leave row {leaving_row + 1}"))
    values = {name: 0.0 for name in var_names}
    for row, basic_idx in enumerate(basis):
        values[var_names[basic_idx]] = float(tableau[row, -1])
    solution = {name: round(values[name], 8) for name in var_names[:n]}
    objective_value = float(tableau[-1, -1])
    alternate = any(abs(v) <= 1e-8 for j, v in enumerate(tableau[-1, :-1]) if j not in basis)
    degeneracy = any(abs(tableau[i, -1]) <= 1e-8 for i in range(m))
    sections = [
        ("Dạng chuẩn", "Thêm biến slack `s_i` để đưa `A x <= b` thành `A x + s = b`, với `x,s >= 0`."),
        ("Simplex tableau từng bước", "\n\n".join(_tableau_markdown(step) for step in steps)),
        ("Phát hiện trạng thái", f"Status = `{status}`. Degeneracy = `{degeneracy}`. Alternate optimum = `{alternate}`."),
    ]
    report = wrap_teaching_report(
        problem.get("title", "Linear Programming Simplex"),
        "Linear Programming Step-by-Step Simplex",
        sections,
        f"Z* = {objective_value:.6f}; nghiệm chính: {solution}",
        [f"Reduced costs hàng objective đều không âm: {status == 'optimal'}"],
    )
    return {
        "status": status,
        "solver": "primal_simplex_tableau",
        "objective_value": round(objective_value, 8),
        "solution": solution,
        "basis": [var_names[i] for i in basis],
        "degeneracy": degeneracy,
        "alternate_optimum": alternate,
        "steps": steps,
        "markdown_report": report,
    }


def duality_analysis(problem: dict[str, Any]) -> dict[str, Any]:
    c = np.asarray(problem["c"], dtype=float)
    a = np.asarray(problem.get("A_ub", []), dtype=float)
    b = np.asarray(problem.get("b_ub", []), dtype=float)
    if problem.get("sense", "maximize") != "maximize" or a.size == 0:
        return {"status": "unsupported", "message": "Duality module hiện hỗ trợ primal max với Ax<=b, x>=0."}
    primal = solve_lp_raw(problem)
    dual_problem = {
        "sense": "minimize",
        "c": b.tolist(),
        "A_ub": (-a.T).tolist(),
        "b_ub": (-c).tolist(),
        "bounds": [(0, None)] * len(b),
        "variable_names": [f"y{i+1}" for i in range(len(b))],
        "constraint_names_ub": [f"dual_c{j+1}" for j in range(len(c))],
        "title": "Dual LP",
    }
    dual = solve_lp_raw(dual_problem)
    y = dual.get("all_values", {})
    x = primal.get("all_values", {})
    var_names = problem.get("variable_names", [f"x{i+1}" for i in range(len(c))])
    comp_rows = []
    for i in range(len(b)):
        lhs = float(a[i] @ np.array([x.get(name, 0.0) for name in var_names]))
        slack = float(b[i] - lhs)
        yi = float(y.get(f"y{i+1}", 0.0))
        comp_rows.append({"constraint": f"c{i+1}", "slack": round(slack, 8), "dual": round(yi, 8), "product": round(slack * yi, 8)})
    sections = [
        ("Bài toán đối ngẫu", _dual_formulation_markdown(a, b, c)),
        ("Complementary slackness", _rows_table(comp_rows)),
        ("Shadow price", "Biến đối ngẫu `y_i` là giá trị biên của RHS ràng buộc `i` trong vùng nhạy cục bộ."),
    ]
    return {
        "status": "computed",
        "primal": primal,
        "dual_problem": dual_problem,
        "dual": dual,
        "complementary_slackness": comp_rows,
        "markdown_report": wrap_teaching_report(
            "Duality Analysis",
            "LP Duality + Complementary Slackness",
            sections,
            f"Primal objective = {primal.get('objective_value')}; Dual objective = {dual.get('objective_value')}.",
        ),
    }


def sensitivity_ranges(problem: dict[str, Any]) -> dict[str, Any]:
    base = solve_lp_raw(problem)
    if base.get("status") != "optimal":
        return {"status": "not_optimal", "base": base}
    rhs_ranges = []
    a_ub = problem.get("A_ub", [])
    b_ub = [float(v) for v in problem.get("b_ub", [])]
    for i, rhs in enumerate(b_ub):
        low, high = _rhs_allowable_range(problem, i, rhs)
        rhs_ranges.append({"constraint": problem.get("constraint_names_ub", [])[i] if i < len(problem.get("constraint_names_ub", [])) else f"c{i+1}", "base_rhs": rhs, "allowable_min": low, "allowable_max": high})
    sections = [
        ("Kết quả gốc", f"Z* = {base.get('objective_value')}; nghiệm = {base.get('all_values')}; shadow prices = {base.get('shadow_prices', {})}."),
        ("RHS sensitivity", _rows_table(rhs_ranges)),
        ("Objective coefficient sensitivity", _rows_table(_objective_coefficient_ranges(problem))),
        ("Lưu ý", "Khoảng RHS/objective được ước lượng bằng parametric re-solve. Với bài toán production lớn, nên dùng basis sensitivity chính xác từ solver thương mại."),
    ]
    return {
        "status": "computed",
        "base": base,
        "rhs_ranges": rhs_ranges,
        "objective_coefficient_ranges": _objective_coefficient_ranges(problem),
        "reduced_costs": _reduced_costs(problem, base),
        "markdown_report": wrap_teaching_report("Sensitivity Analysis", "LP Sensitivity", sections, "Nghiệm tối ưu giữ nguyên trong các khoảng RHS ước lượng ở bảng trên."),
    }


def _objective_coefficient_ranges(problem: dict[str, Any]) -> list[dict[str, Any]]:
    base = solve_lp_raw(problem)
    if base.get("status") != "optimal":
        return []
    names = problem.get("variable_names", [f"x{i+1}" for i in range(len(problem["c"]))])
    base_vec = tuple(round(base.get("all_values", {}).get(name, 0.0), 6) for name in names)
    ranges = []
    for idx, coef in enumerate([float(v) for v in problem["c"]]):
        def same_solution(new_coef: float) -> bool:
            test = {**problem, "c": list(problem["c"])}
            test["c"][idx] = new_coef
            result = solve_lp_raw(test)
            vec = tuple(round(result.get("all_values", {}).get(name, 0.0), 6) for name in names)
            return result.get("status") == "optimal" and vec == base_vec

        low = coef
        step = max(1.0, abs(coef) * 0.1)
        for _ in range(30):
            if same_solution(low - step):
                low -= step
                step *= 1.5
            else:
                break
        high = coef
        step = max(1.0, abs(coef) * 0.1)
        for _ in range(30):
            if same_solution(high + step):
                high += step
                step *= 1.5
            else:
                break
        ranges.append({"variable": names[idx], "base_coefficient": coef, "allowable_min": round(low, 6), "allowable_max": round(high, 6)})
    return ranges


def _rhs_allowable_range(problem: dict[str, Any], index: int, base_rhs: float) -> tuple[float, float]:
    base_solution = solve_lp_raw(problem).get("all_values", {})
    names = problem.get("variable_names", [f"x{i+1}" for i in range(len(problem["c"]))])
    base_vec = tuple(round(base_solution.get(name, 0.0), 6) for name in names)

    def same_basis(rhs_value: float) -> bool:
        test = {**problem, "b_ub": list(problem.get("b_ub", []))}
        test["b_ub"][index] = rhs_value
        result = solve_lp_raw(test)
        vec = tuple(round(result.get("all_values", {}).get(name, 0.0), 6) for name in names)
        return result.get("status") == "optimal" and vec == base_vec

    low = base_rhs
    step = max(1.0, abs(base_rhs) * 0.1)
    while low - step >= 0 and same_basis(low - step):
        low -= step
        step *= 1.5
    high = base_rhs
    step = max(1.0, abs(base_rhs) * 0.1)
    for _ in range(30):
        if same_basis(high + step):
            high += step
            step *= 1.5
        else:
            break
    return round(low, 6), round(high, 6)


def _reduced_costs(problem: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    c = np.asarray(problem["c"], dtype=float)
    a = np.asarray(problem.get("A_ub", []), dtype=float)
    y_values = base.get("shadow_prices", {})
    if a.size == 0 or not y_values:
        return {}
    y = np.array([float(y_values.get(name, 0.0)) for name in problem.get("constraint_names_ub", [f"ub_{i}" for i in range(a.shape[0])])])
    reduced = c - y @ a
    names = problem.get("variable_names", [f"x{i+1}" for i in range(len(c))])
    return {name: round(float(value), 8) for name, value in zip(names, reduced)}


def _snapshot(tableau: np.ndarray, basis: list[int], var_names: list[str], label: str) -> dict[str, Any]:
    return {"label": label, "basis": [var_names[i] for i in basis], "tableau": np.round(tableau, 8).tolist()}


def _tableau_markdown(step: dict[str, Any]) -> str:
    rows = step["tableau"]
    headers = [f"v{j+1}" for j in range(len(rows[0]) - 1)] + ["RHS"]
    lines = [f"**{step['label']}**", "", "| Basis | " + " | ".join(headers) + " |", "|---|" + "|".join("---:" for _ in headers) + "|"]
    for i, row in enumerate(rows):
        basis = step["basis"][i] if i < len(step["basis"]) else "Z"
        lines.append("| " + basis + " | " + " | ".join(f"{v:.4g}" for v in row) + " |")
    return "\n".join(lines)


def _dual_formulation_markdown(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> str:
    lines = ["Primal max `c^T x` với `Ax <= b, x >= 0` có dual:", "", "```text", "min b^T y", "s.t. A^T y >= c", "y >= 0", "```", ""]
    lines.append(f"`b = {b.tolist()}`, `c = {c.tolist()}`.")
    return "\n".join(lines)


def _rows_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_Không có dữ liệu._"
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join("---" for _ in keys) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines)
