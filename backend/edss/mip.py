from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    HAS_MILP = True
except Exception:
    HAS_MILP = False

from .teaching_report import wrap_teaching_report


def solve_mip(problem: dict[str, Any]) -> dict[str, Any]:
    c = np.asarray(problem["c"], dtype=float)
    sense = problem.get("sense", "minimize")
    c_solver = -c if sense == "maximize" else c
    bounds_raw = problem.get("bounds") or [(0, None)] * len(c)
    lb = np.array([0 if b[0] is None else float(b[0]) for b in bounds_raw], dtype=float)
    ub = np.array([np.inf if len(b) < 2 or b[1] is None else float(b[1]) for b in bounds_raw], dtype=float)
    integrality = np.array(problem.get("integrality", [1] * len(c)), dtype=int)
    names = problem.get("variable_names", [f"x{i+1}" for i in range(len(c))])
    if HAS_MILP:
        constraints = []
        if problem.get("A_ub"):
            constraints.append(LinearConstraint(np.asarray(problem["A_ub"], dtype=float), -np.inf, np.asarray(problem["b_ub"], dtype=float)))
        if problem.get("A_eq"):
            rhs = np.asarray(problem["b_eq"], dtype=float)
            constraints.append(LinearConstraint(np.asarray(problem["A_eq"], dtype=float), rhs, rhs))
        result = milp(c=c_solver, integrality=integrality, bounds=Bounds(lb, ub), constraints=constraints or None)
        if not result.success:
            return {"status": "failed", "solver": "scipy_milp", "message": result.message}
        x = result.x
    else:
        x = _enumerate_small_mip(c_solver, problem, lb, ub, integrality)
        if x is None:
            return {"status": "unsupported", "message": "scipy.milp unavailable and enumeration too large."}
    objective = float(c @ x)
    rows = [{"variable": name, "value": round(float(value), 8), "type": _integrality_name(integrality[i])} for i, (name, value) in enumerate(zip(names, x))]
    report = wrap_teaching_report(
        "Integer / Mixed-Integer Programming",
        "MIP",
        [
            ("Mô hình", "Biến integer/binary được giữ rời rạc; solver dùng branch-and-bound/HiGHS integrality nếu có."),
            ("Nghiệm", _table(rows)),
        ],
        f"Objective = {objective:.6f}",
    )
    return {
        "status": "optimal",
        "solver": "scipy_milp" if HAS_MILP else "enumeration_mip",
        "objective_value": round(objective, 8),
        "solution": {name: round(float(value), 8) for name, value in zip(names, x)},
        "markdown_report": report,
    }


def _enumerate_small_mip(c_solver: np.ndarray, problem: dict[str, Any], lb, ub, integrality) -> np.ndarray | None:
    domains = []
    for lo, hi, integ in zip(lb, ub, integrality):
        if not np.isfinite(hi) or hi - lo > 25 or integ == 0:
            return None
        domains.append(range(int(lo), int(hi) + 1))
    best_x = None
    best_value = float("inf")
    a_ub = np.asarray(problem.get("A_ub", []), dtype=float)
    b_ub = np.asarray(problem.get("b_ub", []), dtype=float)
    a_eq = np.asarray(problem.get("A_eq", []), dtype=float)
    b_eq = np.asarray(problem.get("b_eq", []), dtype=float)
    for candidate in product(*domains):
        x = np.asarray(candidate, dtype=float)
        if a_ub.size and np.any(a_ub @ x - b_ub > 1e-8):
            continue
        if a_eq.size and np.any(np.abs(a_eq @ x - b_eq) > 1e-8):
            continue
        value = float(c_solver @ x)
        if value < best_value:
            best_value = value
            best_x = x
    return best_x


def _integrality_name(code: int) -> str:
    return {0: "continuous", 1: "integer", 2: "semi-continuous", 3: "semi-integer"}.get(int(code), "integer")


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    keys = list(rows[0].keys())
    out = ["| " + " | ".join(keys) + " |", "|" + "|".join("---" for _ in keys) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row[k]) for k in keys) + " |")
    return "\n".join(out)
