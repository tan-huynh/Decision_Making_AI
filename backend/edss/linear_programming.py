"""Linear Programming Solver — scipy.optimize.linprog + pure-Python fallback.

Uses HiGHS solver via scipy for production-quality LP. Falls back to
vertex enumeration for small problems when scipy is unavailable.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

# Try scipy first; fall back to pure-Python vertex enumeration
try:
    from scipy.optimize import linprog as scipy_linprog  # type: ignore[import-untyped]

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


EPS = 1e-8


def solve_lp(problem: dict[str, Any]) -> dict[str, Any]:
    """Solve an LP problem. Prefers scipy HiGHS; falls back to vertex enum."""
    variables = problem.get("variables", [])
    if not variables:
        raise ValueError("LP requires decision variables.")
    names = [item["name"] for item in variables]

    if HAS_SCIPY:
        return _solve_with_scipy(problem, names, variables)
    return _solve_vertex_enum(problem, names, variables)


# ── scipy HiGHS Solver ────────────────────────────────────────────────────


def _solve_with_scipy(
    problem: dict[str, Any],
    names: list[str],
    variables: list[dict[str, Any]],
    _skip_shadow: bool = False,
) -> dict[str, Any]:
    n = len(names)
    obj = problem.get("objective") or {}
    sense = obj.get("sense", "maximize")
    constant = float(obj.get("constant", 0))

    # Objective coefficients (linprog minimizes, so negate for maximize)
    c = [float(obj.get("coefficients", {}).get(name, 0)) for name in names]
    if sense == "maximize":
        c_min = [-ci for ci in c]
    else:
        c_min = c[:]

    # Constraints: separate <= and = types
    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    a_eq: list[list[float]] = []
    b_eq: list[float] = []
    constraint_names: list[str] = []
    constraint_types: list[str] = []

    for con in problem.get("constraints", []):
        coeffs = [float(con.get("coefficients", {}).get(name, 0)) for name in names]
        rhs = float(con.get("rhs", 0))
        op = con.get("operator", "<=")
        cname = con.get("name", "")

        if op == "<=":
            a_ub.append(coeffs)
            b_ub.append(rhs)
        elif op == ">=":
            a_ub.append([-ci for ci in coeffs])
            b_ub.append(-rhs)
        elif op == "=":
            a_eq.append(coeffs)
            b_eq.append(rhs)
        constraint_names.append(cname)
        constraint_types.append(op)

    # Variable bounds
    bounds = []
    for var in variables:
        lb = float(var.get("lower_bound", 0))
        ub = var.get("upper_bound")
        bounds.append((lb, float(ub) if ub is not None else None))

    # Solve
    result = scipy_linprog(
        c_min,
        A_ub=a_ub if a_ub else None,
        b_ub=b_ub if b_ub else None,
        A_eq=a_eq if a_eq else None,
        b_eq=b_eq if b_eq else None,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        status = "infeasible" if "infeasible" in result.message.lower() else "unbounded" if "unbounded" in result.message.lower() else "failed"
        return {"status": status, "message": result.message, "missing_data": []}

    # Extract solution
    x = list(result.x)
    obj_value = -result.fun + constant if sense == "maximize" else result.fun + constant

    # Compute slacks and binding constraints
    slacks: dict[str, float] = {}
    binding: list[str] = []
    shadow_prices: dict[str, float] = {}

    ub_idx = 0
    eq_idx = 0
    for i, con in enumerate(problem.get("constraints", [])):
        cname = con.get("name", f"c{i}")
        coeffs = [float(con.get("coefficients", {}).get(name, 0)) for name in names]
        rhs = float(con.get("rhs", 0))
        op = con.get("operator", "<=")
        lhs = sum(a * xi for a, xi in zip(coeffs, x))

        if op == "<=":
            slack = rhs - lhs
            slacks[cname] = round(slack, 6)
            if abs(slack) <= 1e-5:
                binding.append(cname)
            # Dual from ineq_marginals
            if hasattr(result, "ineq_lin") and result.ineq_lin is not None:
                try:
                    dual = float(result.ineq_lin.marginals[ub_idx])
                    shadow_prices[cname] = round(-dual if sense == "maximize" else dual, 6)
                except (IndexError, AttributeError, TypeError):
                    pass
            ub_idx += 1
        elif op == ">=":
            slack = lhs - rhs
            slacks[cname] = round(slack, 6)
            if abs(slack) <= 1e-5:
                binding.append(cname)
            if hasattr(result, "ineq_lin") and result.ineq_lin is not None:
                try:
                    dual = float(result.ineq_lin.marginals[ub_idx])
                    shadow_prices[cname] = round(dual if sense == "maximize" else -dual, 6)
                except (IndexError, AttributeError, TypeError):
                    pass
            ub_idx += 1
        elif op == "=":
            slacks[cname] = 0.0
            binding.append(cname)
            if hasattr(result, "eqlin") and result.eqlin is not None:
                try:
                    dual = float(result.eqlin.marginals[eq_idx])
                    shadow_prices[cname] = round(-dual if sense == "maximize" else dual, 6)
                except (IndexError, AttributeError, TypeError):
                    pass
            eq_idx += 1

    # Fallback shadow prices via finite-difference if duals unavailable
    if not shadow_prices and not _skip_shadow:
        shadow_prices = _approximate_shadow_prices_scipy(problem, names, variables, obj_value, sense)

    return {
        "status": "optimal",
        "solver": "scipy_highs",
        "objective_value": round(obj_value, 6),
        "solution": {name: round(x[i], 6) for i, name in enumerate(names)},
        "binding_constraints": binding,
        "slacks": slacks,
        "shadow_prices": shadow_prices,
        "assumptions": [
            "LP solved with scipy HiGHS interior-point/simplex solver.",
            "Shadow prices indicate marginal value of relaxing each binding constraint by 1 unit.",
        ],
        "recommendation": f"Nghiệm tối ưu có objective = {round(obj_value, 3)}.",
    }


def _approximate_shadow_prices_scipy(
    problem: dict[str, Any],
    names: list[str],
    variables: list[dict[str, Any]],
    base_value: float,
    sense: str,
    delta: float = 1.0,
) -> dict[str, float]:
    """Finite-difference shadow prices as fallback."""
    import copy
    prices: dict[str, float] = {}
    for i, con in enumerate(problem.get("constraints", [])):
        perturbed = copy.deepcopy(problem)
        perturbed["constraints"][i]["rhs"] = float(perturbed["constraints"][i].get("rhs", 0)) + delta
        try:
            result = _solve_with_scipy(perturbed, names, variables, _skip_shadow=True)
            if result.get("status") == "optimal":
                prices[con.get("name", f"c{i}")] = round(
                    (result["objective_value"] - base_value) / delta, 4
                )
        except Exception:
            prices[con.get("name", f"c{i}")] = 0.0
    return prices


# ── Pure-Python Vertex Enumeration (fallback) ─────────────────────────────


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < EPS:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        div = a[col][col]
        a[col] = [value / div for value in a[col]]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            a[row] = [a[row][i] - factor * a[col][i] for i in range(n + 1)]
    return [a[row][n] for row in range(n)]


def canonical_constraints(problem: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    constraints = []
    for item in problem.get("constraints", []):
        coeffs = [float(item.get("coefficients", {}).get(name, 0)) for name in names]
        op = item.get("operator", "<=")
        rhs = float(item.get("rhs", 0))
        if op == ">=":
            coeffs = [-value for value in coeffs]
            rhs = -rhs
        constraints.append({**item, "coeff_vector": coeffs, "rhs": rhs, "operator": "<="})
    for index, variable in enumerate(problem.get("variables", [])):
        lb = float(variable.get("lower_bound", 0))
        coeffs = [0.0] * len(names)
        coeffs[index] = -1.0
        constraints.append({"name": f"{names[index]}_lower_bound", "coeff_vector": coeffs, "rhs": -lb, "operator": "<="})
        if variable.get("upper_bound") is not None:
            coeffs = [0.0] * len(names)
            coeffs[index] = 1.0
            constraints.append({"name": f"{names[index]}_upper_bound", "coeff_vector": coeffs, "rhs": float(variable["upper_bound"]), "operator": "<="})
    return constraints


def feasible(point: list[float], constraints: list[dict[str, Any]]) -> bool:
    return all(sum(a * x for a, x in zip(item["coeff_vector"], point)) <= float(item["rhs"]) + 1e-6 for item in constraints)


def objective_value(problem: dict[str, Any], names: list[str], point: list[float]) -> float:
    objective = problem.get("objective") or {}
    coeffs = objective.get("coefficients", {})
    return float(objective.get("constant", 0)) + sum(float(coeffs.get(name, 0)) * point[index] for index, name in enumerate(names))


def _solve_vertex_enum(
    problem: dict[str, Any],
    names: list[str],
    variables: list[dict[str, Any]],
) -> dict[str, Any]:
    n = len(names)
    constraints = canonical_constraints(problem, names)
    if len(constraints) < n:
        raise ValueError("LP requires at least as many constraints as variables.")

    candidates = []
    for combo in combinations(constraints, n):
        solution = solve_linear_system([item["coeff_vector"] for item in combo], [float(item["rhs"]) for item in combo])
        if solution is not None and feasible(solution, constraints):
            candidates.append(solution)
    if not candidates:
        return {"status": "infeasible", "message": "No feasible vertex found.", "missing_data": []}

    sense = (problem.get("objective") or {}).get("sense", "maximize")
    reverse = sense == "maximize"
    best = sorted(candidates, key=lambda point: objective_value(problem, names, point), reverse=reverse)[0]
    best_value = objective_value(problem, names, best)
    slacks: dict[str, float] = {}
    binding: list[str] = []
    for item in constraints:
        lhs = sum(a * x for a, x in zip(item["coeff_vector"], best))
        slack = float(item["rhs"]) - lhs
        slacks[item["name"]] = slack
        if abs(slack) <= 1e-5:
            binding.append(item["name"])

    shadow_prices = _approximate_shadow_prices_vertex(problem, names, best_value)
    return {
        "status": "optimal",
        "solver": "pure_python_vertex_lp",
        "objective_value": best_value,
        "solution": {name: best[index] for index, name in enumerate(names)},
        "binding_constraints": binding,
        "slacks": slacks,
        "shadow_prices": shadow_prices,
        "assumptions": [
            "LP solved via vertex enumeration (pure-Python fallback). Install scipy for HiGHS.",
            "Shadow prices are finite-difference approximations.",
        ],
        "recommendation": f"Nghiệm tối ưu có objective = {best_value:.3f}.",
    }


def _approximate_shadow_prices_vertex(problem: dict[str, Any], names: list[str], base_value: float) -> dict[str, float]:
    prices = {}
    for index, constraint in enumerate(problem.get("constraints", [])):
        import copy
        perturbed = copy.deepcopy(problem)
        perturbed["constraints"][index]["rhs"] = float(perturbed["constraints"][index].get("rhs", 0)) + 1.0
        try:
            result = _solve_vertex_enum(perturbed, names, problem.get("variables", []))
            if result.get("status") == "optimal":
                prices[constraint.get("name", f"c{index}")] = result["objective_value"] - base_value
        except Exception:
            prices[constraint.get("name", f"c{index}")] = 0.0
    return prices
