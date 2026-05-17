"""Sensitivity Analysis Engine — Parametric analysis and what-if scenarios.

Provides:
- Shadow price calculation (exact from dual, or finite-difference)
- Objective coefficient ranging
- RHS ranging
- Tornado chart data
- What-if scenario runner
"""

from __future__ import annotations

from typing import Any


def sensitivity_analysis(
    problem: dict[str, Any],
    solver_result: dict[str, Any],
    solve_fn: Any = None,
) -> dict[str, Any]:
    """Run comprehensive sensitivity analysis on a solved problem.

    Parameters
    ----------
    problem : dict
        Original problem definition.
    solver_result : dict
        Result from the solver (must include objective_value, solution).
    solve_fn : callable, optional
        Function to re-solve perturbed problems. If None, uses finite-difference.
    """
    kind = problem.get("problem_type", "linear_programming")
    base_obj = float(solver_result.get("objective_value", 0))

    result: dict[str, Any] = {
        "base_objective": base_obj,
        "problem_type": kind,
    }

    if kind == "linear_programming":
        result["shadow_prices"] = solver_result.get("shadow_prices", {})
        result["slacks"] = solver_result.get("slacks", {})
        result["binding_constraints"] = solver_result.get("binding_constraints", [])
        result["rhs_sensitivity"] = _rhs_sensitivity(problem, base_obj, solve_fn)
        result["objective_sensitivity"] = _objective_coeff_sensitivity(problem, base_obj, solve_fn)
        result["tornado"] = _build_tornado(result)

    elif kind in ("decision_tree", "expected_value"):
        result["probability_sensitivity"] = _probability_sensitivity(problem)

    elif kind == "transportation":
        result["reduced_costs"] = solver_result.get("optimality_certificate", {}).get("reduced_costs", {})
        result["shadow_prices"] = {
            **(solver_result.get("optimality_certificate", {}).get("u", {})),
            **(solver_result.get("optimality_certificate", {}).get("v", {})),
        }

    return result


def what_if_scenario(
    problem: dict[str, Any],
    changes: dict[str, Any],
    solve_fn: Any,
) -> dict[str, Any]:
    """Run a what-if scenario by modifying problem parameters and re-solving.

    Parameters
    ----------
    changes : dict
        Keys like "constraints.labor_hours.rhs": 120, or
        "objective.coefficients.x_A": 50.
    solve_fn : callable
        Function that takes a problem dict and returns solver result.
    """
    import copy
    modified = copy.deepcopy(problem)

    applied: list[str] = []
    for path, new_value in changes.items():
        parts = path.split(".")
        try:
            _set_nested(modified, parts, new_value)
            applied.append(f"{path} = {new_value}")
        except (KeyError, IndexError, TypeError):
            pass

    try:
        new_result = solve_fn(modified)
        return {
            "status": "computed",
            "changes_applied": applied,
            "original_objective": float(problem.get("objective", {}).get("constant", 0)),
            "new_result": new_result,
        }
    except Exception as exc:
        return {"status": "error", "changes_applied": applied, "error": str(exc)}


def _rhs_sensitivity(
    problem: dict[str, Any],
    base_obj: float,
    solve_fn: Any,
    delta: float = 1.0,
) -> list[dict[str, Any]]:
    """Finite-difference sensitivity on constraint RHS values."""
    if solve_fn is None:
        return []

    import copy
    results: list[dict[str, Any]] = []
    for i, con in enumerate(problem.get("constraints", [])):
        name = con.get("name", f"c{i}")
        rhs = float(con.get("rhs", 0))

        # Increase RHS by delta
        modified = copy.deepcopy(problem)
        modified["constraints"][i]["rhs"] = rhs + delta
        try:
            res_up = solve_fn(modified)
            obj_up = float(res_up.get("objective_value", base_obj))
        except Exception:
            obj_up = base_obj

        # Decrease RHS by delta
        modified = copy.deepcopy(problem)
        modified["constraints"][i]["rhs"] = rhs - delta
        try:
            res_down = solve_fn(modified)
            obj_down = float(res_down.get("objective_value", base_obj))
        except Exception:
            obj_down = base_obj

        shadow = (obj_up - base_obj) / delta if delta else 0
        results.append({
            "constraint": name,
            "current_rhs": rhs,
            "shadow_price": round(shadow, 4),
            "obj_if_increase": round(obj_up, 4),
            "obj_if_decrease": round(obj_down, 4),
            "impact": round(abs(obj_up - obj_down), 4),
        })

    results.sort(key=lambda r: r["impact"], reverse=True)
    return results


def _objective_coeff_sensitivity(
    problem: dict[str, Any],
    base_obj: float,
    solve_fn: Any,
    delta_pct: float = 0.10,
) -> list[dict[str, Any]]:
    """Sensitivity on objective function coefficients (±10% by default)."""
    if solve_fn is None:
        return []

    import copy
    obj = problem.get("objective", {})
    coeffs = obj.get("coefficients", {})
    results: list[dict[str, Any]] = []

    for var_name, coeff in coeffs.items():
        c = float(coeff)
        delta = max(abs(c * delta_pct), 0.01)

        modified = copy.deepcopy(problem)
        modified["objective"]["coefficients"][var_name] = c + delta
        try:
            res_up = solve_fn(modified)
            obj_up = float(res_up.get("objective_value", base_obj))
        except Exception:
            obj_up = base_obj

        modified = copy.deepcopy(problem)
        modified["objective"]["coefficients"][var_name] = c - delta
        try:
            res_down = solve_fn(modified)
            obj_down = float(res_down.get("objective_value", base_obj))
        except Exception:
            obj_down = base_obj

        results.append({
            "variable": var_name,
            "current_coefficient": c,
            "delta": round(delta, 4),
            "obj_if_increase": round(obj_up, 4),
            "obj_if_decrease": round(obj_down, 4),
            "impact": round(abs(obj_up - obj_down), 4),
        })

    results.sort(key=lambda r: r["impact"], reverse=True)
    return results


def _probability_sensitivity(problem: dict[str, Any]) -> list[dict[str, Any]]:
    """For decision under uncertainty: sensitivity to probability changes."""
    from .uncertainty import normalize_states, payoff_lookup

    states = problem.get("states", [])
    alts = [a["name"] for a in problem.get("alternatives", [])]
    lookup = payoff_lookup(problem.get("payoff_matrix", []))
    norm_states = normalize_states(states)

    # Base expected values
    base_ev: dict[str, float] = {}
    for alt in alts:
        base_ev[alt] = sum(
            s["probability"] * lookup.get((alt, s["name"]), 0)
            for s in norm_states
        )
    base_best = max(base_ev, key=base_ev.get)  # type: ignore[arg-type]

    results: list[dict[str, Any]] = []
    for j, state in enumerate(norm_states):
        # Find breakeven probability where decision changes
        sn = state["name"]
        p_orig = state["probability"]

        # Try sweeping probability from 0 to 1
        decision_changes_at = None
        for p_test in [i / 100 for i in range(101)]:
            # Redistribute other probabilities proportionally
            remaining = 1.0 - p_test
            other_total = sum(s["probability"] for k, s in enumerate(norm_states) if k != j)

            test_ev: dict[str, float] = {}
            for alt in alts:
                ev = p_test * lookup.get((alt, sn), 0)
                for k, s2 in enumerate(norm_states):
                    if k == j:
                        continue
                    p_adj = (s2["probability"] / other_total * remaining) if other_total > 0 else 0
                    ev += p_adj * lookup.get((alt, s2["name"]), 0)
                test_ev[alt] = ev

            test_best = max(test_ev, key=test_ev.get)  # type: ignore[arg-type]
            if test_best != base_best and decision_changes_at is None:
                decision_changes_at = p_test

        results.append({
            "state": sn,
            "current_probability": round(p_orig, 4),
            "decision_changes_at": round(decision_changes_at, 4) if decision_changes_at is not None else None,
            "is_sensitive": decision_changes_at is not None,
        })

    return results


def _build_tornado(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Build tornado chart data from RHS and coefficient sensitivity."""
    tornado: list[dict[str, Any]] = []
    for item in result.get("rhs_sensitivity", []):
        tornado.append({
            "parameter": f"RHS: {item['constraint']}",
            "low": item["obj_if_decrease"],
            "high": item["obj_if_increase"],
            "impact": item["impact"],
        })
    for item in result.get("objective_sensitivity", []):
        tornado.append({
            "parameter": f"Coeff: {item['variable']}",
            "low": item["obj_if_decrease"],
            "high": item["obj_if_increase"],
            "impact": item["impact"],
        })
    tornado.sort(key=lambda t: t["impact"], reverse=True)
    return tornado


def _set_nested(obj: Any, keys: list[str], value: Any) -> None:
    """Set a nested dict value by key path."""
    for key in keys[:-1]:
        if isinstance(obj, list):
            obj = obj[int(key)]
        elif isinstance(obj, dict):
            obj = obj[key]
        else:
            raise TypeError
    final_key = keys[-1]
    if isinstance(obj, dict):
        obj[final_key] = value
    elif isinstance(obj, list):
        obj[int(final_key)] = value
