"""Value of Information Engine — EVPI, EVI, max price for information.

Implements exact calculations for:
- EVPI (Expected Value of Perfect Information)
- EVI (Expected Value of Imperfect Information) via Bayesian update
- Decision change detection
- Max price worth paying for additional information
"""

from __future__ import annotations

from typing import Any


def compute_evpi(
    alternatives: list[str],
    states: list[dict[str, Any]],
    payoff_lookup: dict[tuple[str, str], float],
) -> dict[str, Any]:
    """Compute Expected Value of Perfect Information.

    EVPI = E[max payoff per state] - max E[payoff per alternative]
    """
    # Normalize probabilities
    probs = _normalize([float(s.get("probability", 0)) for s in states])
    state_names = [s["name"] for s in states]

    # EV without information: best expected value
    ev_per_alt: dict[str, float] = {}
    for alt in alternatives:
        ev = sum(
            probs[j] * payoff_lookup.get((alt, state_names[j]), 0.0)
            for j in range(len(states))
        )
        ev_per_alt[alt] = ev

    best_alt = max(ev_per_alt, key=ev_per_alt.get)  # type: ignore[arg-type]
    ev_without = ev_per_alt[best_alt]

    # EV with perfect information: for each state, pick best alternative
    ev_with_pi = 0.0
    best_per_state: dict[str, dict[str, Any]] = {}
    for j, state in enumerate(states):
        sn = state["name"]
        best_payoff = float("-inf")
        best_action = ""
        for alt in alternatives:
            payoff = payoff_lookup.get((alt, sn), 0.0)
            if payoff > best_payoff:
                best_payoff = payoff
                best_action = alt
        ev_with_pi += probs[j] * best_payoff
        best_per_state[sn] = {"best_action": best_action, "payoff": best_payoff}

    evpi = max(0.0, ev_with_pi - ev_without)

    return {
        "EVwoPI": round(ev_without, 4),
        "EVwPI": round(ev_with_pi, 4),
        "EVPI": round(evpi, 4),
        "best_alternative_without_info": best_alt,
        "ev_per_alternative": {k: round(v, 4) for k, v in ev_per_alt.items()},
        "best_per_state": best_per_state,
        "recommendation": _evpi_recommendation(evpi, ev_without),
    }


def compute_evi(
    alternatives: list[str],
    states: list[dict[str, Any]],
    payoff_lookup: dict[tuple[str, str], float],
    test_sensitivity: dict[str, dict[str, float]],
    test_cost: float = 0.0,
) -> dict[str, Any]:
    """Compute Expected Value of Imperfect Information.

    Parameters
    ----------
    test_sensitivity : dict
        Maps state_name -> {test_result_name: P(result|state)}.
        Example: {"Oil": {"Positive": 0.9, "Negative": 0.1},
                  "No Oil": {"Positive": 0.3, "Negative": 0.7}}
    test_cost : float
        Cost of acquiring the information (survey, test, etc.)
    """
    probs = _normalize([float(s.get("probability", 0)) for s in states])
    state_names = [s["name"] for s in states]

    # EV without information
    ev_per_alt: dict[str, float] = {}
    for alt in alternatives:
        ev_per_alt[alt] = sum(
            probs[j] * payoff_lookup.get((alt, state_names[j]), 0.0)
            for j in range(len(states))
        )
    ev_without = max(ev_per_alt.values())

    # Collect all possible test results
    all_results: set[str] = set()
    for state_results in test_sensitivity.values():
        all_results.update(state_results.keys())

    # For each test result, compute posterior and best action
    ev_with_test = 0.0
    test_analysis: list[dict[str, Any]] = []
    for result in sorted(all_results):
        # P(result) = sum_j P(result|state_j) * P(state_j)
        p_result = sum(
            test_sensitivity.get(state_names[j], {}).get(result, 0) * probs[j]
            for j in range(len(states))
        )
        if p_result < 1e-12:
            continue

        # P(state_j | result) = P(result|state_j) * P(state_j) / P(result)
        posteriors = []
        for j in range(len(states)):
            likelihood = test_sensitivity.get(state_names[j], {}).get(result, 0)
            posteriors.append(likelihood * probs[j] / p_result)

        # Best action given this test result
        best_action = ""
        best_ev = float("-inf")
        for alt in alternatives:
            ev = sum(
                posteriors[j] * payoff_lookup.get((alt, state_names[j]), 0.0)
                for j in range(len(states))
            )
            if ev > best_ev:
                best_ev = ev
                best_action = alt

        ev_with_test += p_result * best_ev
        test_analysis.append({
            "test_result": result,
            "P_result": round(p_result, 4),
            "posteriors": {state_names[j]: round(posteriors[j], 4) for j in range(len(states))},
            "best_action": best_action,
            "conditional_EV": round(best_ev, 4),
        })

    evi = max(0.0, ev_with_test - ev_without)
    net_evi = evi - test_cost

    return {
        "EVwoI": round(ev_without, 4),
        "EVwI": round(ev_with_test, 4),
        "EVI": round(evi, 4),
        "test_cost": test_cost,
        "net_EVI": round(net_evi, 4),
        "max_price_for_info": round(evi, 4),
        "should_buy_info": net_evi > 0,
        "test_analysis": test_analysis,
        "decision_changes": any(
            t["best_action"] != test_analysis[0]["best_action"]
            for t in test_analysis
        ) if test_analysis else False,
        "recommendation": _evi_recommendation(evi, test_cost, net_evi),
    }


def compute_voi_from_problem(problem: dict[str, Any]) -> dict[str, Any]:
    """Convenience wrapper that extracts EVPI from a standard EDSSProblem dict."""
    alternatives = [a["name"] for a in problem.get("alternatives", [])]
    states = problem.get("states", [])
    matrix = problem.get("payoff_matrix", [])

    lookup: dict[tuple[str, str], float] = {}
    for cell in matrix:
        key = (cell["alternative"], cell["state"])
        lookup[key] = float(cell.get("payoff", 0)) - float(cell.get("cost", 0))

    result = compute_evpi(alternatives, states, lookup)

    # If test info is available, also compute EVI
    test_info = problem.get("information_test")
    if test_info:
        evi_result = compute_evi(
            alternatives, states, lookup,
            test_sensitivity=test_info.get("sensitivity", {}),
            test_cost=float(test_info.get("cost", 0)),
        )
        result["evi"] = evi_result

    return result


# ── Helpers ───────────────────────────────────────────────────────────────


def _normalize(probs: list[float]) -> list[float]:
    total = sum(max(0, p) for p in probs)
    if total <= 0:
        return [1 / max(1, len(probs))] * len(probs)
    return [max(0, p) / total for p in probs]


def _evpi_recommendation(evpi: float, ev_without: float) -> str:
    if evpi < 1e-6:
        return "EVPI = 0: thông tin hoàn hảo không thay đổi quyết định. Hành động ngay."
    ratio = evpi / abs(ev_without) if ev_without else 1.0
    if ratio > 0.3:
        return (
            f"EVPI = {evpi:.2f} (>{ratio:.0%} của EV). "
            "Giá trị thông tin rất cao — nên đầu tư thu thập dữ liệu trước khi quyết định."
        )
    if ratio > 0.1:
        return (
            f"EVPI = {evpi:.2f}. Thông tin có giá trị đáng kể — "
            "cân nhắc khảo sát/thử nghiệm nếu chi phí thấp hơn EVPI."
        )
    return f"EVPI = {evpi:.2f}. Giá trị thông tin thấp — có thể quyết định dựa trên dữ liệu hiện có."


def _evi_recommendation(evi: float, cost: float, net: float) -> str:
    if net > 0:
        return (
            f"EVI = {evi:.2f} > chi phí thông tin {cost:.2f}. "
            "Nên mua/thu thập thông tin trước khi quyết định."
        )
    if evi > 0:
        return (
            f"EVI = {evi:.2f} nhưng chi phí {cost:.2f} quá cao. "
            "Thông tin có giá trị nhưng không đáng mua với giá này."
        )
    return "Thông tin bổ sung không thay đổi quyết định tối ưu."
