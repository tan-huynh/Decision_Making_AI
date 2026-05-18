from __future__ import annotations

import math
from typing import Any


def expected_utility(outcomes: list[dict[str, float]]) -> dict[str, Any]:
    if not outcomes:
        raise ValueError("Cần danh sách outcomes.")
    probability_sum = sum(float(item["probability"]) for item in outcomes)
    if abs(probability_sum - 1) > 1e-6:
        raise ValueError("Tổng xác suất phải bằng 1.")
    eu = sum(float(item["probability"]) * float(item["utility"]) for item in outcomes)
    ev = sum(float(item["probability"]) * float(item.get("value", 0)) for item in outcomes)
    return {
        "status": "computed",
        "model": "expected_utility",
        "expected_utility": eu,
        "expected_value": ev,
        "markdown_report": (
            "### Expected Utility\n\n"
            f"EU = Σ p_i u(x_i) = {eu:.6f}.\n\n"
            f"Expected monetary value = {ev:.6f}."
        ),
    }


def exponential_certainty_equivalent(outcomes: list[dict[str, float]], risk_tolerance: float) -> dict[str, Any]:
    if risk_tolerance <= 0:
        raise ValueError("risk_tolerance phải dương.")
    if not outcomes:
        raise ValueError("Cần danh sách outcomes.")
    probability_sum = sum(float(item["probability"]) for item in outcomes)
    if abs(probability_sum - 1) > 1e-6:
        raise ValueError("Tổng xác suất phải bằng 1.")
    expected_value = sum(float(item["probability"]) * float(item["value"]) for item in outcomes)
    expected_exp = sum(float(item["probability"]) * math.exp(-float(item["value"]) / risk_tolerance) for item in outcomes)
    ce = -risk_tolerance * math.log(expected_exp)
    risk_premium = expected_value - ce
    return {
        "status": "computed",
        "model": "exponential_utility",
        "risk_attitude": "risk_averse",
        "risk_tolerance": risk_tolerance,
        "expected_value": expected_value,
        "certainty_equivalent": ce,
        "risk_premium": risk_premium,
        "markdown_report": (
            "### Certainty Equivalent\n\n"
            "Dùng utility mũ u(x)=1-exp(-x/R).\n\n"
            f"EV = {expected_value:.6f}, CE = {ce:.6f}, risk premium = {risk_premium:.6f}."
        ),
    }


def fit_exponential_risk_tolerance(certainty_equivalents: list[dict[str, float]]) -> dict[str, Any]:
    """Estimate risk tolerance from simple lottery CE answers.

    Each item: low, high, probability_high, certainty_equivalent.
    Grid-search R for u(x)=1-exp(-x/R).
    """
    if not certainty_equivalents:
        raise ValueError("Cần dữ liệu certainty equivalent.")
    candidates = [10 ** (-2 + i * 0.02) for i in range(401)]
    scale = max(abs(float(item.get("high", 1))) for item in certainty_equivalents)
    candidates = [r * max(1.0, scale) for r in candidates]
    best_r = candidates[0]
    best_err = float("inf")
    for r in candidates:
        err = 0.0
        for item in certainty_equivalents:
            low = float(item["low"])
            high = float(item["high"])
            p_high = float(item.get("probability_high", 0.5))
            ce = float(item["certainty_equivalent"])
            expected_exp = (1 - p_high) * math.exp(-low / r) + p_high * math.exp(-high / r)
            pred = -r * math.log(expected_exp)
            err += (pred - ce) ** 2
        if err < best_err:
            best_err = err
            best_r = r
    attitude = "risk_averse" if best_r < 1e8 else "risk_neutral"
    return {
        "status": "computed",
        "model": "exponential_utility_fit",
        "risk_tolerance": round(best_r, 6),
        "sse": round(best_err, 6),
        "risk_attitude": attitude,
        "bias_warnings": [
            "CE answers can be affected by anchoring, framing, overconfidence, and loss aversion.",
            "Use multiple preference questions and review inconsistency before using utility for high-stakes decisions.",
        ],
        "markdown_report": (
            "### Utility Curve Fitting\n\n"
            f"Estimated risk tolerance R = {best_r:.6f}; SSE = {best_err:.6f}.\n\n"
            "Cảnh báo: đây là subjective elicitation, cần kiểm tra framing/anchoring bias."
        ),
    }
