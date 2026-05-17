"""Risk Engine — VaR, CVaR, loss distribution, risk metrics.

Provides exact Value-at-Risk and Conditional VaR from raw samples,
plus distribution-based risk metrics and risk preference recommendations.
"""

from __future__ import annotations

import math
from typing import Any


def value_at_risk(
    samples: list[float],
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compute VaR and CVaR from raw samples.

    VaR_α = the α-th percentile loss (worst outcome not exceeded with prob α).
    CVaR_α = expected value of outcomes below VaR (expected shortfall).
    """
    if not samples:
        return {"VaR": 0.0, "CVaR": 0.0, "probability_of_loss": 0.0, "worst_case": 0.0}

    n = len(samples)
    ordered = sorted(samples)

    # VaR: the value at the (1-confidence) percentile
    var_index = max(0, min(n - 1, int(math.floor((1 - confidence) * n))))
    var_value = ordered[var_index]

    # CVaR: mean of all values at or below VaR
    tail = [v for v in ordered if v <= var_value]
    if not tail:
        tail = ordered[:max(1, var_index + 1)]
    cvar_value = sum(tail) / len(tail) if tail else var_value

    # Loss probability
    prob_loss = sum(1 for v in samples if v < 0) / n

    # Percentiles
    p5 = _percentile(ordered, 0.05)
    p25 = _percentile(ordered, 0.25)
    p50 = _percentile(ordered, 0.50)
    p75 = _percentile(ordered, 0.75)
    p95 = _percentile(ordered, 0.95)

    mean = sum(samples) / n
    variance = sum((x - mean) ** 2 for x in samples) / n
    std = math.sqrt(variance)

    # Sharpe-like ratio (mean / std)
    reward_risk_ratio = round(mean / std, 4) if std > 1e-8 else None

    return {
        "confidence": confidence,
        "n_samples": n,
        "VaR": round(var_value, 4),
        "CVaR": round(cvar_value, 4),
        "probability_of_loss": round(prob_loss, 4),
        "worst_case": round(ordered[0], 4),
        "best_case": round(ordered[-1], 4),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "percentiles": {
            "p5": round(p5, 4),
            "p25": round(p25, 4),
            "p50": round(p50, 4),
            "p75": round(p75, 4),
            "p95": round(p95, 4),
        },
        "reward_risk_ratio": reward_risk_ratio,
        "histogram": _mini_histogram(ordered, n_bins=10),
        "risk_assessment": _risk_assessment(mean, std, prob_loss, var_value),
    }


def risk_from_simulation(simulation_result: dict[str, Any], confidence: float = 0.95) -> dict[str, Any]:
    """Extract risk metrics from simulation results."""
    risks = []
    for item in simulation_result.get("results", []):
        raw = item.get("raw_samples", [])
        if raw:
            risk = value_at_risk(raw, confidence)
            risk["alternative"] = item.get("alternative", "")
            risks.append(risk)
        else:
            risks.append({
                "alternative": item.get("alternative", ""),
                "downside_proxy": item.get("p05", 0),
                "median": item.get("p50", 0),
                "upside_proxy": item.get("p95", 0),
                "risk_note": "Percentile proxies; store raw samples for exact VaR/CVaR.",
            })
    return {"confidence": confidence, "risks": risks}


def _percentile(sorted_data: list[float], p: float) -> float:
    n = len(sorted_data)
    if n == 0:
        return 0.0
    k = (n - 1) * p
    f = int(math.floor(k))
    c = min(n - 1, f + 1)
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def _mini_histogram(sorted_data: list[float], n_bins: int = 10) -> list[dict[str, Any]]:
    """Build a compact histogram for visualization."""
    if not sorted_data:
        return []
    mn, mx = sorted_data[0], sorted_data[-1]
    if mn == mx:
        return [{"low": mn, "high": mx, "count": len(sorted_data)}]
    width = (mx - mn) / n_bins
    bins = []
    for i in range(n_bins):
        low = mn + i * width
        high = mn + (i + 1) * width
        count = sum(1 for v in sorted_data if (low <= v < high if i < n_bins - 1 else low <= v <= high))
        bins.append({"low": round(low, 2), "high": round(high, 2), "count": count})
    return bins


def _risk_assessment(mean: float, std: float, prob_loss: float, var: float) -> str:
    if prob_loss > 0.3:
        return "⚠️ Rủi ro cao: P(loss) > 30%. Cân nhắc hedge hoặc chọn phương án bảo thủ hơn."
    if prob_loss > 0.1:
        return "Rủi ro trung bình: P(loss) = 10-30%. Cần sensitivity analysis để xác nhận."
    if mean > 0 and std < abs(mean) * 0.5:
        return "✓ Rủi ro thấp: mean dương với biến động nhỏ. Phương án ổn định."
    return "Rủi ro vừa phải. Xem xét phân bổ để giảm downside."
