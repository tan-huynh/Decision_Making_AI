"""Data Estimation Engine — Statistical analysis of historical data.

Provides:
- Descriptive statistics (mean, variance, median, quartiles)
- Histogram and empirical CDF
- Simpson's paradox detector
- Small sample warnings
- Comparison of suppliers/processes
"""

from __future__ import annotations

import math
import statistics
from typing import Any


def descriptive_stats(data: list[float], label: str = "data") -> dict[str, Any]:
    """Compute comprehensive descriptive statistics."""
    if not data:
        return {"error": "No data provided.", "label": label}

    n = len(data)
    sorted_data = sorted(data)
    mean = statistics.mean(data)
    warnings: list[str] = []

    if n < 5:
        warnings.append(f"Chỉ có {n} quan sát; thống kê có thể không đáng tin.")
    elif n < 30:
        warnings.append(f"Sample size = {n} < 30; ước lượng có thể không ổn định.")

    result: dict[str, Any] = {
        "label": label,
        "n": n,
        "mean": round(mean, 6),
        "median": round(statistics.median(data), 6),
        "min": round(sorted_data[0], 6),
        "max": round(sorted_data[-1], 6),
        "range": round(sorted_data[-1] - sorted_data[0], 6),
    }

    if n > 1:
        var = statistics.variance(data)
        std = statistics.stdev(data)
        result["variance"] = round(var, 6)
        result["std"] = round(std, 6)
        result["cv"] = round(std / abs(mean), 4) if abs(mean) > 1e-12 else None
        # Quartiles
        q1 = _percentile(sorted_data, 0.25)
        q3 = _percentile(sorted_data, 0.75)
        result["q1"] = round(q1, 6)
        result["q3"] = round(q3, 6)
        result["iqr"] = round(q3 - q1, 6)
        # Standard error
        result["se_mean"] = round(std / math.sqrt(n), 6)
        # 95% CI for mean (normal approx)
        z = 1.96
        result["ci_95"] = [round(mean - z * result["se_mean"], 4), round(mean + z * result["se_mean"], 4)]

    result["warnings"] = warnings
    return result


def build_histogram(
    data: list[float],
    n_bins: int | None = None,
    label: str = "data",
) -> dict[str, Any]:
    """Build histogram bins from data."""
    if not data:
        return {"error": "No data."}

    n = len(data)
    if n_bins is None:
        n_bins = max(5, min(30, int(math.sqrt(n))))

    mn, mx = min(data), max(data)
    if mn == mx:
        return {"bins": [{"low": mn, "high": mx, "count": n, "frequency": 1.0}], "n_bins": 1}

    width = (mx - mn) / n_bins
    bins: list[dict[str, Any]] = []
    for i in range(n_bins):
        low = mn + i * width
        high = mn + (i + 1) * width
        count = sum(1 for v in data if low <= v < high) if i < n_bins - 1 else sum(1 for v in data if low <= v <= high)
        bins.append({
            "bin": i,
            "low": round(low, 4),
            "high": round(high, 4),
            "count": count,
            "frequency": round(count / n, 4),
        })

    return {"label": label, "n": n, "n_bins": n_bins, "bin_width": round(width, 4), "bins": bins}


def empirical_cdf(data: list[float], label: str = "data") -> dict[str, Any]:
    """Build empirical CDF points."""
    if not data:
        return {"error": "No data."}
    sorted_data = sorted(data)
    n = len(sorted_data)
    points = [{"x": round(sorted_data[i], 6), "F_x": round((i + 1) / n, 6)} for i in range(n)]
    return {"label": label, "n": n, "cdf_points": points}


def compare_groups(
    groups: dict[str, list[float]],
) -> dict[str, Any]:
    """Compare multiple groups (e.g., suppliers, processes)."""
    summaries: list[dict[str, Any]] = []
    for name, data in groups.items():
        summaries.append(descriptive_stats(data, label=name))

    # Rank by mean
    ranked = sorted(summaries, key=lambda s: s.get("mean", 0), reverse=True)
    warnings: list[str] = []

    # Check for Simpson's paradox hint
    if len(groups) >= 2:
        names = list(groups.keys())
        means = [statistics.mean(groups[n]) for n in names]
        # If combining all data gives a different ranking than individual groups
        combined = []
        for data in groups.values():
            combined.extend(data)
        combined_mean = statistics.mean(combined) if combined else 0

        # Simple heuristic: warn if group sizes are very different
        sizes = [len(groups[n]) for n in names]
        if max(sizes) > 3 * min(sizes):
            warnings.append(
                "Kích thước nhóm chênh lệch lớn; kết quả aggregated "
                "có thể gây hiểu sai (Simpson's paradox risk)."
            )

    return {
        "n_groups": len(groups),
        "summaries": ranked,
        "best_by_mean": ranked[0]["label"] if ranked else None,
        "warnings": warnings,
    }


def _percentile(sorted_data: list[float], p: float) -> float:
    """Linear interpolation percentile."""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_data[0]
    k = (n - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)
