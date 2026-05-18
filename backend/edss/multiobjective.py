from __future__ import annotations

from typing import Any
import math


def weighted_score(problem: dict[str, Any]) -> dict[str, Any]:
    objectives = problem.get("objectives", [])
    alternatives = problem.get("alternatives", [])
    if not objectives or not alternatives:
        raise ValueError("Multi-objective scoring requires objectives and alternatives.")
    scores = []
    for alternative in alternatives:
        attrs = alternative.get("attributes", {})
        total = 0.0
        parts = {}
        for objective in objectives:
            name = objective["name"]
            weight = float(objective.get("weight", 1))
            direction = objective.get("direction", "maximize")
            value = float(attrs.get(name, 0))
            normalized = value if direction == "maximize" else -value
            parts[name] = weight * normalized
            total += parts[name]
        scores.append({"alternative": alternative["name"], "score": total, "parts": parts})
    scores.sort(key=lambda item: item["score"], reverse=True)
    return {"status": "computed", "method": "weighted_sum", "scores": scores, "recommendation": scores[0]["alternative"]}


def pareto_frontier(problem: dict[str, Any]) -> dict[str, Any]:
    objectives = problem.get("objectives", [])
    alternatives = problem.get("alternatives", [])
    frontier = []
    for candidate in alternatives:
        dominated = False
        for other in alternatives:
            if other is candidate:
                continue
            better_or_equal = True
            strictly_better = False
            for objective in objectives:
                name = objective["name"]
                direction = objective.get("direction", "maximize")
                c = float(candidate.get("attributes", {}).get(name, 0))
                o = float(other.get("attributes", {}).get(name, 0))
                if direction == "maximize":
                    better_or_equal &= o >= c
                    strictly_better |= o > c
                else:
                    better_or_equal &= o <= c
                    strictly_better |= o < c
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate["name"])
    return {"frontier": frontier}


def additive_utility(problem: dict[str, Any]) -> dict[str, Any]:
    objectives = problem.get("objectives", [])
    alternatives = problem.get("alternatives", [])
    if not objectives or not alternatives:
        raise ValueError("MAUT cần objectives và alternatives.")
    weight_sum = sum(float(obj.get("weight", 1)) for obj in objectives)
    scores = []
    for alt in alternatives:
        attrs = alt.get("attributes", {})
        total = 0.0
        parts = {}
        for obj in objectives:
            name = obj["name"]
            weight = float(obj.get("weight", 1)) / weight_sum if weight_sum else 0
            direction = obj.get("direction", "maximize")
            lo = float(obj.get("min", 0))
            hi = float(obj.get("max", 1))
            value = float(attrs.get(name, lo))
            normalized = 0.0 if abs(hi - lo) < 1e-12 else (value - lo) / (hi - lo)
            if direction == "minimize":
                normalized = 1 - normalized
            normalized = max(0.0, min(1.0, normalized))
            utility = weight * normalized
            parts[name] = round(utility, 6)
            total += utility
        scores.append({"alternative": alt["name"], "utility": round(total, 6), "parts": parts})
    scores.sort(key=lambda item: item["utility"], reverse=True)
    return {
        "status": "computed",
        "method": "additive_multi_attribute_utility",
        "scores": scores,
        "recommendation": scores[0]["alternative"] if scores else None,
        "markdown_report": _maut_report(scores),
    }


def ahp_weights(pairwise_matrix: list[list[float]], criteria: list[str] | None = None) -> dict[str, Any]:
    if not pairwise_matrix:
        raise ValueError("AHP cần pairwise_matrix.")
    n = len(pairwise_matrix)
    criteria = criteria or [f"c{i+1}" for i in range(n)]
    if any(len(row) != n for row in pairwise_matrix):
        raise ValueError("pairwise_matrix phải là ma trận vuông.")
    col_sums = [sum(float(pairwise_matrix[i][j]) for i in range(n)) for j in range(n)]
    normalized = [[float(pairwise_matrix[i][j]) / col_sums[j] for j in range(n)] for i in range(n)]
    weights = [sum(row) / n for row in normalized]
    aw = [sum(float(pairwise_matrix[i][j]) * weights[j] for j in range(n)) for i in range(n)]
    lambda_max = sum(aw[i] / weights[i] for i in range(n) if weights[i] > 0) / n
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0
    ri_lookup = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_lookup.get(n, 1.49)
    cr = ci / ri if ri else 0
    rows = [{"criterion": criteria[i], "weight": round(weights[i], 6)} for i in range(n)]
    return {
        "status": "computed",
        "method": "AHP",
        "weights": rows,
        "lambda_max": round(lambda_max, 6),
        "consistency_index": round(ci, 6),
        "consistency_ratio": round(cr, 6),
        "consistent": cr <= 0.1,
        "markdown_report": (
            "### Báo cáo AHP\n\n"
            f"CR = {cr:.4f}. " + ("Ma trận nhất quán chấp nhận được." if cr <= 0.1 else "CR > 0.1, nên đánh giá lại pairwise judgments.")
        ),
    }


def _maut_report(scores: list[dict[str, Any]]) -> str:
    lines = ["### Báo cáo Multi-Attribute Utility", "", "| Alternative | Utility |", "|---|---:|"]
    for row in scores:
        lines.append(f"| {row['alternative']} | {row['utility']:.6f} |")
    if scores:
        lines.append(f"\n**Khuyến nghị:** chọn `{scores[0]['alternative']}` vì có utility cao nhất.")
    return "\n".join(lines)
