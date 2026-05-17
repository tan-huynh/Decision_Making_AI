from __future__ import annotations

from typing import Any


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
