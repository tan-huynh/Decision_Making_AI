from __future__ import annotations

import random
import statistics
from typing import Any


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def run_monte_carlo(payload: dict[str, Any], iterations: int = 1200, seed: int = 42) -> dict[str, Any]:
    rng = random.Random(seed)
    risk_tolerance = float(payload.get("riskTolerance", 0.5))
    option_scores: dict[str, list[float]] = {option["name"]: [] for option in payload.get("options", [])}
    winners: dict[str, int] = {option["name"]: 0 for option in payload.get("options", [])}

    for _ in range(iterations):
        round_scores: dict[str, float] = {}
        for option in payload.get("options", []):
            scenarios = option.get("scenarios", [])
            if not scenarios:
                continue
            noisy_probs = [max(0.001, rng.gauss(float(s.get("probability", 0.1)), 0.06)) for s in scenarios]
            total = sum(noisy_probs)
            score = 0.0
            utilities = []
            for scenario, probability in zip(scenarios, noisy_probs):
                utility = rng.gauss(float(scenario.get("utility", 0)), 8 + (1 - risk_tolerance) * 7)
                utilities.append(utility)
                score += (probability / total) * utility
            cost = max(0, rng.gauss(float(option.get("cost", 0)), 2.5))
            downside = statistics.pstdev(utilities) if len(utilities) > 1 else 0.0
            reversibility = float(option.get("reversibility", 0.5))
            score = score - cost - (1 - risk_tolerance) * downside - (1 - reversibility) * 5
            round_scores[option["name"]] = score
            option_scores[option["name"]].append(score)
        if round_scores:
            winner = max(round_scores.items(), key=lambda item: item[1])[0]
            winners[winner] += 1

    distributions = []
    for name, scores in option_scores.items():
        distributions.append(
            {
                "option": name,
                "mean": statistics.mean(scores) if scores else 0,
                "p10": percentile(scores, 0.1),
                "p50": percentile(scores, 0.5),
                "p90": percentile(scores, 0.9),
                "win_rate": winners.get(name, 0) / max(1, iterations),
            }
        )
    distributions.sort(key=lambda item: item["mean"], reverse=True)
    return {"iterations": iterations, "distributions": distributions}
