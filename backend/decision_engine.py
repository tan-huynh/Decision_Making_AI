from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent / "data"
MEMORY_FILE = DATA_DIR / "decision_memory.jsonl"
PROFILE_FILE = DATA_DIR / "expert_profile.json"

from edss.db import log_decision


@dataclass
class EngineProfile:
    cases: int = 0
    success_rate: float = 0.5
    calibration_shift: float = 0.0
    learned_principles: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "success_rate": self.success_rate,
            "calibration_shift": self.calibration_shift,
            "learned_principles": self.learned_principles or [],
        }


def load_profile() -> EngineProfile:
    if not PROFILE_FILE.exists():
        return EngineProfile(learned_principles=[])
    try:
        data = json.loads(PROFILE_FILE.read_text())
        return EngineProfile(
            cases=int(data.get("cases", 0)),
            success_rate=float(data.get("success_rate", 0.5)),
            calibration_shift=float(data.get("calibration_shift", 0.0)),
            learned_principles=list(data.get("learned_principles", [])),
        )
    except (json.JSONDecodeError, OSError, ValueError):
        return EngineProfile(learned_principles=[])


def save_profile(profile: EngineProfile) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_FILE.write_text(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False))


def normalize_probabilities(scenarios: list[dict[str, Any]], calibration_shift: float) -> tuple[list[dict[str, Any]], float]:
    raw = [max(0.0, float(item.get("probability", 0))) for item in scenarios]
    total = sum(raw)
    if total <= 0:
        raw = [1 / max(1, len(scenarios)) for _ in scenarios]
        total = 1.0
    normalized = []
    entropy_adjustment = min(0.08, abs(calibration_shift))
    uniform = 1 / max(1, len(scenarios))
    for item, probability in zip(scenarios, raw):
        p = probability / total
        p = (1 - entropy_adjustment) * p + entropy_adjustment * uniform
        normalized.append({**item, "probability": p})
    return normalized, total


def compute_results(payload: dict[str, Any]) -> dict[str, Any]:
    profile = load_profile()
    risk_tolerance = min(1.0, max(0.0, float(payload.get("riskTolerance", 0.5))))
    options = payload.get("options", [])
    if len(options) < 2:
        raise ValueError("Cần ít nhất 2 lựa chọn để so sánh.")

    option_results: list[dict[str, Any]] = []
    max_utility_by_scenario_index: dict[int, float] = {}

    normalized_options = []
    for option in options:
        scenarios, probability_sum = normalize_probabilities(option.get("scenarios", []), profile.calibration_shift)
        normalized_options.append((option, scenarios, probability_sum))
        for index, scenario in enumerate(scenarios):
            utility = float(scenario.get("utility", 0))
            max_utility_by_scenario_index[index] = max(max_utility_by_scenario_index.get(index, -math.inf), utility)

    for option, scenarios, probability_sum in normalized_options:
        cost = float(option.get("cost", 0))
        reversibility = min(1.0, max(0.0, float(option.get("reversibility", 0.5))))
        utilities = [float(s.get("utility", 0)) for s in scenarios]
        expected_utility = sum(float(s["probability"]) * float(s.get("utility", 0)) for s in scenarios) - cost
        downside = statistics.pstdev(utilities) if len(utilities) > 1 else 0.0
        worst_case = min(utilities) - cost if utilities else -cost
        best_case = max(utilities) - cost if utilities else -cost
        risk_penalty = (1 - risk_tolerance) * downside + (1 - reversibility) * max(0, 50 - worst_case) * 0.18
        risk_adjusted_score = expected_utility - risk_penalty
        scenario_results = []
        expected_regret = 0.0
        for index, scenario in enumerate(scenarios):
            utility = float(scenario.get("utility", 0))
            regret = max_utility_by_scenario_index.get(index, utility) - utility
            expected_regret += float(scenario["probability"]) * regret
            scenario_results.append(
                {
                    **scenario,
                    "contribution": float(scenario["probability"]) * utility,
                    "regret": regret,
                }
            )
        confidence = min(0.92, max(0.2, 0.55 + profile.success_rate * 0.25 + reversibility * 0.12 - downside / 300))
        option_results.append(
            {
                "id": option.get("id", option.get("name", "")),
                "name": option.get("name", "Unnamed option"),
                "expected_utility": expected_utility,
                "risk_adjusted_score": risk_adjusted_score,
                "expected_regret": expected_regret,
                "worst_case": worst_case,
                "best_case": best_case,
                "confidence": confidence,
                "normalized_probability_sum": probability_sum,
                "scenarios": scenario_results,
            }
        )

    option_results.sort(key=lambda item: (item["risk_adjusted_score"], -item["expected_regret"]), reverse=True)
    best = option_results[0]
    second = option_results[1]
    voi_score = max(0.0, 100 - (best["risk_adjusted_score"] - second["risk_adjusted_score"]) * 3)
    uncertain = []
    for option in option_results:
        for scenario in option["scenarios"]:
            p = float(scenario["probability"])
            if 0.25 <= p <= 0.75:
                uncertain.append(f"{option['name']}: {scenario['name']} ({p:.0%})")
    sensitivity = []
    for option in option_results:
        for scenario in option["scenarios"]:
            sensitivity.append(
                {
                    "option": option["name"],
                    "variable": scenario["name"],
                    "impact": abs(float(scenario["utility"]) - option["expected_utility"]) * float(scenario["probability"]),
                }
            )
    sensitivity.sort(key=lambda item: item["impact"], reverse=True)

    warnings = []
    for option in option_results:
        if abs(option["normalized_probability_sum"] - 1) > 0.05:
            warnings.append(f"Xác suất của '{option['name']}' đã được chuẩn hóa từ tổng {option['normalized_probability_sum']:.2f}.")
        if option["worst_case"] < 0 and risk_tolerance < 0.5:
            warnings.append(f"'{option['name']}' có worst-case âm; nên có phương án đảo ngược hoặc stop-loss.")

    assumptions = [
        "Utility đang dùng thang điểm chủ quan 0-100 trừ chi phí.",
        "Các tình huống trong mỗi lựa chọn được xem như loại trừ nhau.",
        "Risk-adjusted score phạt variance, downside và độ khó đảo ngược.",
    ]

    return {
        "recommendation": best["name"],
        "summary": (
            f"Lựa chọn mạnh nhất là '{best['name']}' với risk-adjusted score {best['risk_adjusted_score']:.1f}. "
            f"Chênh lệch với lựa chọn kế tiếp là {best['risk_adjusted_score'] - second['risk_adjusted_score']:.1f}; "
            f"expected regret {best['expected_regret']:.1f}. "
            "Nếu chênh lệch nhỏ hoặc VOI cao, nên thu thêm dữ liệu trước khi cam kết."
        ),
        "assumptions": assumptions,
        "warnings": warnings or ["Không có cảnh báo nghiêm trọng từ mô hình định lượng."],
        "option_results": option_results,
        "value_of_information": {
            "score": min(100.0, voi_score),
            "most_valuable_unknowns": uncertain[:5] or ["Chưa có biến bất định nổi bật; hãy thêm scenario cụ thể hơn."],
        },
        "sensitivity": sensitivity[:8],
        "learning_note": (
            f"Profile hiện có {profile.cases} case đã lưu, success-rate hiệu chỉnh {profile.success_rate:.0%}. "
            "Khi bạn ghi outcome, hệ thống cập nhật calibration cho các phân tích sau."
        ),
    }


def learn_from_outcome(payload: dict[str, Any]) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    outcome = payload.get("outcome", "mixed")
    score = {"success": 1.0, "mixed": 0.55, "failure": 0.0}.get(outcome, 0.55)
    profile = load_profile()
    next_cases = profile.cases + 1
    profile.success_rate = ((profile.success_rate * profile.cases) + score) / next_cases
    profile.cases = next_cases
    profile.calibration_shift = max(-0.15, min(0.15, (0.5 - profile.success_rate) * 0.2))
    principle = "Ưu tiên lựa chọn có regret thấp khi uncertainty cao." if outcome != "success" else "Giữ lại pattern quyết định có score và confidence cùng cao."
    principles = profile.learned_principles or []
    if principle not in principles:
        principles.append(principle)
    profile.learned_principles = principles[-12:]
    save_profile(profile)
    with MEMORY_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps({**payload, "profile_after": profile.to_dict()}, ensure_ascii=False) + "\n")
        
    # Log to SQLite Database for EDSS Phase 2 Learning
    decision_data = payload.get("decision", {})
    problem_text = decision_data.get("question", "")
    problem_type = decision_data.get("domain", "general")
    solver_used = "decision_engine"
    log_decision(problem_text, problem_type, solver_used, outcome)
    
    return {"ok": True, "profile": profile.to_dict()}
