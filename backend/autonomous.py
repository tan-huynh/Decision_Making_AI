from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edss.classifier import classify_problem


DATA_DIR = Path(__file__).resolve().parent / "data"
AUTO_MEMORY_FILE = DATA_DIR / "autonomous_learning.jsonl"
AUTO_PROFILE_FILE = DATA_DIR / "autonomous_profile.json"


ROUTE_STEPS = {
    "decision_analysis": ["recognize_decision_matrix", "validate_scenarios", "compute_scores", "audit_sensitivity_voi", "record_learning"],
    "edss_solver": ["recognize_case_type", "build_model", "validate_inputs", "run_solver", "audit_optimality", "record_learning"],
    "needs_clarification": ["recognize_gap", "identify_missing_data", "ask_clarifying_questions", "wait_for_user_input"],
}


def _decision_text(payload: dict[str, Any]) -> str:
    parts = [
        payload.get("question", ""),
        payload.get("objective", ""),
        payload.get("context", ""),
        payload.get("timeHorizon", ""),
    ]
    for option in payload.get("options", []):
        parts.append(str(option.get("name", "")))
        for scenario in option.get("scenarios", []):
            parts.append(str(scenario.get("name", "")))
            parts.append(str(scenario.get("evidence", "")))
    return "\n".join(part for part in parts if part)


def select_decision_route(payload: dict[str, Any]) -> dict[str, Any]:
    """Choose the computation route for a decision payload without requiring the user to pick it."""
    text = _decision_text(payload)
    classification = classify_problem(text, payload)
    problem_type = classification.get("problem_type", "unknown")
    confidence = float(classification.get("confidence", 0.25))
    options = payload.get("options", [])
    has_decision_matrix = len(options) >= 2 and all(option.get("scenarios") for option in options)

    if has_decision_matrix:
        selected_solver = "decision_engine"
        route = "decision_analysis"
        problem_type = "decision_theory"
        reason = "Có đủ options và scenarios nên dùng decision engine, đồng thời ghi nhận taxonomy để học route."
    elif confidence >= 0.45 and problem_type != "unknown":
        selected_solver = "edss_text_solver"
        route = "edss_solver"
        reason = "Không có ma trận lựa chọn đầy đủ; taxonomy đủ mạnh để chuyển sang EDSS solver."
    else:
        selected_solver = "clarification"
        route = "needs_clarification"
        reason = "Chưa đủ tín hiệu để chọn solver chắc chắn."

    return {
        "mode": "autonomous",
        "route": route,
        "selected_solver": selected_solver,
        "case_type": problem_type,
        "confidence": confidence,
        "reason": reason,
        "classification": classification,
        "agent_steps": ROUTE_STEPS[route],
        "similar_cases": similar_cases(problem_type, route),
    }


def select_text_solver_route(text: str, structured: dict[str, Any] | None = None) -> dict[str, Any]:
    classification = classify_problem(text, structured or {})
    problem_type = classification.get("problem_type", "unknown")
    confidence = float(classification.get("confidence", 0.25))
    return {
        "mode": "autonomous",
        "route": "edss_solver" if problem_type != "unknown" else "needs_clarification",
        "selected_solver": "edss_text_solver" if problem_type != "unknown" else "clarification",
        "case_type": problem_type,
        "confidence": confidence,
        "reason": classification.get("reason", ""),
        "classification": classification,
        "agent_steps": ROUTE_STEPS["edss_solver" if problem_type != "unknown" else "needs_clarification"],
        "similar_cases": similar_cases(problem_type, "edss_solver" if problem_type != "unknown" else "needs_clarification"),
    }


def _read_recent_events(limit: int = 50) -> list[dict[str, Any]]:
    if not AUTO_MEMORY_FILE.exists():
        return []
    try:
        lines = AUTO_MEMORY_FILE.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def similar_cases(case_type: str, route: str, limit: int = 3) -> list[dict[str, Any]]:
    events = list(reversed(_read_recent_events()))

    def compact(event: dict[str, Any]) -> dict[str, Any]:
        return {
            "route": event.get("route"),
            "case_type": event.get("case_type"),
            "selected_solver": event.get("selected_solver"),
            "created_at": event.get("created_at"),
        }

    exact = [compact(event) for event in events if event.get("case_type") == case_type]
    if len(exact) >= limit:
        return exact[:limit]

    matches = exact
    seen = {(item.get("case_type"), item.get("created_at")) for item in matches}
    for event in events:
        key = (event.get("case_type"), event.get("created_at"))
        if key in seen or event.get("route") != route:
            continue
        matches.append(
            compact(event)
        )
        if len(matches) >= limit:
            break
    return matches


def _load_profile() -> dict[str, Any]:
    if not AUTO_PROFILE_FILE.exists():
        return {"events": 0, "routes": {}, "case_types": {}, "solvers": {}}
    try:
        data = json.loads(AUTO_PROFILE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("events", 0)
            data.setdefault("routes", {})
            data.setdefault("case_types", {})
            data.setdefault("solvers", {})
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"events": 0, "routes": {}, "case_types": {}, "solvers": {}}


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = int(counter.get(key, 0)) + 1


def record_autonomous_learning(event: dict[str, Any]) -> dict[str, Any]:
    """Persist route-selection experience from every run.

    This learns case-selection frequency continuously. Outcome quality is still
    calibrated separately by explicit user feedback because the system should not
    infer success from merely producing an answer.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    entry = {**event, "created_at": now}
    with AUTO_MEMORY_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    profile = _load_profile()
    profile["events"] = int(profile.get("events", 0)) + 1
    _bump(profile["routes"], str(event.get("route", "unknown")))
    _bump(profile["case_types"], str(event.get("case_type", "unknown")))
    _bump(profile["solvers"], str(event.get("selected_solver", "unknown")))
    profile["last_event"] = {
        "route": event.get("route"),
        "case_type": event.get("case_type"),
        "selected_solver": event.get("selected_solver"),
        "confidence": event.get("confidence"),
        "agent_steps": event.get("agent_steps", []),
        "similar_case_count": len(event.get("similar_cases", [])),
        "created_at": now,
    }
    AUTO_PROFILE_FILE.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    return profile
