from __future__ import annotations

from typing import Any


KEYWORDS = {
    "linear_programming": ["resource", "capacity", "production mix", "allocate", "constraint", "profit", "cost minimization", "phân bổ", "công suất", "sản xuất"],
    "transportation": ["transport", "shipment", "warehouse", "supply", "demand", "logistics", "transshipment", "vận tải", "kho"],
    "assignment": ["assign", "matching", "worker", "machine", "job", "crew", "phân công", "ghép"],
    "shortest_path": ["shortest path", "route", "routing", "distance", "đường đi", "tuyến"],
    "dynamic_programming": ["stage", "sequential", "inventory", "maintenance", "policy", "multi-stage", "tồn kho", "bảo trì"],
    "decision_tree": ["chance", "state of nature", "payoff", "expected value", "regret", "uncertainty", "bất định"],
    "simulation_risk": ["simulation", "monte carlo", "distribution", "var", "cvar", "risk", "mô phỏng", "rủi ro"],
    "multi_objective": ["multi-objective", "pareto", "weighted", "safety", "co2", "reliability", "nhiều mục tiêu"],
}


def classify_problem(description: str, structured: dict[str, Any] | None = None) -> dict[str, Any]:
    text = description.lower()
    scores = {kind: 0 for kind in KEYWORDS}
    for kind, words in KEYWORDS.items():
        scores[kind] += sum(1 for word in words if word in text)
    structured = structured or {}
    if structured.get("assignment_costs"):
        scores["assignment"] += 5
    if structured.get("graph", {}).get("edges"):
        scores["shortest_path"] += 4
    if structured.get("variables") and structured.get("constraints") and structured.get("objective"):
        scores["linear_programming"] += 5
    if structured.get("decision_tree"):
        scores["decision_tree"] += 5
    if structured.get("objectives"):
        scores["multi_objective"] += 4
    best = max(scores.items(), key=lambda item: item[1])
    if best[1] == 0:
        return {
            "problem_type": "needs_clarification",
            "confidence": 0.25,
            "scores": scores,
            "reason": "Không tìm thấy đủ tín hiệu để chọn solver chắc chắn.",
        }
    total = sum(scores.values())
    return {
        "problem_type": best[0],
        "confidence": round(best[1] / max(1, total), 2),
        "scores": scores,
        "reason": f"Matched {best[1]} indicators for {best[0]}.",
    }


def missing_data_questions(problem: dict[str, Any]) -> list[str]:
    questions: list[str] = []
    kind = problem.get("problem_type") or classify_problem(problem.get("context", {}).get("description", ""), problem).get("problem_type")
    if not problem.get("context", {}).get("unit"):
        questions.append("Đơn vị đo của hàm mục tiêu là gì, ví dụ USD/tuần, MWh, giờ, tấn-km?")
    if kind == "linear_programming":
        if not problem.get("variables"):
            questions.append("Các biến quyết định là gì và giới hạn dưới/trên của từng biến?")
        if not problem.get("objective"):
            questions.append("Hàm mục tiêu cần maximize/minimize là gì và hệ số của từng biến?")
        if not problem.get("constraints"):
            questions.append("Các ràng buộc tài nguyên/công suất/ngân sách/thời gian là gì?")
    if kind in {"decision_tree", "simulation_risk"}:
        if not problem.get("states"):
            questions.append("Các trạng thái bất định và xác suất tương ứng là gì?")
        if not problem.get("payoff_matrix"):
            questions.append("Payoff/cost/loss của từng phương án dưới từng trạng thái là gì?")
    if kind == "assignment" and not problem.get("assignment_costs"):
        questions.append("Ma trận chi phí/lợi ích giữa từng người-mỗi việc hoặc máy-mỗi job là gì?")
    if kind == "shortest_path" and not problem.get("graph", {}).get("edges"):
        questions.append("Danh sách node, cạnh và cost/distance/time của từng cạnh là gì?")
    return questions or ["Dữ liệu đủ cho mô hình ban đầu; vui lòng xác nhận giả định trước khi chạy solver."]
