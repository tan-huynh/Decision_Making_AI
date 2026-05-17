from __future__ import annotations

import re
from typing import Any

from .router import solve_problem


def parse_probability_tree_text(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    has_probability_language = any(token in lowered for token in ["xác suất", "xac suat", "probability", "biến cố", "bien co"])
    has_tree_language = any(token in lowered for token in ["hình cây", "decision tree", "sơ đồ", "so do", "tree"])
    if not has_probability_language and not has_tree_language:
        return None

    percent_match = re.search(r"(\d+(?:[,.]\d+)?)\s*%", text)
    if not percent_match:
        return None
    success_probability = float(percent_match.group(1).replace(",", ".")) / 100

    trials = 2
    numeric_trial_match = re.search(r"(?:khoan|drill)\s+(\d+)\s+(?:giếng|gieng|wells?)", text, flags=re.I)
    if numeric_trial_match:
        trials = int(numeric_trial_match.group(1))
    elif re.search(r"(?:khoan|drill)\s+hai\s+(?:giếng|gieng)", text, flags=re.I):
        trials = 2

    if trials < 2:
        return None

    return {
        "context": {
            "title": "Oil drilling probability tree",
            "domain": "probability",
            "decision_maker": "exploration analyst",
            "objective_direction": "compute",
            "unit": "probability",
            "description": text[:5000],
        },
        "problem_type": "decision_tree",
        "probability_tree": {
            "success_probability": success_probability,
            "trials": trials,
            "success_label": "Trúng dầu",
            "failure_label": "Không trúng dầu",
        },
        "assumptions": [
            "Xác suất trúng dầu của mỗi giếng là như nhau.",
            "Kết quả khoan giữa các giếng độc lập nếu đề bài không nêu phụ thuộc địa chất.",
            "Mỗi giếng chỉ có hai trạng thái: trúng dầu hoặc không trúng dầu.",
        ],
    }


def parse_power_transportation_text(text: str) -> dict[str, Any] | None:
    supply_matches = re.findall(r"Nhà máy\s+(\d+)\s*\|\s*(\d+(?:\.\d+)?)\s*MW", text, flags=re.I)
    demand_matches = re.findall(r"Thành phố\s+(\d+)\s*\|\s*(\d+(?:\.\d+)?)\s*MW", text, flags=re.I)
    cost_matches = re.findall(r"C(\d)(\d)\s*=\s*(\d+(?:\.\d+)?)", text, flags=re.I)
    if len(supply_matches) < 2 or len(demand_matches) < 2 or len(cost_matches) < 4:
        return None
    supplies = {f"P{i}": float(value) for i, value in supply_matches}
    demands = {f"C{j}": float(value) for j, value in demand_matches}
    costs: dict[str, dict[str, float]] = {}
    for i, j, value in cost_matches:
        costs.setdefault(f"P{i}", {})[f"C{j}"] = float(value)
    return {
        "context": {
            "title": "Power transmission transportation problem",
            "domain": "energy",
            "decision_maker": "system operator",
            "objective_direction": "minimize",
            "unit": "cost",
            "description": text[:5000],
        },
        "problem_type": "transportation",
        "graph": {"supplies": supplies, "demands": demands, "costs": costs},
        "assumptions": [
            "Total supply equals total demand.",
            "Transmission cost is linear per MW.",
            "No line capacity constraints beyond plant supply and city demand.",
        ],
    }


def parse_resource_allocation_dp_text(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    if "quy hoạch động" not in lowered and "dynamic" not in lowered:
        return None
    total_match = re.search(r"(\d+(?:\.\d+)?)\s*tấn", text, flags=re.I)
    profit_rows = []
    for line in text.splitlines():
        normalized = line.lower()
        if "lợi nhuận" not in normalized and "loi nhuan" not in normalized:
            continue
        if "sản xuất / lợi nhuận" in normalized or "san xuat / loi nhuan" in normalized:
            continue
        if "sản xuất (tấn)" in normalized or "san xuat (tan)" in normalized:
            continue
        numbers = [float(value) for value in re.findall(r"(?<!\^)(?<!\d)(\d+(?:\.\d+)?)", line)]
        # Drop exponent/unit markers from markdown such as 10^5 if they appear before the table values.
        if len(numbers) >= 4:
            profit_rows.append(numbers[-4:])
    if len(profit_rows) > 1 and profit_rows[0] == [0.0, 1.0, 2.0, 3.0]:
        profit_rows = profit_rows[1:]
    if not total_match or len(profit_rows) < 2:
        return None
    total_resource = int(float(total_match.group(1)))
    return {
        "context": {
            "title": "Steel production dynamic programming problem",
            "domain": "manufacturing",
            "decision_maker": "production planner",
            "objective_direction": "maximize",
            "unit": "10^5 USD",
            "description": text[:5000],
        },
        "problem_type": "dynamic_programming",
        "resource_allocation": {
            "resource_name": "tons",
            "total_resource": total_resource,
            "stage_returns": profit_rows,
        },
        "assumptions": [
            "Tổng sản lượng 7 tấn phải được phân bổ hết trong 3 ngày.",
            "Mỗi ngày có thể sản xuất 0-3 tấn theo bảng lợi nhuận.",
            "Lợi nhuận từng ngày độc lập và cộng được.",
        ],
    }


def solve_text_problem(text: str) -> dict[str, Any]:
    parsed = parse_probability_tree_text(text) or parse_resource_allocation_dp_text(text) or parse_power_transportation_text(text)
    if not parsed:
        return {
            "status": "needs_clarification",
            "message": "Không đủ dữ liệu để tự tạo mô hình. EDSS hiện cần nhận diện được transportation, dynamic programming resource allocation, hoặc probability tree.",
            "questions": [
                "Nếu là transportation: supply, demand và ma trận chi phí là gì?",
                "Nếu là quy hoạch động: tổng tài nguyên cần phân bổ là bao nhiêu?",
                "Bảng lợi nhuận/chi phí theo từng giai đoạn và từng mức tài nguyên là gì?",
                "Nếu là xác suất/cây biến cố: xác suất mỗi nhánh và số lần thử là gì?",
            ],
        }
    solved = solve_problem(parsed)
    return {"status": "solved", "problem": parsed, "solved": solved}
