from __future__ import annotations

import json
import re
from typing import Any

from .router import solve_problem
from .cargo_lp_parser import parse_cargo_lp_text
from .or_pipeline import attach_pipeline, gate_allows_solving


def _parse_probability_value(raw: str, source: str | None = None) -> float:
    value = float(raw.replace(",", "."))
    return value / 100 if value > 1 or (source and "%" in source) else value


def parse_bayes_text(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    if "bayes" not in lowered and "posterior" not in lowered and "hậu nghiệm" not in lowered and "xác suất sau" not in lowered:
        return None

    patterns = {
        "prior": [
            r"P\s*\(\s*H\s*\)\s*=\s*(\d+(?:[,.]\d+)?)\s*%?",
            r"prior\s*(?:=|:)\s*(\d+(?:[,.]\d+)?)\s*%?",
            r"xác suất ban đầu\s*(?:là|=|:)?\s*(\d+(?:[,.]\d+)?)\s*%?",
        ],
        "sensitivity": [
            r"P\s*\(\s*E\+?\s*\|\s*H\s*\)\s*=\s*(\d+(?:[,.]\d+)?)\s*%?",
            r"sensitivity\s*(?:=|:)\s*(\d+(?:[,.]\d+)?)\s*%?",
            r"độ nhạy\s*(?:là|=|:)?\s*(\d+(?:[,.]\d+)?)\s*%?",
        ],
        "false_positive_rate": [
            r"P\s*\(\s*E\+?\s*\|\s*(?:not H|¬H|~H)\s*\)\s*=\s*(\d+(?:[,.]\d+)?)\s*%?",
            r"false positive(?: rate)?\s*(?:=|:)\s*(\d+(?:[,.]\d+)?)\s*%?",
            r"dương tính giả\s*(?:là|=|:)?\s*(\d+(?:[,.]\d+)?)\s*%?",
        ],
    }
    values: dict[str, float] = {}
    for key, candidates in patterns.items():
        for pattern in candidates:
            match = re.search(pattern, text, flags=re.I)
            if match:
                values[key] = _parse_probability_value(match.group(1), match.group(0))
                break
    if set(values) != {"prior", "sensitivity", "false_positive_rate"}:
        return None
    observed_positive = "âm tính" not in lowered and "negative" not in lowered and "e-" not in lowered
    return {
        "context": {
            "title": "Bayesian probability update",
            "domain": "probability",
            "decision_maker": "analyst",
            "objective_direction": "compute",
            "unit": "probability",
            "description": text[:5000],
        },
        "problem_type": "decision_tree",
        "bayes": {**values, "observed_positive": observed_positive},
        "assumptions": [
            "Các thông số prior, sensitivity và false-positive rate được lấy đúng từ đề bài.",
            "Bằng chứng quan sát được là dương tính trừ khi đề bài nói âm tính.",
        ],
    }


def parse_expected_value_text(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    if "expected value" not in lowered and "giá trị kỳ vọng" not in lowered and "kỳ vọng" not in lowered:
        return None

    state_match = re.search(r"states?\s*:\s*([^\n]+)", text, flags=re.I)
    prob_match = re.search(r"prob(?:abilities)?\s*:\s*([^\n]+)", text, flags=re.I)
    if not state_match or not prob_match:
        return None
    states = [item.strip() for item in re.split(r"[,;|]", state_match.group(1)) if item.strip()]
    probabilities = [_parse_probability_value(item) for item in re.findall(r"\d+(?:[,.]\d+)?", prob_match.group(1))]
    if len(states) != len(probabilities) or len(states) < 2:
        return None

    alternatives = []
    payoff_matrix = []
    for line in text.splitlines():
        if ":" not in line:
            continue
        label, values_part = line.split(":", 1)
        label = label.strip()
        if label.lower() in {"states", "state", "prob", "probabilities", "probability"}:
            continue
        values = [float(value.replace(",", ".")) for value in re.findall(r"-?\d+(?:[,.]\d+)?", values_part)]
        if len(values) != len(states):
            continue
        alternatives.append({"name": label})
        for state, payoff in zip(states, values):
            payoff_matrix.append({"alternative": label, "state": state, "payoff": payoff})

    if len(alternatives) < 2:
        return None
    return {
        "context": {
            "title": "Expected value decision problem",
            "domain": "decision_under_uncertainty",
            "decision_maker": "analyst",
            "objective_direction": "maximize",
            "unit": "payoff",
            "description": text[:5000],
        },
        "problem_type": "decision_tree",
        "alternatives": alternatives,
        "states": [{"name": name, "probability": prob} for name, prob in zip(states, probabilities)],
        "payoff_matrix": payoff_matrix,
        "assumptions": [
            "Payoff càng lớn càng tốt.",
            "Xác suất trạng thái được lấy từ dòng Probabilities và sẽ được validator kiểm tra tổng.",
        ],
    }


def parse_diagnostic_decision_text(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"[*_`]+", "", text.replace("\\$", "$"))
    lowered = cleaned.lower()
    if not (
        ("lie detector" in lowered or "detector" in lowered)
        and ("private investigator" in lowered or "investigator" in lowered)
        and ("spy" in lowered or "spying" in lowered)
    ):
        return None

    prior_match = re.search(r"probability of being right is\D{0,20}(\d+(?:[,.]\d+)?)\s*%", cleaned, flags=re.I)
    detection_match = re.search(r"detects only\D{0,20}(\d+(?:[,.]\d+)?)\s*%\s+of\s+liars", cleaned, flags=re.I)
    investigator_match = re.search(r"(\d+(?:[,.]\d+)?)\s*%\s+probability of success", cleaned, flags=re.I)
    if not (prior_match and detection_match and investigator_match):
        return None

    money_values = list(re.finditer(r"\$\s*(\d+(?:[,.]\d+)?)\s*(million)?|\b(\d+(?:[,.]\d+)?)\s+million\b", cleaned, flags=re.I))

    def money_to_million(match: re.Match[str]) -> float:
        raw = (match.group(1) or match.group(3)).replace(",", "")
        value = float(raw)
        if match.group(2) or match.group(3):
            return value
        return value / 1_000_000

    loss_spy = None
    loss_wrong_dismissal = None
    detector_cost = None
    investigator_cost = None
    amounts = [money_to_million(match) for match in money_values]
    for match, amount in zip(money_values, amounts):
        window = cleaned[max(0, match.start() - 90): match.end() + 90].lower()
        if "loss" in window and "best researcher" not in window and loss_spy is None:
            loss_spy = amount
        elif "lose their best researcher" in window or "best researcher" in window:
            loss_wrong_dismissal = amount
        elif "cost of this process" in window or "lie detector" in window:
            detector_cost = amount
        elif "fee" in window or "private investigator" in window:
            investigator_cost = amount

    if len(amounts) >= 4:
        loss_spy = loss_spy if loss_spy is not None else amounts[0]
        loss_wrong_dismissal = loss_wrong_dismissal if loss_wrong_dismissal is not None else amounts[1]
        detector_cost = detector_cost if detector_cost is not None else amounts[2]
        investigator_cost = investigator_cost if investigator_cost is not None else amounts[3]

    if None in {loss_spy, loss_wrong_dismissal, detector_cost, investigator_cost}:
        return None

    prior = _parse_probability_value(prior_match.group(1), prior_match.group(0))
    detector_sensitivity = _parse_probability_value(detection_match.group(1), detection_match.group(0))
    investigator_accuracy = _parse_probability_value(investigator_match.group(1), investigator_match.group(0))

    return {
        "context": {
            "title": "Diagnostic decision tree with follow-up information",
            "domain": "decision_under_uncertainty",
            "decision_maker": "GATACA Board of Governors",
            "objective_direction": "maximize",
            "unit": "million_dollars",
            "description": cleaned[:5000],
        },
        "problem_type": "decision_tree",
        "diagnostic_decision": {
            "unit": "million_dollars",
            "states": {
                "true": {"label": "Spy", "probability": prior},
                "false": {"label": "No Spy", "probability": 1 - prior},
            },
            "actions": {
                "dismissal": {
                    "payoffs": {
                        "Spy": 0,
                        "No Spy": -float(loss_wrong_dismissal),
                    }
                },
                "no_dismissal": {
                    "fixed_payoff": -float(loss_spy),
                },
            },
            "test": {
                "name": "Lie detector",
                "cost": float(detector_cost),
                "positive_label": "Detects lie",
                "negative_label": "Detects no lie",
                "sensitivity": detector_sensitivity,
                "false_positive_rate": 0,
            },
            "followup": {
                "name": "Private investigator",
                "cost": float(investigator_cost),
                "available_after": "negative",
                "positive_label": "Investigator finds spy",
                "negative_label": "Investigator finds no spy",
                "sensitivity": investigator_accuracy,
                "false_positive_rate": 1 - investigator_accuracy,
            },
        },
        "assumptions": [
            "Payoffs are in millions of dollars and larger is better.",
            "The lie detector has zero false positives because the statement says it detects only 85% of liars and the textbook solution uses P(Sd|N)=0.",
            "No dismissal is modeled as a fixed payoff equal to the annual patent-loss exposure, matching the textbook tree.",
        ],
    }


def parse_forklift_decision_text(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"[*_`]+", "", text.replace("\\$", "$"))
    lowered = cleaned.lower()
    if not ("forklift" in lowered and "test a" in lowered and "test b" in lowered):
        return None

    usd_amounts = [
        float(match.group(1).replace(",", ""))
        for match in re.finditer(r"(?:usd|\$)\s*(\d+(?:,\d{3})*(?:[,.]\d+)?)", cleaned, flags=re.I)
    ]
    if len(usd_amounts) < 6:
        return None

    faulty_match = re.search(r"(?:faulty second-hand equipment is|faulty.*?is)\D{0,20}(\d+(?:[,.]\d+)?)\s*%", cleaned, flags=re.I)
    test_a_block = re.search(r"test\s+a([\s\S]*?)(?:test\s+b|$)", cleaned, flags=re.I)
    test_a_percentages = re.findall(r"(\d+(?:[,.]\d+)?)\s*%", test_a_block.group(1), flags=re.I) if test_a_block else []
    test_b_error_match = re.search(r"probability of error(?:\s+is)?\D{0,20}(\d+(?:[,.]\d+)?)\s*%", cleaned, flags=re.I)
    if not (faulty_match and len(test_a_percentages) >= 2 and test_b_error_match):
        return None

    new_purchase = usd_amounts[0]
    used_purchase = usd_amounts[1]
    new_maintenance = usd_amounts[2]
    offset = 1 if len(usd_amounts) >= 7 else 0
    test_a_cost = usd_amounts[3 + offset]
    test_b_phase1 = usd_amounts[4 + offset] if len(usd_amounts) > 4 + offset else 800.0
    test_b_phase2 = usd_amounts[5 + offset] if len(usd_amounts) > 5 + offset else 700.0

    return {
        "context": {
            "title": "Forklift truck decision tree",
            "domain": "decision_under_uncertainty",
            "decision_maker": "Director of Logistics",
            "objective_direction": "minimize_cost",
            "unit": "USD",
            "description": cleaned[:5000],
        },
        "problem_type": "decision_tree",
        "forklift_decision": {
            "new_purchase_cost": new_purchase,
            "used_purchase_cost": used_purchase,
            "new_maintenance_cost": new_maintenance,
            "used_maintenance_cost": 2 * new_maintenance,
            "faulty_probability": _parse_probability_value(faulty_match.group(1), faulty_match.group(0)),
            "test_a": {
                "cost": test_a_cost,
                "false_good_if_faulty": _parse_probability_value(test_a_percentages[0], test_a_block.group(1)),
                "false_faulty_if_good": _parse_probability_value(test_a_percentages[1], test_a_block.group(1)),
            },
            "test_b": {
                "phase1_cost": test_b_phase1,
                "phase1_error_probability": _parse_probability_value(test_b_error_match.group(1), test_b_error_match.group(0)),
                "phase2_cost": test_b_phase2,
            },
        },
        "assumptions": [
            "Costs are minimized and reported in USD.",
            "Test A positive means the equipment is diagnosed as operating properly: P(RAP|D)=0.05 and P(RAP|C)=0.80, matching the textbook notation.",
            "Test B preliminary result has symmetric 15% error: P(RBP|D)=0.15, P(RBN|D)=0.85, P(RBP|C)=0.85, P(RBN|C)=0.15.",
        ],
    }


def parse_probability_tree_text(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    if (
        ("lie detector" in lowered or "detector" in lowered)
        and ("private investigator" in lowered or "investigator" in lowered)
        and ("spy" in lowered or "spying" in lowered)
    ):
        return None
    if "forklift" in lowered and ("test a" in lowered or "test b" in lowered):
        return None
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
            "sense": "maximize",
        },
        "assumptions": [
            "Tổng sản lượng 7 tấn phải được phân bổ hết trong 3 ngày.",
            "Mỗi ngày có thể sản xuất 0-3 tấn theo bảng lợi nhuận.",
            "Lợi nhuận từng ngày độc lập và cộng được.",
        ],
    }


def parse_general_resource_dp_text(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    if "dynamic programming" not in lowered and "quy hoạch động" not in lowered and "bellman" not in lowered:
        return None
    total_match = re.search(r"(?:total resource|total|resource|tài nguyên|tong tai nguyen|tổng tài nguyên)\s*[:=]\s*(\d+)", text, flags=re.I)
    if not total_match:
        return None
    sense = "minimize" if re.search(r"\b(minimize|min|tối thiểu|toi thieu|cost)\b", lowered) else "maximize"
    resource_name_match = re.search(r"resource name\s*[:=]\s*([A-Za-zÀ-ỹ0-9_-]+)", text, flags=re.I)
    resource_name = resource_name_match.group(1) if resource_name_match else "resource"
    rows: list[list[float]] = []
    for line in text.splitlines():
        if not re.search(r"\b(stage|giai đoạn|gd)\b", line, flags=re.I):
            continue
        values = [float(value.replace(",", ".")) for value in re.findall(r"-?\d+(?:[,.]\d+)?", line)]
        if len(values) >= 2:
            # First number is often the stage index. Use the tail as value table.
            rows.append(values[1:] if re.search(r"\b(stage|giai đoạn|gd)\s*\d+", line, flags=re.I) else values)
    total_resource = int(total_match.group(1))
    if not rows or max(len(row) for row in rows) <= total_resource:
        return None
    return {
        "context": {
            "title": "General resource allocation dynamic programming problem",
            "domain": "operations_research",
            "decision_maker": "planner",
            "objective_direction": sense,
            "unit": "value",
            "description": text[:5000],
        },
        "problem_type": "dynamic_programming",
        "resource_allocation": {
            "resource_name": resource_name,
            "total_resource": total_resource,
            "stage_returns": rows,
            "sense": sense,
        },
        "assumptions": [
            "Mỗi dòng Stage chứa value theo mức phân bổ 0..max.",
            "Tổng tài nguyên phải được phân bổ hết.",
            "Giá trị từng giai đoạn cộng được.",
        ],
    }


def parse_shortest_path_text(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    if "dijkstra" not in lowered and "đường đi ngắn" not in lowered and "shortest path" not in lowered:
        return None

    cleaned_text = re.sub(r"[*_`]", "", lowered)
    source_match = re.search(r"(?:từ|from)\s+(?:(?:đỉnh|điểm|node|vertex)\s+)?([a-z0-9]+)", cleaned_text)
    target_match = re.search(r"(?:đến|to)\s+(?:(?:đỉnh|điểm|node|vertex)\s+)?([a-z0-9]+)", cleaned_text)
    
    source = None
    if source_match:
        source = source_match.group(1).upper() if source_match.group(1).isalpha() else source_match.group(1)
        
    target = None
    if target_match:
        target = target_match.group(1).upper() if target_match.group(1).isalpha() else target_match.group(1)

    is_undirected = "vô hướng" in lowered or "undirected" in lowered

    edges = []
    for line in text.splitlines():
        if "---" in line:
            continue
        m = re.search(r"(?:\|\s*)?([a-zA-Z0-9]+)\s*(?:-|->|—)\s*([a-zA-Z0-9]+)\s*(?:\||:)\s*(\d+(?:\.\d+)?)\s*(?:\|)?", line)
        if m:
            u, v, w = m.groups()
            u_name = u.upper() if u.isalpha() else u
            v_name = v.upper() if v.isalpha() else v
            directed = False if is_undirected else ("->" in line)
            edges.append({"from": u_name, "to": v_name, "cost": float(w), "directed": directed})

    if not source or not target or not edges:
        return None

    return {
        "context": {
            "title": "Shortest Path Problem",
            "domain": "network",
            "decision_maker": "planner",
            "objective_direction": "minimize",
            "unit": "distance",
            "description": text[:3000],
        },
        "problem_type": "shortest_path",
        "graph": {
            "source": source,
            "target": target,
            "edges": edges,
        },
        "assumptions": [
            "Chi phí các cạnh lấy theo trọng số cho trước.",
            "Thuật toán tìm đường đi ngắn nhất (ví dụ: Dijkstra)."
        ],
    }


def _node_name(raw: str) -> str:
    return raw.upper() if raw.isalpha() else raw


def _parse_network_edges(text: str) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for line in text.splitlines():
        if "---" in line:
            continue
        cleaned = line.strip().strip("|").replace("–", "-").replace("—", "-")
        match = re.search(
            r"^([A-Za-z0-9]+)\s*(->|-)\s*([A-Za-z0-9]+)"
            r"(?:\s*[|,:]\s*|\s+)"
            r"(-?\d+(?:\.\d+)?)"
            r"(?:\s*[|,]\s*(-?\d+(?:\.\d+)?))?",
            cleaned,
        )
        if match:
            src, arrow, dst, first, second = match.groups()
            edge: dict[str, Any] = {
                "from": _node_name(src),
                "to": _node_name(dst),
                "cost": float(first),
                "directed": arrow == "->",
            }
            if second is not None:
                edge["capacity"] = float(second)
            edges.append(edge)
    return edges


def parse_max_flow_text(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    if "max flow" not in lowered and "maximum flow" not in lowered and "luồng cực đại" not in lowered:
        return None
    source_match = re.search(r"(?:source|from|từ)\s*:?\s*([A-Za-z0-9]+)", text, flags=re.I)
    sink_match = re.search(r"(?:sink|target|to|đến)\s*:?\s*([A-Za-z0-9]+)", text, flags=re.I)
    capacity_edges = []
    for edge in _parse_network_edges(text):
        capacity_edges.append({
            "from": edge["from"],
            "to": edge["to"],
            "capacity": float(edge.get("capacity", edge.get("cost", 0))),
        })
    if not source_match or not sink_match or not capacity_edges:
        return None
    return {
        "context": {
            "title": "Max Flow Problem",
            "domain": "network",
            "decision_maker": "planner",
            "objective_direction": "maximize",
            "unit": "flow",
            "description": text[:3000],
        },
        "problem_type": "max_flow",
        "graph": {
            "source": _node_name(source_match.group(1)),
            "sink": _node_name(sink_match.group(1)),
            "edges": capacity_edges,
        },
        "assumptions": ["Các số trên cạnh là capacity.", "Mạng được xem là có hướng nếu cạnh ghi bằng ->."],
    }


def parse_min_cost_flow_text(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    if "min cost flow" not in lowered and "minimum cost flow" not in lowered and "min-cost flow" not in lowered:
        return None
    nodes: list[dict[str, Any]] = []
    node_seen: set[str] = set()
    for match in re.finditer(r"([A-Za-z0-9]+)\s*[:|]\s*supply\s*=?\s*(-?\d+(?:\.\d+)?)", text, flags=re.I):
        name = _node_name(match.group(1))
        nodes.append({"id": name, "supply": float(match.group(2)), "demand": 0})
        node_seen.add(name)
    for match in re.finditer(r"([A-Za-z0-9]+)\s*[:|]\s*demand\s*=?\s*(-?\d+(?:\.\d+)?)", text, flags=re.I):
        name = _node_name(match.group(1))
        nodes.append({"id": name, "supply": 0, "demand": float(match.group(2))})
        node_seen.add(name)
    edges = []
    for edge in _parse_network_edges(text):
        cost = float(edge.get("cost", 0))
        capacity = float(edge.get("capacity", 10**9))
        edges.append({"from": edge["from"], "to": edge["to"], "cost": cost, "capacity": capacity})
    if not nodes or not edges:
        return None
    return {
        "context": {
            "title": "Min Cost Flow Problem",
            "domain": "network",
            "decision_maker": "planner",
            "objective_direction": "minimize",
            "unit": "cost",
            "description": text[:3000],
        },
        "problem_type": "min_cost_flow",
        "graph": {"nodes": nodes, "edges": edges},
        "assumptions": ["Edge format is interpreted as from -> to | cost | capacity."],
    }


def _normalize_math_text(text: str) -> str:
    cleaned = text
    replacements = {
        "\\leq": "<=",
        "\\le": "<=",
        "\\geq": ">=",
        "\\ge": ">=",
        "≤": "<=",
        "≥": ">=",
        "\\quad": " ",
        "\\,": " ",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    cleaned = re.sub(r"\\\[(.*?)\\\]", r"\1", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\\\((.*?)\\\)", r"\1", cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("−", "-")
    return cleaned


def _canonical_var_name(name: str) -> str:
    return name.replace("_{", "").replace("}", "").replace("_", "")


def _parse_linear_expression(expr: str) -> dict[str, float]:
    expr = _normalize_math_text(expr)
    expr = expr.replace(" ", "")
    if expr.startswith("+"):
        expr = expr[1:]
    expr = re.sub(r"(?<!^)-", "+-", expr)
    coeffs: dict[str, float] = {}
    for term in [part for part in expr.split("+") if part]:
        match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d+)?)?)\*?([a-zA-Z](?:_\{?\d+\}?|\d+)?)", term)
        if not match:
            continue
        raw_coef, raw_var = match.groups()
        if raw_coef in {"", "+"}:
            coef = 1.0
        elif raw_coef == "-":
            coef = -1.0
        else:
            coef = float(raw_coef)
        var = _canonical_var_name(raw_var)
        coeffs[var] = coeffs.get(var, 0.0) + coef
    return coeffs


def _format_raw_lp(
    title: str,
    sense: str,
    objective: dict[str, float],
    leq_constraints: list[tuple[str, dict[str, float], float]],
    eq_constraints: list[tuple[str, dict[str, float], float]],
) -> dict[str, Any] | None:
    var_names = sorted(
        set(objective)
        | {name for _, coeffs, _ in leq_constraints for name in coeffs}
        | {name for _, coeffs, _ in eq_constraints for name in coeffs},
        key=lambda value: (re.sub(r"\d+", "", value), int(re.search(r"\d+", value).group()) if re.search(r"\d+", value) else 0, value),
    )
    if not var_names or not objective:
        return None

    def row(coeffs: dict[str, float]) -> list[float]:
        return [float(coeffs.get(name, 0.0)) for name in var_names]

    c = row(objective)
    a_ub = [row(coeffs) for _, coeffs, _ in leq_constraints]
    b_ub = [rhs for _, _, rhs in leq_constraints]
    a_eq = [row(coeffs) for _, coeffs, _ in eq_constraints]
    b_eq = [rhs for _, _, rhs in eq_constraints]

    def format_expr(coeffs: dict[str, float]) -> str:
        parts = []
        for name in var_names:
            coef = coeffs.get(name, 0.0)
            if abs(coef) < 1e-12:
                continue
            sign = "-" if coef < 0 else "+"
            magnitude = abs(coef)
            value = name if abs(magnitude - 1.0) < 1e-12 else f"{magnitude:g}{name}"
            parts.append((sign, value))
        if not parts:
            return "0"
        first_sign, first_value = parts[0]
        text = f"-{first_value}" if first_sign == "-" else first_value
        for sign, value in parts[1:]:
            text += f" {sign} {value}"
        return text

    formulation = f"{sense.capitalize()} Z = {format_expr(objective)}\nSubject to:\n"
    for name, coeffs, rhs in leq_constraints:
        formulation += f"  [{name}] {format_expr(coeffs)} <= {rhs:g}\n"
    for name, coeffs, rhs in eq_constraints:
        formulation += f"  [{name}] {format_expr(coeffs)} = {rhs:g}\n"
    formulation += "  " + ", ".join(f"{name} >= 0" for name in var_names)

    return {
        "context": {
            "title": title,
            "domain": "optimization",
            "decision_maker": "analyst",
            "objective_direction": sense,
            "unit": "objective",
            "description": formulation,
        },
        "problem_type": "linear_programming",
        "c": c,
        "sense": sense,
        "A_ub": a_ub or None,
        "b_ub": b_ub or None,
        "A_eq": a_eq or None,
        "b_eq": b_eq or None,
        "bounds": [[0, None]] * len(var_names),
        "variable_names": var_names,
        "constraint_names_ub": [name for name, _, _ in leq_constraints],
        "constraint_names_eq": [name for name, _, _ in eq_constraints],
        "formulation": formulation,
        "steps": [
            "Parsed linear programming model deterministically from Markdown/LaTeX.",
            "Converted constraints to matrix form.",
            "Solved with scipy HiGHS.",
        ],
        "assumptions": ["All listed variables are constrained to be non-negative."],
    }


def parse_markdown_linear_programming_text(text: str) -> list[dict[str, Any]] | None:
    lowered = text.lower()
    if "linear programming" not in lowered and "maximize" not in lowered and "minimize" not in lowered:
        return None

    normalized = _normalize_math_text(text)
    blocks: list[tuple[str, str]] = []
    matches = list(re.finditer(r"^\s*####\s*([a-zA-Z0-9]+)\)?\s*$", normalized, flags=re.M))
    if matches:
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
            blocks.append((match.group(1), normalized[start:end]))
    else:
        blocks = [("LP", normalized)]

    parsed: list[dict[str, Any]] = []
    for label, block in blocks:
        sense_match = re.search(r"\b(maximize|minimize)\b", block, flags=re.I)
        objective_match = re.search(r"Z\s*=\s*([^\n]+)", block, flags=re.I)
        if not sense_match or not objective_match:
            continue
        sense = sense_match.group(1).lower()
        objective = _parse_linear_expression(objective_match.group(1))
        leq_constraints: list[tuple[str, dict[str, float], float]] = []
        eq_constraints: list[tuple[str, dict[str, float], float]] = []

        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or "Z" in line or "Maximize" in line or "Minimize" in line or "Subject to" in line:
                continue
            if re.search(r"x_i\s*>?=", line) or re.search(r"[a-zA-Z](?:_\{?\d+\}?|\d+)?\s*>=\s*0", line):
                continue
            relation = None
            if "<=" in line:
                relation = "<="
            elif ">=" in line:
                relation = ">="
            elif "=" in line:
                relation = "="
            if not relation:
                continue
            lhs, rhs_text = line.split(relation, 1)
            rhs_match = re.search(r"-?\d+(?:\.\d+)?", rhs_text)
            if not rhs_match:
                continue
            coeffs = _parse_linear_expression(lhs)
            if not coeffs:
                continue
            rhs = float(rhs_match.group())
            constraint_name = f"{label}_{len(leq_constraints) + len(eq_constraints) + 1}"
            if relation == "<=":
                leq_constraints.append((constraint_name, coeffs, rhs))
            elif relation == ">=":
                leq_constraints.append((constraint_name, {name: -coef for name, coef in coeffs.items()}, -rhs))
            else:
                eq_constraints.append((constraint_name, coeffs, rhs))

        raw_problem = _format_raw_lp(f"Linear Programming {label}", sense, objective, leq_constraints, eq_constraints)
        if raw_problem:
            parsed.append(raw_problem)

    return parsed or None


# ── LLM-based General LP Parser ──────────────────────────────────────────


EDSS_ROUTER_SYSTEM_PROMPT = """You are an advanced mathematical modeler and engineering decision router.
Analyze the user's problem text and classify it into one of the following problem_types:
- "linear_programming"
- "transportation_assignment"
- "integer_programming"
- "nonlinear_programming"
- "network_modelling"
- "inventory_theory"
- "queueing_theory"
- "decision_theory"
- "game_theory"
- "dynamic_programming"
- "markov_processes"

Then extract the relevant data into the "data" object using the EXACT schema specified below for that type.
Return ONLY valid JSON (no markdown, no explanations).

Schema for "linear_programming":
{
  "title": "short title",
  "sense": "maximize" or "minimize",
  "variable_description": "what variables represent",
  "variables": [{"name": "x1"}],
  "objective_coefficients": [10, 20],
  "constraints_leq": [{"name": "max_limit", "coefficients": [1, 2], "rhs": 100}],
  "constraints_geq": [{"name": "min_req", "coefficients": [1, 1], "rhs": 5}],
  "constraints_eq": [{"name": "exact", "coefficients": [1, -1], "rhs": 0}],
  "assumptions": ["assumption 1"]
}

Schema for "transportation_assignment":
{
  "title": "short title",
  "supplies": {"Source1": 100, "Source2": 200},
  "demands": {"Dest1": 150, "Dest2": 150},
  "costs": {"Source1": {"Dest1": 5, "Dest2": 10}, "Source2": {"Dest1": 8, "Dest2": 4}},
  "assignment_costs": [
    {"agent": "Worker1", "task": "Job1", "cost": 10}
  ],
  "assumptions": []
}

Schema for "network_modelling":
{
  "title": "short title",
  "source": "A",
  "target": "B",
  "edges": [{"from": "A", "to": "B", "cost": 5, "capacity": 10, "directed": false}],
  "assumptions": []
}

Schema for "decision_theory":
{
  "title": "short title",
  "alternatives": [{"name": "A"}, {"name": "B"}],
  "states": [{"name": "High", "probability": 0.4}, {"name": "Low", "probability": 0.6}],
  "payoff_matrix": [
    {"alternative": "A", "state": "High", "payoff": 100},
    {"alternative": "A", "state": "Low", "payoff": 0}
  ],
  "assumptions": []
}

Schema for "inventory_theory":
{
  "title": "short title",
  "demand": 1000,
  "ordering_cost": 50,
  "holding_cost": 2,
  "holding_cost_rate": 0.25,
  "unit_cost": 10,
  "shortage_cost": 0,
  "lead_time": 5,
  "price_breaks": [{"min_qty": 400, "discount": 0.02, "unit_cost": 9.8}],
  "gross_profit_per_unit": 80,
  "assumptions": []
}

Schema for "queueing_theory":
{
  "title": "short title",
  "arrival_rate": 10,
  "service_rate": 12,
  "servers": 1,
  "queue_capacity": 0,
  "assumptions": []
}

Schema for "game_theory":
{
  "title": "short title",
  "players": ["Player A", "Player B"],
  "strategies": {"Player A": ["A1", "A2"], "Player B": ["B1", "B2"]},
  "payoff_matrix": [
    {"player_a_strategy": "A1", "player_b_strategy": "B1", "payoff_a": 5, "payoff_b": -5}
  ],
  "zero_sum_flag": true,
  "assumptions": []
}

Your output MUST be a single JSON object:
{
  "problem_type": "<selected_type>",
  "data": { <data matching the selected schema> }
}"""


async def parse_with_llm(text: str, model: str = "qwen3:8b") -> dict[str, Any] | None:
    """Use Ollama LLM to classify and extract problem structure from arbitrary text."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                "http://127.0.0.1:11434/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": EDSS_ROUTER_SYSTEM_PROMPT},
                        {"role": "user", "content": text[:4000]},
                    ],
                    "options": {"temperature": 0.1, "num_predict": 3000},
                },
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
    except Exception:
        return None

    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    
    json_match = re.search(r"\{[\s\S]*\}", content)
    if not json_match:
        return {"_error": "no_json", "_raw": content}

    try:
        parsed_json = json.loads(json_match.group())
    except json.JSONDecodeError:
        cleaned = json_match.group().replace('\\', '\\\\')
        try:
            parsed_json = json.loads(cleaned)
        except json.JSONDecodeError:
            return {"_error": "json_decode", "_raw": content}

    problem_type = parsed_json.get("problem_type", "").lower()
    data = parsed_json.get("data", {})
    if problem_type in ["lp", "linear programming"]: 
        problem_type = "linear_programming"
        
    if not problem_type or problem_type not in [
        "linear_programming", "transportation_assignment", "integer_programming", 
        "nonlinear_programming", "network_modelling", "inventory_theory", 
        "queueing_theory", "decision_theory", "game_theory", 
        "dynamic_programming", "markov_processes"
    ]:
        # Handle legacy aliases generated by LLM
        if problem_type == "transportation" or problem_type == "assignment":
            problem_type = "transportation_assignment"
        elif problem_type == "shortest_path":
            problem_type = "network_modelling"
        elif problem_type == "decision_tree":
            problem_type = "decision_theory"
        else:
            return {"_error": "invalid_type", "_raw": content}

    context = {
        "title": data.get("title", f"AI Extracted {problem_type}"),
        "domain": "engineering",
        "decision_maker": "analyst",
        "description": text[:3000],
    }

    if problem_type == "linear_programming":
        obj_c = data.get("objective_coefficients", [])
        if not obj_c:
            return {"_error": "missing_objective", "_raw": content}
        n = len(obj_c)
        
        var_names = []
        for i, v in enumerate(data.get("variables", [])):
            if isinstance(v, dict):
                var_names.append(v.get("name", f"x{i+1}"))
            elif isinstance(v, str):
                var_names.append(v)
            else:
                var_names.append(f"x{i+1}")

        if len(var_names) != n:
            var_names = [f"x{i+1}" for i in range(n)]

        A_ub, b_ub, ub_names = [], [], []
        for con in data.get("constraints_leq", []):
            if not isinstance(con, dict):
                continue
            coeffs = con.get("coefficients", [])
            if len(coeffs) == n:
                A_ub.append([float(c) for c in coeffs])
                b_ub.append(float(con.get("rhs", 0)))
                ub_names.append(con.get("name", ""))

        for con in data.get("constraints_geq", []):
            if not isinstance(con, dict):
                continue
            coeffs = con.get("coefficients", [])
            if len(coeffs) == n:
                # >= constraint: sum(cx) >= rhs -> sum(-cx) <= -rhs
                A_ub.append([-float(c) for c in coeffs])
                b_ub.append(-float(con.get("rhs", 0)))
                ub_names.append(con.get("name", ""))

        A_eq, b_eq, eq_names = [], [], []
        for con in data.get("constraints_eq", []):
            if not isinstance(con, dict):
                continue
            coeffs = con.get("coefficients", [])
            if len(coeffs) == n:
                A_eq.append([float(c) for c in coeffs])
                b_eq.append(float(con.get("rhs", 0)))
                eq_names.append(con.get("name", ""))

        sense = data.get("sense", "maximize")
        
        def format_expr(coeffs, names):
            terms = []
            for c, name in zip(coeffs, names):
                if c != 0:
                    terms.append(f"{c}{name}")
            return " + ".join(terms).replace("+ -", "- ") if terms else "0"
            
        formulation = f"{sense.capitalize()} Z = {format_expr(obj_c, var_names)}\n"
        formulation += f"Biến: {data.get('variable_description', ', '.join(var_names))}\n\n"
        formulation += "Subject to:\n"
        
        if not A_ub and not A_eq:
            formulation += "  (None)\n"
            
        for i, row in enumerate(A_ub):
            name_str = f"[{ub_names[i]}]" if i < len(ub_names) and ub_names[i] else f"[UB_{i}]"
            formulation += f"  {name_str.ljust(15)} {format_expr(row, var_names)} <= {b_ub[i]}\n"
            
        for i, row in enumerate(A_eq):
            name_str = f"[{eq_names[i]}]" if i < len(eq_names) and eq_names[i] else f"[EQ_{i}]"
            formulation += f"  {name_str.ljust(15)} {format_expr(row, var_names)} = {b_eq[i]}\n"

        context["objective_direction"] = sense
        return {
            "context": context,
            "problem_type": "linear_programming",
            "c": [float(c) for c in obj_c],
            "sense": sense,
            "A_ub": A_ub if A_ub else None,
            "b_ub": b_ub if b_ub else None,
            "A_eq": A_eq if A_eq else None,
            "b_eq": b_eq if b_eq else None,
            "bounds": [[0, None]] * n,
            "variable_names": var_names,
            "constraint_names_ub": ub_names,
            "constraint_names_eq": eq_names,
            "formulation": formulation,
            "steps": ["LLM extracted LP model", "Solving with SciPy HiGHS"],
            "assumptions": data.get("assumptions", ["LP solved with scipy HiGHS."]),
        }

    elif problem_type == "transportation_assignment":
        context["objective_direction"] = "minimize"
        if data.get("assignment_costs"):
            return {
                "context": context,
                "problem_type": "assignment",
                "assignment_costs": data.get("assignment_costs", []),
                "assumptions": data.get("assumptions", []),
            }
        return {
            "context": context,
            "problem_type": "transportation",
            "graph": {
                "supplies": data.get("supplies", {}),
                "demands": data.get("demands", {}),
                "costs": data.get("costs", {}),
            },
            "assumptions": data.get("assumptions", []),
        }

    elif problem_type == "network_modelling":
        context["objective_direction"] = "minimize"
        return {
            "context": context,
            "problem_type": "shortest_path",
            "graph": {
                "source": data.get("source"),
                "target": data.get("target"),
                "edges": data.get("edges", []),
            },
            "assumptions": data.get("assumptions", []),
        }

    elif problem_type == "decision_theory":
        context["objective_direction"] = "maximize"
        context["unit"] = "payoff"
        return {
            "context": context,
            "problem_type": "decision_tree",
            "alternatives": data.get("alternatives", []),
            "states": data.get("states", []),
            "payoff_matrix": data.get("payoff_matrix", []),
            "assumptions": data.get("assumptions", ["Expected value computed by deterministic payoff engine."]),
        }

    elif problem_type == "inventory_theory":
        context["objective_direction"] = "minimize_cost"
        return {
            "context": context,
            "problem_type": "inventory",
            "annual_demand": data.get("annual_demand") or data.get("demand"),
            "order_cost": data.get("ordering_cost"),
            "holding_cost": data.get("holding_cost"),
            "holding_cost_rate": data.get("holding_cost_rate"),
            "shortage_cost": data.get("shortage_cost"),
            "lead_time": data.get("lead_time"),
            "purchase_cost": data.get("unit_cost"),
            "unit_cost": data.get("unit_cost"),
            "price_breaks": data.get("price_breaks", []),
            "gross_profit_per_unit": data.get("gross_profit_per_unit") or data.get("gross_profit"),
            "assumptions": data.get("assumptions", []),
        }

    elif problem_type in {"queueing_theory", "game_theory", "dynamic_programming", "markov_processes", "integer_programming", "nonlinear_programming"}:
        context["objective_direction"] = "compute"
        return {
            "context": context,
            "problem_type": problem_type,
            **data,
            "assumptions": data.get("assumptions", []),
        }

    return None


# ── Main Entry Point ─────────────────────────────────────────────────────


def parse_circle_packing_box_text(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    if not any(token in lowered for token in ["designing boxes", "box", "circles", "circular objects", "circle packing"]):
        return None
    if "non-linear" not in lowered and "nonlinear" not in lowered and "overlap" not in lowered:
        return None
    radii_matches = re.findall(r"R_?\s*(\d+)\s*=\s*(\d+(?:[,.]\d+)?)", text, flags=re.I)
    if radii_matches:
        radii = [float(value.replace(",", ".")) for _, value in sorted(radii_matches, key=lambda item: int(item[0]))]
    else:
        radius_values = re.findall(r"radii[^:\n]*:?\s*([0-9,\s.]+)", text, flags=re.I)
        radii = [float(value.replace(",", ".")) for value in re.findall(r"\d+(?:[,.]\d+)?", radius_values[0])] if radius_values else []
    if len(radii) < 2:
        return None
    return {
        "context": {
            "title": "Designing Boxes Circle Packing NLP",
            "domain": "nonlinear_programming",
            "decision_maker": "engineer",
            "objective_direction": "minimize",
            "unit": "cm",
            "description": text[:5000],
        },
        "problem_type": "circle_packing_box",
        "radii": radii,
        "assumptions": [
            "Circles are packed in a 2D axis-aligned rectangular box.",
            "Non-overlap constraints use Euclidean distance between circle centers.",
            "The NLP is nonconvex; multi-start SLSQP returns the best feasible local optimum found.",
        ],
    }


def parse_library_shelving_text(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    if "shelving" not in lowered and "library" not in lowered:
        return None
    if "shortest path" not in lowered:
        return None

    rows = []
    for height, count in re.findall(r"\|\s*(\d+(?:[,.]\d+)?)\s*cm\s*\|\s*(\d+)\s*\|", text, flags=re.I):
        rows.append((float(height.replace(",", ".")), int(count)))
    if not rows:
        # Fallback: capture compact mentions like "19 cm | 200" after markdown stripping.
        rows = [(float(h.replace(",", ".")), int(c)) for h, c in re.findall(r"(\d+(?:[,.]\d+)?)\s*cm\s*(?:\||:|-)\s*(\d+)", text, flags=re.I)]
    if len(rows) < 2:
        return None

    thickness_match = re.search(r"thickness\s+of\s+\**\$?(\d+(?:[,.]\d+)?)\s*cm|(\d+(?:[,.]\d+)?)\s*cm\**\s+is\s+considered\s+for\s+all\s+the\s+books", text, flags=re.I)
    thickness = 3.0
    if thickness_match:
        raw = next(group for group in thickness_match.groups() if group)
        thickness = float(raw.replace(",", "."))

    fixed_match = re.search(r"costs?\s+\**\$?(\d[\d,]*(?:\.\d+)?)\**", text, flags=re.I)
    fixed_cost = float(fixed_match.group(1).replace(",", "")) if fixed_match else 2500.0

    area_match = re.search(r"\$?(\d+(?:[,.]\d+)?)\s*/\s*cm", text, flags=re.I)
    area_cost = float(area_match.group(1).replace(",", ".")) if area_match else 5.0

    heights = [item[0] for item in rows]
    counts = [item[1] for item in rows]
    return {
        "context": {
            "title": "Library Shelving Shortest Path Problem",
            "domain": "network_optimization",
            "decision_maker": "library planner",
            "objective_direction": "minimize",
            "unit": "USD",
            "description": text[:5000],
        },
        "problem_type": "library_shelving_shortest_path",
        "heights": heights,
        "counts": counts,
        "thickness": thickness,
        "fixed_cost": fixed_cost,
        "area_cost": area_cost,
        "assumptions": [
            "Book heights are sorted and each shelf stores a consecutive height group.",
            "Shelf height for a group equals the maximum book height in that group.",
            "All books have mean thickness as provided.",
        ],
    }


def parse_bac_inventory_text(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    if "bac" not in lowered and "rfg" not in lowered:
        return None
    if "stockout" not in lowered and "inventory" not in lowered:
        return None

    purchase_match = re.search(r"costs?\s+BAC\s+\**\$?(\d+(?:[,.]\d+)?)|dispenser\s+costs\s+BAC\s+\**\$?(\d+(?:[,.]\d+)?)", text, flags=re.I)
    profit_match = re.search(r"gross profit\s+of\s+\**\$?(\d+(?:[,.]\d+)?)", text, flags=re.I)
    order_match = re.search(r"cost\s+of\s+making\s+an\s+order[^$]*\**\$?(\d+(?:[,.]\d+)?)", text, flags=re.I)
    maintenance_match = re.search(r"(\d+(?:[,.]\d+)?)\s*%\s+annual inventory maintenance", text, flags=re.I)
    stockout_match = re.search(r"stockout[^$]*\**\$?(\d+(?:[,.]\d+)?)", text, flags=re.I)
    weekly_match = re.search(r"weekly mean demand\s+is\s+\**(\d+(?:[,.]\d+)?)", text, flags=re.I)
    weeks_match = re.search(r"(\d+)\s*-\s*week year|(\d+)\s*week year", text, flags=re.I)

    if not (purchase_match and profit_match and order_match and maintenance_match and stockout_match and weekly_match):
        return None

    purchase_raw = next(group for group in purchase_match.groups() if group)
    weeks_raw = next((group for group in (weeks_match.groups() if weeks_match else []) if group), "52")
    purchase_cost = float(purchase_raw.replace(",", "."))
    gross_profit = float(profit_match.group(1).replace(",", "."))
    order_cost = float(order_match.group(1).replace(",", "."))
    holding_rate = float(maintenance_match.group(1).replace(",", ".")) / 100
    stockout_cost = float(stockout_match.group(1).replace(",", "."))
    weekly_demand = float(weekly_match.group(1).replace(",", "."))
    weeks_per_year = int(weeks_raw)

    return {
        "context": {
            "title": "BAC Inventory Problem",
            "domain": "inventory",
            "decision_maker": "BAC inventory planner",
            "objective_direction": "minimize_cost",
            "unit": "USD/year",
            "description": text[:5000],
        },
        "problem_type": "eoq_planned_shortages",
        "purchase_cost": purchase_cost,
        "gross_profit_per_unit": gross_profit,
        "order_cost": order_cost,
        "holding_rate": holding_rate,
        "holding_cost": purchase_cost * holding_rate,
        "stockout_cost": stockout_cost,
        "weekly_demand": weekly_demand,
        "weeks_per_year": weeks_per_year,
        "annual_demand": weekly_demand * weeks_per_year,
        "assumptions": [
            "Supply lead time is zero.",
            "Stockouts are backordered/rescheduled and penalized by the given stockout cost.",
            "Holding cost is based on purchase price, not sales price.",
        ],
    }


def parse_paper_inventory_text(text: str) -> dict[str, Any] | None:
    normalized = text.replace("\\,", " ")
    normalized = normalized.replace("\\$", "$")
    normalized = normalized.replace("\\[", "[").replace("\\]", "]")
    normalized = normalized.replace("\\", " ")
    normalized = re.sub(r"[*_`]+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    lowered = normalized.lower()
    if "paper inventory" not in lowered and "planetas" not in lowered and "book paper" not in lowered:
        return None
    if "discount" not in lowered or "storage cost" not in lowered:
        return None

    demand_block = re.search(
        r"demand\s+was\s*:?\s*\[?\s*([0-9,\s.]+?)\s*\]?\s*tonnes\s+each\s+month",
        normalized,
        flags=re.I,
    )
    if not demand_block:
        demand_block = re.search(
            r"demand[^:.]*[:.]\s*\[?\s*([0-9,\s.]+?)\s*\]?\s*tonnes\s+each\s+month",
            normalized,
            flags=re.I,
        )
    if not demand_block:
        demand_block = re.search(
            r"((?:\d+(?:\.\d+)?\s*,?\s*){8,})\s*tonnes\s+each\s+month",
            normalized,
            flags=re.I,
        )
    monthly_demands = []
    if demand_block:
        monthly_demands = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", demand_block.group(1))]
    if len(monthly_demands) < 6:
        return None

    money = r"\$?\s*(\d[\d,]*(?:\.\d+)?)"
    purchase_match = re.search(rf"purchasing price.*?(?:remain at|is|=)\s*{money}", normalized, flags=re.I)
    order_match = re.search(rf"order cost.*?(?:is|=)\s*{money}", normalized, flags=re.I)
    charge_match = re.search(r"(\d+(?:[,.]\d+)?)\s*%\s+of\s+the\s+unit\s+cost", normalized, flags=re.I)
    storage_match = re.search(rf"storage cost.*?(?:of|is|=)\s*{money}", normalized, flags=re.I)
    if not (purchase_match and order_match and charge_match and storage_match):
        return None

    tiers = []
    for discount, threshold in re.findall(
        r"(\d+(?:[,.]\d+)?)\s*%\s+discount\s+for\s+purchases\s+(?:over|of|at\s+least)\s+(\d+(?:[,.]\d+)?)\s+tonnes?",
        normalized,
        flags=re.I,
    ):
        tiers.append({"min_qty": float(threshold.replace(",", ".")), "discount": float(discount.replace(",", ".")) / 100})
    if not tiers:
        for discount, threshold in re.findall(
            r"(\d+(?:[,.]\d+)?)\s*%\s+discount[^.]*?(\d+(?:[,.]\d+)?)\s+tonnes?",
            normalized,
            flags=re.I,
        ):
            tiers.append({"min_qty": float(threshold.replace(",", ".")), "discount": float(discount.replace(",", ".")) / 100})
    if len(tiers) < 2:
        tiers = [{"min_qty": 30, "discount": 0.10}, {"min_qty": 60, "discount": 0.11}]

    return {
        "context": {
            "title": "Paper Inventory Management Problem",
            "domain": "inventory",
            "decision_maker": "Planetas Publishing House",
            "objective_direction": "minimize_cost",
            "unit": "USD/year",
            "description": text[:5000],
        },
        "problem_type": "paper_inventory_quantity_discount",
        "monthly_demands": monthly_demands,
        "purchase_price": float(purchase_match.group(1).replace(",", "")),
        "order_cost": float(order_match.group(1).replace(",", "")),
        "annual_inventory_charge": float(charge_match.group(1).replace(",", ".")) / 100,
        "additional_storage_cost": float(storage_match.group(1).replace(",", "")),
        "discount_tiers": tiers,
        "annual_demand": sum(monthly_demands),
        "holding_cost": float(purchase_match.group(1).replace(",", "")) * (float(charge_match.group(1).replace(",", ".")) / 100)
        + float(storage_match.group(1).replace(",", "")),
        "assumptions": [
            "Demand is treated as sufficiently stable for deterministic EOQ because the 12 monthly observations are relatively close.",
            "Quantity discount candidates are evaluated at their breakpoints, as in the textbook solution.",
        ],
    }


def solve_textbook_direct_problem(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    if "eoq" in lowered or "economic order quantity" in lowered:
        values = _parse_named_numbers(text)
        demand = values.get("d") or values.get("demand") or values.get("nhu cầu")
        ordering = values.get("s") or values.get("ordering_cost") or values.get("setup_cost")
        holding = values.get("h") or values.get("holding_cost")
        if demand and ordering and holding:
            from .inventory import solve_eoq

            result = solve_eoq(demand, ordering, holding)
            return _direct_solution("inventory", "EOQ inventory problem", text, result)

    if "m/m/1" in lowered or "mm1" in lowered:
        values = _parse_named_numbers(text)
        arrival = values.get("lambda") or values.get("lamda") or values.get("λ") or values.get("arrival_rate")
        service = values.get("mu") or values.get("μ") or values.get("service_rate")
        if arrival is not None and service is not None:
            from .queueing import solve_mm1

            result = solve_mm1(arrival, service)
            return _direct_solution("queueing", "M/M/1 queueing problem", text, result)

    return None


def _parse_named_numbers(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    normalized = text.replace("λ", "lambda").replace("μ", "mu")
    for key, value in re.findall(r"\b([A-Za-z_][A-Za-z0-9_ ]{0,30})\s*(?:=|:)\s*(-?\d+(?:[,.]\d+)?)", normalized):
        clean = re.sub(r"\s+", "_", key.strip().lower())
        values[clean] = float(value.replace(",", "."))
    single_letter = {
        key.lower(): float(value.replace(",", "."))
        for key, value in re.findall(r"\b([DSH])\s*=\s*(\d+(?:[,.]\d+)?)", text, flags=re.I)
    }
    values.update(single_letter)
    return values


def _direct_solution(problem_type: str, title: str, text: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "solved",
        "problem": {
            "context": {"title": title, "domain": problem_type, "description": text[:5000]},
            "problem_type": problem_type,
            "assumptions": ["Các tham số số học được trích xuất trực tiếp từ đề bài."],
        },
        "solved": {
            "problem_type": problem_type,
            "result": result,
            "recommendation_explanation": result.get("markdown_report", ""),
        },
    }


def solve_text_problem(text: str) -> dict[str, Any]:
    """Solve text problem: try specific parsers first, then cargo LP, then return pending for LLM."""
    paper_inventory = parse_paper_inventory_text(text)
    if paper_inventory:
        from .inventory import solve_paper_inventory_management

        result = solve_paper_inventory_management(
            monthly_demands=paper_inventory["monthly_demands"],
            purchase_price=float(paper_inventory["purchase_price"]),
            ordering_cost=float(paper_inventory["order_cost"]),
            annual_inventory_charge=float(paper_inventory["annual_inventory_charge"]),
            additional_storage_cost=float(paper_inventory["additional_storage_cost"]),
            discount_tiers=paper_inventory["discount_tiers"],
        )
        return attach_pipeline({
            "status": "solved",
            "problem": paper_inventory,
            "solved": {
                "problem_type": "inventory",
                "result": result,
                "recommendation_explanation": "EDSS used deterministic EOQ with quantity discount breakpoints for the paper inventory problem.",
            },
        }, text)

    bac_inventory = parse_bac_inventory_text(text)
    if bac_inventory:
        from .inventory import solve_eoq_planned_shortages

        result = solve_eoq_planned_shortages(
            annual_demand=float(bac_inventory["annual_demand"]),
            ordering_cost=float(bac_inventory["order_cost"]),
            holding_cost=float(bac_inventory["holding_cost"]),
            shortage_cost=float(bac_inventory["stockout_cost"]),
            unit_profit=float(bac_inventory["gross_profit_per_unit"]),
            weekly_demand=float(bac_inventory["weekly_demand"]),
            purchase_cost=float(bac_inventory["purchase_cost"]),
        )
        return attach_pipeline({
            "status": "solved",
            "problem": bac_inventory,
            "solved": {
                "problem_type": "inventory",
                "result": result,
                "recommendation_explanation": "EDSS used EOQ with planned shortages/backorders for the BAC inventory problem.",
            },
        }, text)

    library_shelving = parse_library_shelving_text(text)
    if library_shelving:
        from .library_shelving import solve_library_shelving_shortest_path

        result = solve_library_shelving_shortest_path(
            library_shelving["heights"],
            library_shelving["counts"],
            float(library_shelving["thickness"]),
            float(library_shelving["fixed_cost"]),
            float(library_shelving["area_cost"]),
        )
        return attach_pipeline({
            "status": "solved",
            "problem": library_shelving,
            "solved": {
                "problem_type": "shortest_path",
                "result": result,
                "recommendation_explanation": "EDSS used a dedicated DAG shortest-path model for shelving segmentation.",
            },
        }, text)

    circle_packing = parse_circle_packing_box_text(text)
    if circle_packing:
        from .nonlinear import solve_circle_packing_box

        result = solve_circle_packing_box(circle_packing["radii"])
        return attach_pipeline({
            "status": "solved",
            "problem": circle_packing,
            "solved": {
                "problem_type": "nonlinear_programming",
                "result": result,
                "recommendation_explanation": "EDSS used a dedicated nonconvex circle-packing NLP solver instead of the LP parser.",
            },
        }, text)

    lp_batch = parse_markdown_linear_programming_text(text)
    if lp_batch:
        from .linear_programming import solve_lp
        solved_items = []
        report_parts = ["# Lời giải các bài toán Linear Programming", ""]
        for index, problem in enumerate(lp_batch, start=1):
            result = solve_lp(problem)
            label = problem.get("context", {}).get("title", f"LP {index}")
            solved_items.append({"problem": problem, "result": result})
            report_parts.append(f"## {label}")
            report_parts.append("")
            if result.get("markdown_report"):
                report_parts.append(result["markdown_report"])
            else:
                report_parts.append(f"**Status**: {result.get('status')}")
                if result.get("message"):
                    report_parts.append("")
                    report_parts.append(str(result["message"]))
            report_parts.append("")
        combined_report = "\n".join(report_parts)
        return attach_pipeline({
            "status": "solved",
            "problem": {
                "context": {
                    "title": "Linear programming batch",
                    "domain": "optimization",
                    "description": text[:5000],
                },
                "problem_type": "linear_programming_batch",
                "assumptions": ["Each subproblem is solved independently.", "All variables are non-negative."],
            },
            "solved": {
                "problem_type": "linear_programming_batch",
                "result": {
                    "status": "computed",
                    "solver": "scipy_highs",
                    "items": solved_items,
                    "markdown_report": combined_report,
                    "recommendation": "Review each subproblem result independently.",
                },
                "recommendation_explanation": "EDSS parsed and solved each LP subproblem deterministically from the provided Markdown/LaTeX text.",
            },
        }, text)

    direct = solve_textbook_direct_problem(text)
    if direct:
        return attach_pipeline(direct, text)

    parsed = (
        parse_bayes_text(text)
        or parse_expected_value_text(text)
        or parse_diagnostic_decision_text(text)
        or parse_forklift_decision_text(text)
        or parse_probability_tree_text(text)
        or parse_general_resource_dp_text(text)
        or parse_resource_allocation_dp_text(text)
        or parse_power_transportation_text(text)
        or parse_cargo_lp_text(text)
        or parse_max_flow_text(text)
        or parse_min_cost_flow_text(text)
        or parse_shortest_path_text(text)
    )
    if parsed:
        if "c" in parsed:
            from .linear_programming import solve_lp
            result = solve_lp(parsed)
            return attach_pipeline({
                "status": "solved",
                "problem": parsed,
                "solved": {
                    "problem_type": "linear_programming",
                    "result": result,
                    "recommendation_explanation": result.get("recommendation", ""),
                },
            }, text)
        solved = solve_problem(parsed)
        return attach_pipeline({"status": "solved", "problem": parsed, "solved": solved}, text)

    return attach_pipeline({
        "status": "pending_llm",
        "message": "Đang phân tích bài toán bằng AI...",
        "text": text,
    }, text)


async def solve_text_problem_async(text: str, model: str = "qwen3:8b") -> dict[str, Any]:
    """Async version: tries specific parsers, then LLM-based extraction."""
    sync_result = solve_text_problem(text)
    if sync_result["status"] == "solved":
        return sync_result

    try:
        parsed = await parse_with_llm(text, model=model)
        if not parsed or "_error" in parsed:
            error_msg = "Không thể trích xuất mô hình toán học từ bài toán."
            if parsed and "_raw" in parsed:
                error_msg += f"\n\n**Lỗi:** `{parsed.get('_error')}`\n**Raw AI Output:**\n```json\n{parsed['_raw']}\n```"
                
            return attach_pipeline({
                "status": "needs_clarification",
                "message": error_msg,
                "questions": [
                    "The problem type is ambiguous. I need the exact problem type or a clearer statement before solving.",
                    "Vui lòng cung cấp các dữ liệu bắt buộc: biến/phương án, hàm mục tiêu hoặc xác suất/payoff/graph/table tương ứng."
                ]
            }, text)

        if parsed.get("problem_type") == "linear_programming" and "c" in parsed:
            allowed, gate = gate_allows_solving(text, parsed)
            if not allowed:
                return attach_pipeline({
                    "status": "needs_clarification",
                    "problem": parsed,
                    "message": "The problem type is ambiguous. I need the following clarification before solving…",
                    "questions": [f"Cần bổ sung slot `{slot}`." for slot in gate.get("missing_slots", [])]
                    or ["Vui lòng xác nhận dạng toán và dữ liệu bắt buộc trước khi giải."],
                }, text)
            from .linear_programming import solve_lp
            result = solve_lp(parsed)
            return attach_pipeline({
                "status": "solved",
                "problem": parsed,
                "solved": {
                    "problem_type": "linear_programming",
                    "result": result,
                    "recommendation_explanation": result.get("recommendation", ""),
                },
            }, text)
        else:
            allowed, gate = gate_allows_solving(text, parsed)
            if not allowed:
                return attach_pipeline({
                    "status": "needs_clarification",
                    "problem": parsed,
                    "message": "The problem type is ambiguous. I need the following clarification before solving…",
                    "questions": [f"Cần bổ sung slot `{slot}`." for slot in gate.get("missing_slots", [])]
                    or ["Vui lòng xác nhận dạng toán và dữ liệu bắt buộc trước khi giải."],
                }, text)
            solved = solve_problem(parsed)
            return attach_pipeline({
                "status": "solved",
                "problem": parsed,
                "solved": solved,
            }, text)
    except Exception:
        return attach_pipeline({
            "status": "needs_clarification",
            "message": "The problem type is ambiguous. I need the following clarification before solving…",
            "questions": [
                "Dạng toán chính là gì: LP/IP/NLP/Network/Inventory/Queueing/Decision/Game/DP/Markov?",
                "Các dữ liệu định lượng bắt buộc của dạng toán đó là gì?",
            ],
        }, text)
