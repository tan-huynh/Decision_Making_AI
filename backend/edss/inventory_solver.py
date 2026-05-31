from __future__ import annotations

from typing import Any

from .inventory import solve_inventory_problem, solve_reorder_point


def recognize_inventory_theory(problem: dict[str, Any]) -> dict[str, Any]:
    description = problem.get("context", {}).get("description", "") or ""
    lowered = description.lower()
    evidence: list[str] = []
    missing: list[str] = []

    demand = _num(problem.get("annual_demand"), problem.get("demand"))
    if demand is None and problem.get("weekly_demand") is not None:
        demand = float(problem["weekly_demand"]) * 52
        evidence.append("Weekly demand converted to annual demand.")
    ordering = _num(problem.get("order_cost"), problem.get("ordering_cost"))
    unit_cost = _num(problem.get("purchase_cost"), problem.get("unit_cost"))
    holding = _num(problem.get("holding_cost"))
    holding_rate = _num(problem.get("holding_cost_rate"))
    shortage = _num(problem.get("shortage_cost"))
    lead_time = _num(problem.get("lead_time"))
    price_breaks = problem.get("price_breaks") or []

    if holding_rate and holding_rate > 1:
        holding_rate = holding_rate / 100
    holding_rule = "fixed_per_unit_per_year"
    if holding is None and holding_rate is not None and unit_cost is not None:
        holding = holding_rate * unit_cost
        holding_rule = "percentage_of_unit_cost"
        evidence.append("Holding cost computed from holding rate and unit purchase cost.")
    elif holding_rate is not None and unit_cost is not None:
        holding_rule = "percentage_plus_fixed" if holding is not None else "percentage_of_unit_cost"

    if demand is not None:
        evidence.append("Có deterministic demand D.")
    if ordering is not None:
        evidence.append("Có ordering/setup cost K.")
    if holding is not None:
        evidence.append("Có holding cost h hoặc tính được h.")
    if price_breaks:
        evidence.append("Có quantity discount/price break table.")
    if shortage and shortage > 0:
        evidence.append("Có shortage/backorder cost.")
    if lead_time is not None:
        evidence.append("Có lead time để tính reorder point.")
    if any(word in lowered for word in ["eoq", "inventory", "order quantity", "reorder", "stockout", "holding cost", "quantity discount"]):
        evidence.append("Có keyword Inventory Theory.")

    if price_breaks:
        subtype = "INV_EOQ_WITH_QUANTITY_DISCOUNT"
    elif shortage and shortage > 0:
        subtype = "INV_EOQ_WITH_BACKORDERS"
    elif lead_time is not None:
        subtype = "INV_EOQ_WITH_LEAD_TIME"
    else:
        subtype = "INV_BASIC_EOQ"

    if demand is None or demand <= 0:
        missing.append("demand_D")
    if ordering is None or ordering <= 0:
        missing.append("ordering_cost_K")
    if holding is None or holding <= 0:
        if holding_rate is not None and unit_cost is None:
            missing.append("unit_purchase_cost_c_for_holding_rate")
        else:
            missing.append("holding_cost_h")
    if price_breaks and unit_cost is None and any("unit_cost" not in pb for pb in price_breaks):
        missing.append("unit_purchase_cost_c_or_discount_unit_prices")
    if subtype == "INV_EOQ_WITH_BACKORDERS" and (shortage is None or shortage <= 0):
        missing.append("shortage_cost_pi")

    confidence = 0.35 + 0.15 * (demand is not None) + 0.15 * (ordering is not None) + 0.15 * (holding is not None) + 0.15 * bool(evidence)
    if price_breaks or shortage or lead_time is not None:
        confidence += 0.1
    return {
        "problem_type": "Inventory Theory",
        "subtype": subtype,
        "confidence": round(min(confidence, 0.99), 2),
        "evidence": evidence,
        "demand_D": demand,
        "demand_unit": "units/year",
        "ordering_cost_K": ordering,
        "unit_purchase_cost_c": unit_cost,
        "holding_cost_h": holding,
        "holding_cost_rule": holding_rule if holding is not None else "unknown",
        "lead_time_L": lead_time,
        "lead_time_unit": problem.get("lead_time_unit", "unknown" if lead_time is None else "years"),
        "shortage_allowed": bool(shortage and shortage > 0),
        "shortage_cost_pi": shortage,
        "discount_table": price_breaks,
        "perishability_or_expiry_limit": problem.get("expiry_limit") or problem.get("shelf_life"),
        "current_policy": problem.get("current_policy", {}),
        "objective": _objective(problem, price_breaks, shortage),
        "requested_outputs": problem.get("requested_outputs", ["Q*", "orders_per_year", "cycle_time", "total_cost"]),
        "missing_information": missing,
        "can_solve": confidence >= 0.85 and not missing,
    }


def solve_inventory_theory_problem(problem: dict[str, Any]) -> dict[str, Any]:
    recognition = recognize_inventory_theory(problem)
    gate_md = _recognition_markdown(recognition)
    if not recognition["can_solve"]:
        return {
            "status": "needs_clarification",
            "solver": "inventory_recognition_gate",
            "recognition": recognition,
            "missing_data": recognition["missing_information"],
            "markdown_report": gate_md + "\nChưa đủ dữ liệu để chọn mô hình tồn kho và tính chính sách tối ưu.\n",
        }

    normalized = dict(problem)
    normalized["annual_demand"] = recognition["demand_D"]
    normalized["order_cost"] = recognition["ordering_cost_K"]
    normalized["holding_cost"] = recognition["holding_cost_h"]
    if recognition["unit_purchase_cost_c"] is not None:
        normalized["purchase_cost"] = recognition["unit_purchase_cost_c"]
    result = solve_inventory_problem(normalized)
    if recognition["lead_time_L"] is not None and result.get("status") == "optimal":
        demand_rate = float(recognition["demand_D"])
        rop = solve_reorder_point(demand_rate=demand_rate, lead_time=float(recognition["lead_time_L"]))
        result["reorder_point"] = rop["reorder_point"]
    verification = verify_inventory_solution(recognition, result)
    result["recognition"] = recognition
    result["verification"] = verification
    result["markdown_report"] = gate_md + "\n" + result.get("markdown_report", "") + _verification_markdown(verification)
    return result


def verify_inventory_solution(recognition: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    checks: list[str] = []
    passed = True
    q = result.get("order_quantity") or result.get("best", {}).get("candidate_quantity")
    if q is not None and float(q) > 0:
        checks.append("Q* > 0.")
    elif result.get("status") == "optimal":
        passed = False
        checks.append("Không tìm thấy order quantity dương trong nghiệm.")
    if result.get("model") == "EOQ":
        ordering = float(result.get("ordering_cost", 0))
        holding = float(result.get("holding_cost", 0))
        if abs(ordering - holding) <= max(1e-6, 1e-5 * max(abs(ordering), abs(holding), 1)):
            checks.append("Basic EOQ check passed: ordering cost ≈ holding cost.")
        else:
            passed = False
            checks.append("Basic EOQ check failed: ordering cost không xấp xỉ holding cost.")
    if result.get("model") == "quantity_discount_eoq":
        best = result.get("best", {})
        if best and float(best.get("candidate_quantity", 0)) >= float(best.get("min_qty", 0)):
            checks.append("Discount feasibility passed: selected Q satisfies tier minimum.")
        else:
            passed = False
            checks.append("Discount feasibility failed.")
    if result.get("model") == "EOQ_with_planned_shortages":
        q_val = float(result.get("order_quantity", 0))
        total = float(result.get("maximum_inventory", 0)) + float(result.get("maximum_shortage", 0))
        if abs(total - q_val) <= 1e-5:
            checks.append("Backorder check passed: Imax + Bmax = Q.")
        else:
            passed = False
            checks.append("Backorder check failed: Imax + Bmax != Q.")
    checks.append(f"Holding cost rule: {recognition['holding_cost_rule']}.")
    return {"passed": passed, "checks": checks}


def _num(*values: Any) -> float | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _objective(problem: dict[str, Any], price_breaks: list[dict[str, Any]], shortage: float | None) -> str:
    if problem.get("current_policy"):
        return "compare_policies"
    if problem.get("gross_profit_per_unit") or problem.get("unit_profit"):
        return "maximize_profit"
    return "minimize_total_cost" if price_breaks or shortage else "find_order_policy"


def _recognition_markdown(r: dict[str, Any]) -> str:
    evidence = "\n".join(f"- {item}" for item in r["evidence"]) or "- Chưa có dấu hiệu đủ mạnh."
    missing = "\n".join(f"- {item}" for item in r["missing_information"]) or "- Không thiếu dữ liệu bắt buộc."
    return (
        "# Lời giải\n\n"
        "## 1. Nhận dạng dạng toán\n\n"
        f"- Dạng toán chính: {r['problem_type']}\n"
        f"- Dạng toán phụ: {r['subtype']}\n"
        "- Demand type: deterministic\n"
        "- Review type: continuous\n"
        f"- Stockout allowed: {'yes' if r['shortage_allowed'] else 'no'}\n"
        f"- Discount: {'yes' if r['discount_table'] else 'no'}\n"
        f"- Lead time: {'non-zero' if r['lead_time_L'] is not None else 'zero/not requested'}\n"
        f"- Objective: {r['objective']}\n"
        f"- Mức tin cậy: {r['confidence']:.2f}\n\n"
        "Evidence:\n"
        f"{evidence}\n\n"
        "Missing information:\n"
        f"{missing}\n\n"
        "## 2. Dữ liệu đã trích xuất và chuẩn hóa đơn vị\n\n"
        "| Parameter | Meaning | Original value | Converted value |\n|---|---|---:|---:|\n"
        f"| D | annual demand | {r['demand_D']} | {r['demand_D']} units/year |\n"
        f"| K | ordering cost | {r['ordering_cost_K']} | {r['ordering_cost_K']} per order |\n"
        f"| c | unit purchase cost | {r['unit_purchase_cost_c']} | {r['unit_purchase_cost_c']} |\n"
        f"| h | holding cost | {r['holding_cost_h']} | {r['holding_cost_h']} per unit/year |\n"
        f"| pi | shortage cost | {r['shortage_cost_pi']} | {r['shortage_cost_pi']} |\n\n"
    )


def _verification_markdown(v: dict[str, Any]) -> str:
    checks = "\n".join(f"- {item}" for item in v.get("checks", []))
    return f"\n## 6. Kiểm tra nghiệm\n\n- Trạng thái kiểm tra: {'passed' if v.get('passed') else 'failed'}\n{checks}\n"
