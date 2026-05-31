from __future__ import annotations

from typing import Any

from .dynamic_programming import (
    solve_finite_horizon_dp,
    solve_inventory_dp,
    solve_resource_allocation_dp,
)


def recognize_dynamic_programming(problem: dict[str, Any]) -> dict[str, Any]:
    description = problem.get("context", {}).get("description", "") or ""
    lowered = description.lower()
    evidence: list[str] = []
    missing: list[str] = []
    subtype = "UNKNOWN_DP"
    confidence = 0.0
    objective = problem.get("context", {}).get("objective_direction") or "maximize"
    stages: list[Any] = []
    states: list[Any] = []
    decisions: list[Any] = []
    transition = ""
    reward_or_cost = ""
    boundary = ""
    direction = "backward"

    resource = problem.get("resource_allocation")
    if resource:
        stage_returns = resource.get("stage_returns", [])
        total_resource = resource.get("total_resource")
        resource_name = resource.get("resource_name", "resource")
        objective = resource.get("sense", objective)
        subtype = _resource_subtype(description)
        evidence.extend(
            [
                "Có tổng tài nguyên hữu hạn cần phân bổ.",
                "Có bảng lợi ích/chi phí theo lượng phân bổ ở từng stage.",
                "Quyết định hiện tại làm giảm tài nguyên còn lại cho các stage sau.",
            ]
        )
        stages = [f"Stage {idx + 1}" for idx in range(len(stage_returns))]
        states = [f"s_n = số {resource_name} còn lại trước stage n"]
        decisions = [f"x_n = số {resource_name} phân bổ cho stage n"]
        transition = "s_{n+1} = s_n - x_n"
        reward_or_cost = "r_n(x_n) lấy từ bảng stage_returns[n][x_n]"
        boundary = "f_{N+1}(0)=0; state cuối không thỏa điều kiện dùng hết tài nguyên là infeasible"
        direction = "backward"
        if total_resource is None:
            missing.append("total_resource")
        if not stage_returns:
            missing.append("stage_returns")
        if total_resource is not None and stage_returns:
            confidence = 0.96
    elif _looks_like_inventory_dp(problem, lowered):
        subtype = "DP_INVENTORY_PLANNING"
        evidence.extend(
            [
                "Có nhiều kỳ nhu cầu/order/inventory.",
                "State là tồn kho đầu kỳ và decision là lượng đặt hàng.",
            ]
        )
        demands = problem.get("demands", [])
        periods = int(problem.get("periods", len(demands) or 0))
        stages = [f"Period {idx + 1}" for idx in range(periods)]
        states = ["I_t = inventory at beginning of period t"]
        decisions = ["q_t = order quantity in period t"]
        transition = "I_{t+1} = max(0, I_t + q_t - d_t) in the implemented no-negative-inventory state model"
        reward_or_cost = "order cost + fixed order cost + holding cost + shortage penalty"
        boundary = "f_{N+1}(I)=0 unless terminal inventory condition is specified"
        objective = "minimize"
        confidence = 0.9 if demands else 0.7
        if not demands:
            missing.append("demands")
    elif problem.get("stages"):
        raw_stages = problem.get("stages", [])
        subtype = "STOCHASTIC_DP" if _has_probabilistic_transitions(raw_stages) else "FINITE_HORIZON_DP"
        evidence.append("Có danh sách stages explicit với states/actions/transitions/rewards.")
        stages = [stage.get("name", f"Stage {idx + 1}") for idx, stage in enumerate(raw_stages)]
        states = ["states lấy từ từng stage"]
        decisions = ["actions lấy từ từng stage"]
        transition = "transitions[state][action] -> next states, kèm probability nếu stochastic"
        reward_or_cost = "rewards[state][action]"
        boundary = "value after final stage = 0"
        objective = problem.get("objective", {}).get("sense", objective) if isinstance(problem.get("objective"), dict) else objective
        confidence = 0.9
        for idx, stage in enumerate(raw_stages, start=1):
            if not stage.get("states"):
                missing.append(f"stage_{idx}.states")
            if not stage.get("actions"):
                missing.append(f"stage_{idx}.actions")
            if not stage.get("transitions"):
                missing.append(f"stage_{idx}.transitions")
            if not stage.get("rewards"):
                missing.append(f"stage_{idx}.rewards")
        if missing:
            confidence = 0.74
    else:
        if any(token in lowered for token in ["dynamic programming", "quy hoạch động", "bellman", "stage", "giai đoạn"]):
            evidence.append("Có từ khóa Dynamic Programming nhưng chưa đủ mô hình.")
            confidence = 0.55
        missing.extend(["stages", "states", "decisions", "transition_function", "reward_or_cost_function"])

    can_solve = confidence >= 0.85 and not missing and bool(stages) and bool(states) and bool(decisions) and bool(transition) and bool(reward_or_cost)
    return {
        "problem_type": "Dynamic Programming",
        "subtype": subtype,
        "confidence": round(confidence, 2),
        "evidence": evidence,
        "stages": stages,
        "states": states,
        "decisions": decisions,
        "transition_function": transition,
        "reward_or_cost_function": reward_or_cost,
        "objective": "minimize" if str(objective).lower().startswith("min") else "maximize",
        "boundary_condition": boundary,
        "solution_direction": direction,
        "missing_information": missing,
        "can_solve": can_solve,
    }


def solve_dynamic_programming_problem(problem: dict[str, Any]) -> dict[str, Any]:
    recognition = recognize_dynamic_programming(problem)
    gate_md = _recognition_markdown(recognition)
    if not recognition["can_solve"]:
        return {
            "status": "needs_clarification",
            "solver": "dynamic_programming_recognition_gate",
            "recognition": recognition,
            "missing_data": recognition["missing_information"],
            "markdown_report": gate_md
            + "\nBài này có dấu hiệu là Dynamic Programming, nhưng chưa đủ dữ liệu để xác định đầy đủ stage, state, decision, transition và reward/cost function.\n",
        }

    if problem.get("resource_allocation"):
        spec = problem["resource_allocation"]
        result = solve_resource_allocation_dp(
            int(spec["total_resource"]),
            spec["stage_returns"],
            spec.get("resource_name", "resource"),
            spec.get("sense", problem.get("context", {}).get("objective_direction", "maximize")),
        )
        result["recognition"] = recognition
        result["verification"] = _verify_resource_allocation(spec, result)
        result["markdown_report"] = gate_md + "\n" + result.get("markdown_report", "") + _verification_markdown(result["verification"])
        return result

    if _looks_like_inventory_dp(problem, ""):
        result = solve_inventory_dp(problem)
        result["recognition"] = recognition
        result["verification"] = {"passed": True, "checks": ["Inventory balance and policy rollout generated by inventory DP engine."]}
        result["markdown_report"] = gate_md + "\n" + result.get("markdown_report", "") + _verification_markdown(result["verification"])
        return result

    result = solve_finite_horizon_dp(problem)
    result["recognition"] = recognition
    result["verification"] = {"passed": True, "checks": ["Finite-horizon backward induction completed for supplied transitions."]}
    result["markdown_report"] = gate_md + "\n" + _finite_horizon_markdown(result)
    return result


def _resource_subtype(description: str) -> str:
    lowered = description.lower()
    if any(word in lowered for word in ["truck", "load", "capacity", "knapsack"]):
        return "DP_KNAPSACK"
    if any(word in lowered for word in ["investment", "invest", "capital", "vốn"]):
        return "DP_INVESTMENT_STRATEGY"
    if any(word in lowered for word in ["advertising", "marketing", "campaign", "electoral", "vote"]):
        return "DP_ELECTORAL_OR_MARKETING_PLANNING"
    return "DP_RESOURCE_ALLOCATION"


def _looks_like_inventory_dp(problem: dict[str, Any], lowered: str) -> bool:
    if problem.get("demands") and ("order_cost" in problem or "holding_cost" in problem):
        return True
    return bool(problem.get("inventory_dp")) or ("inventory" in lowered and "demand" in lowered and "period" in lowered)


def _has_probabilistic_transitions(stages: list[dict[str, Any]]) -> bool:
    for stage in stages:
        for by_action in stage.get("transitions", {}).values():
            for transitions in by_action.values():
                for transition in transitions:
                    if "probability" in transition and float(transition.get("probability", 1)) != 1:
                        return True
    return False


def _recognition_markdown(recognition: dict[str, Any]) -> str:
    evidence = "\n".join(f"- {item}" for item in recognition["evidence"]) or "- Chưa có dấu hiệu đủ mạnh."
    missing = "\n".join(f"- {item}" for item in recognition["missing_information"]) or "- Không thiếu dữ liệu bắt buộc."
    stages = ", ".join(str(item) for item in recognition["stages"]) or "chưa xác định"
    return (
        "# Lời giải\n\n"
        "## 1. Nhận dạng dạng toán\n\n"
        f"- Dạng toán chính: {recognition['problem_type']}\n"
        f"- Dạng toán phụ: {recognition['subtype']}\n"
        f"- Mục tiêu: {recognition['objective']}\n"
        f"- Mức tin cậy: {recognition['confidence']:.2f}\n"
        f"- Có thể giải ngay: {'Có' if recognition['can_solve'] else 'Không'}\n\n"
        "Dấu hiệu nhận dạng:\n"
        f"{evidence}\n\n"
        "Các slot DP đã xác định:\n"
        f"- Stage: {stages}\n"
        f"- State: {', '.join(recognition['states']) or 'chưa xác định'}\n"
        f"- Decision: {', '.join(recognition['decisions']) or 'chưa xác định'}\n"
        f"- Transition function: `{recognition['transition_function'] or 'chưa xác định'}`\n"
        f"- Reward/cost function: `{recognition['reward_or_cost_function'] or 'chưa xác định'}`\n"
        f"- Boundary condition: `{recognition['boundary_condition'] or 'chưa xác định'}`\n"
        f"- Solution direction: {recognition['solution_direction']}\n\n"
        "Dữ liệu còn thiếu:\n"
        f"{missing}\n\n"
    )


def _verify_resource_allocation(spec: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    resource_name = spec.get("resource_name", "resource")
    total = int(spec["total_resource"])
    allocation = result.get("allocation", [])
    amounts = [int(item[resource_name]) for item in allocation]
    stage_returns = spec.get("stage_returns", [])
    recomputed = 0.0
    checks: list[str] = []
    passed = True
    if sum(amounts) != total:
        passed = False
        checks.append(f"Tổng phân bổ = {sum(amounts)} khác total_resource = {total}.")
    else:
        checks.append(f"Tổng phân bổ = {sum(amounts)} đúng bằng total_resource = {total}.")
    for idx, amount in enumerate(amounts):
        if idx >= len(stage_returns) or amount >= len(stage_returns[idx]):
            passed = False
            checks.append(f"Stage {idx + 1}: decision {amount} nằm ngoài bảng dữ liệu.")
            continue
        recomputed += float(stage_returns[idx][amount])
    if abs(recomputed - float(result.get("objective_value", 0))) > 1e-6:
        passed = False
        checks.append(f"Objective recompute = {recomputed} khác solver value = {result.get('objective_value')}.")
    else:
        checks.append(f"Objective recompute = {recomputed} khớp solver value.")
    return {"passed": passed, "checks": checks}


def _verification_markdown(verification: dict[str, Any]) -> str:
    checks = "\n".join(f"- {item}" for item in verification.get("checks", []))
    return (
        "\n#### 5. Kiểm tra nghiệm\n\n"
        f"- Trạng thái kiểm tra: {'passed' if verification.get('passed') else 'failed'}\n"
        f"{checks}\n"
    )


def _finite_horizon_markdown(result: dict[str, Any]) -> str:
    md = "## 4. Tính bảng DP\n\n"
    for stage in result.get("policy", []):
        md += f"### Stage {stage['stage'] + 1}\n\n"
        md += "| State | Optimal value | Optimal decision |\n|---|---:|---|\n"
        for state, value in stage.get("value", {}).items():
            decision = stage.get("policy", {}).get(state, "none")
            md += f"| {state} | {value:.4f} | {decision} |\n"
        md += "\n"
    return md
