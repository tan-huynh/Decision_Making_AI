from __future__ import annotations

from typing import Any

from .uncertainty import (
    rollback_decision_tree,
    solve_bayes_problem,
    solve_binary_event_tree,
    solve_diagnostic_decision_tree,
    solve_forklift_decision_tree,
    solve_imperfect_information_decision_tree,
    solve_independent_probability,
    value_of_information,
)


def recognize_decision_theory(problem: dict[str, Any]) -> dict[str, Any]:
    description = problem.get("context", {}).get("description", "") or ""
    lowered = description.lower()
    evidence: list[str] = []
    missing: list[str] = []
    subtype = "DT_EMV_BASIC"
    confidence = 0.0
    decision_maker = _infer_decision_maker(description)
    alternatives = [alt.get("name", str(alt)) for alt in problem.get("alternatives", [])]
    states = [state.get("name", str(state)) for state in problem.get("states", [])]
    probabilities = [state.get("probability") for state in problem.get("states", []) if "probability" in state]
    payoffs = problem.get("payoff_matrix", [])
    information_options: list[str] = []
    test_accuracy_data: list[Any] = []
    objective = _objective(problem)

    if problem.get("game"):
        return _not_decision_theory("Game Theory data detected: players/strategies/payoff matrix.", ["game_theory_agent"])
    if problem.get("resource_allocation") or (problem.get("problem_type") == "dynamic_programming"):
        return _not_decision_theory("Dynamic Programming data detected: stages/state transitions.", ["dynamic_programming_agent"])

    if problem.get("imperfect_information_decision"):
        spec = problem["imperfect_information_decision"]
        subtype = "DT_IMPERFECT_INFORMATION"
        evidence.extend(["Có prior states, information/test signals và likelihood P(result|state).", "Cần Bayes posterior rồi rollback decision tree."])
        decision_maker = decision_maker or spec.get("decision_maker", "decision maker")
        alternatives = [action.get("name", "") for action in spec.get("actions", [])]
        states = [state.get("name", "") for state in spec.get("states", [])]
        probabilities = [state.get("probability") for state in spec.get("states", [])]
        information_options = [spec.get("information_name", "sample information")]
        test_accuracy_data = spec.get("signals", [])
        objective = "minimize cost" if str(spec.get("objective", "")).lower().startswith("min") else "maximize benefit"
        if not states:
            missing.append("states_of_nature")
        if not alternatives:
            missing.append("alternatives")
        if not test_accuracy_data:
            missing.append("conditional_probabilities_or_test_accuracy")
        if not spec.get("actions"):
            missing.append("payoffs_or_costs")
        confidence = 0.97
    elif problem.get("diagnostic_decision"):
        spec = problem["diagnostic_decision"]
        subtype = "DT_DIAGNOSTIC_TEST"
        evidence.extend(["Có diagnostic/test option với sensitivity hoặc false-positive rate.", "Cần phân biệt likelihood với posterior bằng Bayes."])
        decision_maker = decision_maker or spec.get("decision_maker", "decision maker")
        alternatives = list(spec.get("actions", {}).keys())
        state_spec = spec.get("states", {})
        states = [state_spec.get("true", {}).get("label", "true"), state_spec.get("false", {}).get("label", "false")]
        probabilities = [state_spec.get("true", {}).get("probability")]
        information_options = [spec.get("test", {}).get("name", "diagnostic test")]
        test_accuracy_data = [spec.get("test", {})]
        if not alternatives:
            missing.append("alternatives")
        if not spec.get("test"):
            missing.append("test_accuracy_data")
        confidence = 0.94
    elif problem.get("forklift_decision"):
        spec = problem["forklift_decision"]
        subtype = "DT_DECISION_TREE"
        evidence.extend(["Có nhiều quyết định theo trình tự và chi phí theo nhánh.", "Mục tiêu là minimize expected cost."])
        decision_maker = decision_maker or "firm"
        alternatives = ["buy_new", "buy_used", "test_used"]
        states = ["used_good", "used_faulty"]
        probabilities = [spec.get("prior_good")]
        information_options = ["test_used"]
        test_accuracy_data = [spec.get("test", {})]
        objective = "minimize cost"
        confidence = 0.94
    elif problem.get("bayes"):
        subtype = "DT_BAYES_INFORMATION"
        evidence.append("Có prior và test accuracy; yêu cầu cập nhật Bayes.")
        decision_maker = decision_maker or "analyst"
        states = ["hypothesis true", "hypothesis false"]
        probabilities = [problem["bayes"].get("prior")]
        test_accuracy_data = [problem["bayes"]]
        objective = "posterior probability update"
        confidence = 0.9
    elif problem.get("decision_tree"):
        subtype = "DT_DECISION_TREE"
        evidence.append("Có decision/chance/terminal nodes để rollback.")
        decision_maker = decision_maker or "decision maker"
        alternatives = _root_decision_labels(problem.get("decision_tree", []))
        objective = "minimize cost" if _objective(problem).startswith("minimize") else "maximize benefit"
        confidence = 0.9
        if not alternatives:
            missing.append("alternatives_at_root_decision")
    elif problem.get("probability_tree") or problem.get("independent_probabilities"):
        subtype = "PROBABILITY_TREE_ONLY"
        evidence.append("Đây là cây xác suất/chance calculation, chưa phải quyết định tối ưu.")
        decision_maker = "none"
        objective = "compute probabilities"
        confidence = 0.86
    else:
        if alternatives:
            evidence.append("Có alternatives.")
        if states:
            evidence.append("Có states of nature.")
        if probabilities:
            evidence.append("Có probabilities cho states.")
        if payoffs:
            evidence.append("Có payoff/cost matrix.")
        if any(token in lowered for token in ["decision tree", "emv", "evpi", "evsi", "bayes", "market research", "pilot", "test", "survey"]):
            evidence.append("Có keyword Decision Theory.")
        if _matrix_is_cost_oriented(payoffs, objective):
            subtype = "DT_COST_MINIMIZATION"
            objective = "minimize cost"
        elif not probabilities and states and payoffs:
            subtype = "DT_UNCERTAINTY_NO_PROBABILITY"
        confidence = 0.35 + 0.15 * bool(alternatives) + 0.15 * bool(states) + 0.15 * bool(payoffs) + 0.1 * bool(probabilities)
        if any(token in lowered for token in ["emv", "expected monetary", "decision tree", "evpi", "regret"]):
            confidence += 0.1
        if not decision_maker:
            missing.append("decision_maker")
        if not alternatives:
            missing.append("alternatives")
        if not states:
            missing.append("states_of_nature")
        if not payoffs:
            missing.append("payoffs_or_costs")
        if subtype == "DT_EMV_BASIC" and states and probabilities and len(probabilities) != len(states):
            missing.append("state_probabilities")
        if subtype == "DT_EMV_BASIC" and states and not probabilities:
            missing.append("state_probabilities")
        missing.extend(_missing_payoff_pairs(alternatives, states, payoffs))

    can_solve = confidence >= 0.85 and not missing
    if subtype in {"DT_BAYES_INFORMATION", "PROBABILITY_TREE_ONLY"}:
        can_solve = confidence >= 0.85 and not missing
    return {
        "problem_type": "Decision Theory",
        "subtype": subtype,
        "confidence": round(min(confidence, 0.99), 2),
        "evidence": evidence,
        "decision_maker": decision_maker or "",
        "alternatives": alternatives,
        "states_of_nature": states,
        "probabilities": probabilities,
        "payoffs_or_costs": payoffs,
        "information_options": information_options,
        "test_accuracy_data": test_accuracy_data,
        "objective": objective,
        "missing_information": missing,
        "can_solve": can_solve,
    }


def solve_decision_theory_problem(problem: dict[str, Any]) -> dict[str, Any]:
    recognition = recognize_decision_theory(problem)
    gate_md = _recognition_markdown(recognition)
    if not recognition["can_solve"]:
        return {
            "status": "needs_clarification",
            "solver": "decision_theory_recognition_gate",
            "recognition": recognition,
            "missing_data": recognition["missing_information"],
            "markdown_report": gate_md
            + "\nBài này có dấu hiệu Decision Theory nhưng chưa đủ dữ liệu để tính kết luận tối ưu.\n",
        }

    if problem.get("probability_tree"):
        result = solve_binary_event_tree(problem)
    elif problem.get("bayes"):
        result = solve_bayes_problem(problem)
    elif problem.get("diagnostic_decision"):
        result = solve_diagnostic_decision_tree(problem)
    elif problem.get("imperfect_information_decision"):
        result = solve_imperfect_information_decision_tree(problem)
    elif problem.get("forklift_decision"):
        result = solve_forklift_decision_tree(problem)
    elif problem.get("independent_probabilities"):
        result = solve_independent_probability(problem)
    elif problem.get("decision_tree"):
        result = rollback_decision_tree(problem)
    else:
        result = _solve_basic_emv(problem, recognition)
        if problem.get("payoff_matrix"):
            result["voi"] = _safe_voi(problem)

    result["recognition"] = recognition
    verification = verify_decision_solution(problem, recognition, result)
    result["verification"] = verification
    result["markdown_report"] = gate_md + "\n" + result.get("markdown_report", "") + _verification_markdown(verification)
    return result


def verify_decision_solution(problem: dict[str, Any], recognition: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    checks: list[str] = []
    passed = True
    states = problem.get("states", [])
    if states:
        probs = [float(state.get("probability", 0.0)) for state in states if "probability" in state]
        if probs:
            total = sum(probs)
            if abs(total - 1.0) > 1e-6:
                passed = False
                checks.append(f"Prior probabilities sum = {total:.6f}, không bằng 1.")
            else:
                checks.append("Prior probabilities sum to 1.")

    spec = problem.get("imperfect_information_decision")
    if spec:
        state_names = [str(state.get("name")) for state in spec.get("states", [])]
        signal_sums = {state: 0.0 for state in state_names}
        for signal in spec.get("signals", []):
            likelihoods = signal.get("likelihoods", {})
            for state in state_names:
                signal_sums[state] += float(likelihoods.get(state, 0.0))
        for state, total in signal_sums.items():
            if abs(total - 1.0) > 1e-6:
                passed = False
                checks.append(f"Conditional probabilities for state `{state}` sum = {total:.6f}, không bằng 1.")
            else:
                checks.append(f"Conditional probabilities for state `{state}` sum to 1.")
        for row in result.get("signals", []):
            total = sum(float(v) for v in row.get("posterior", {}).values())
            if abs(total - 1.0) > 1e-6:
                passed = False
                checks.append(f"Posterior probabilities for `{row.get('signal')}` sum = {total:.6f}.")
        piv = result.get("perfect_information", {}).get("value")
        evsi = result.get("sample_information_value")
        if piv is not None and evsi is not None:
            if evsi - piv > 1e-6:
                passed = False
                checks.append(f"EVSI/IISV = {evsi:.6f} > EVPI/PIV = {piv:.6f}.")
            else:
                checks.append("EVSI/IISV <= EVPI/PIV.")

    if recognition["objective"].startswith("minimize"):
        checks.append("Orientation: minimize expected cost/loss.")
    elif recognition["objective"].startswith("maximize"):
        checks.append("Orientation: maximize expected payoff/saving/utility.")
    return {"passed": passed, "checks": checks}


def _solve_basic_emv(problem: dict[str, Any], recognition: dict[str, Any]) -> dict[str, Any]:
    states = problem.get("states", [])
    alternatives = problem.get("alternatives", [])
    payoff_matrix = problem.get("payoff_matrix", [])
    minimize = recognition["objective"].startswith("minimize")
    probs = {state["name"]: float(state.get("probability", 0.0)) for state in states}
    lookup: dict[tuple[str, str], float] = {}
    for cell in payoff_matrix:
        value = float(cell.get("cost", cell.get("payoff", 0.0))) if minimize else float(cell.get("payoff", 0.0)) - float(cell.get("cost", 0.0))
        lookup[(cell["alternative"], cell["state"])] = value

    rows = []
    for alt in alternatives:
        name = alt["name"]
        terms = []
        expected = 0.0
        for state in states:
            value = lookup.get((name, state["name"]), 0.0)
            prob = probs[state["name"]]
            expected += prob * value
            terms.append(f"{prob:.4f} x {value:,.2f}")
        rows.append({"alternative": name, "calculation": " + ".join(terms), "expected_value": expected})
    rows.sort(key=lambda item: item["expected_value"], reverse=not minimize)
    label = "Expected Cost" if minimize else "EMV"
    md = "## 4. Tính toán\n\n"
    md += f"| Alternative | Calculation | {label} |\n|---|---|---:|\n"
    for row in rows:
        md += f"| {row['alternative']} | {row['calculation']} | {row['expected_value']:,.4f} |\n"
    if rows:
        md += f"\nChọn **{rows[0]['alternative']}**, {label} = **{rows[0]['expected_value']:,.4f}**.\n"
    return {
        "status": "computed",
        "solver": "decision_theory_basic_emv",
        "criterion": "expected_cost" if minimize else "expected_value",
        "results": rows,
        "recommendation": rows[0]["alternative"] if rows else None,
        "markdown_report": md,
    }


def _safe_voi(problem: dict[str, Any]) -> dict[str, Any]:
    try:
        return value_of_information(problem)
    except Exception as exc:
        return {"error": str(exc)}


def _matrix_is_cost_oriented(payoffs: list[dict[str, Any]], objective: str) -> bool:
    if objective.startswith("minimize"):
        return True
    if not payoffs:
        return False
    return all("payoff" not in cell and "cost" in cell for cell in payoffs)


def _objective(problem: dict[str, Any]) -> str:
    objective = problem.get("context", {}).get("objective_direction") or ""
    if isinstance(problem.get("objective"), dict):
        objective = problem["objective"].get("sense", objective)
    lowered = str(objective).lower()
    if any(word in lowered for word in ["min", "cost", "loss"]):
        return "minimize cost"
    if "utility" in lowered:
        return "maximize utility"
    return "maximize benefit"


def _infer_decision_maker(description: str) -> str:
    lowered = description.lower()
    for word in ["firm", "company", "investor", "hospital", "manufacturer", "tayota", "adviser"]:
        if word in lowered:
            return word
    return "decision maker" if description else ""


def _root_decision_labels(nodes: list[dict[str, Any]]) -> list[str]:
    by_id = {node.get("id"): node for node in nodes}
    child_ids = {child for node in nodes for child in node.get("children", [])}
    roots = [node for node in nodes if node.get("id") not in child_ids and node.get("node_type") == "decision"]
    if not roots:
        return []
    return [by_id.get(child, {}).get("label", str(child)) for child in roots[0].get("children", [])]


def _missing_payoff_pairs(alternatives: list[str], states: list[str], payoffs: list[dict[str, Any]]) -> list[str]:
    if not alternatives or not states or not payoffs:
        return []
    keys = {(cell.get("alternative"), cell.get("state")) for cell in payoffs}
    missing = []
    for alt in alternatives:
        for state in states:
            if (alt, state) not in keys:
                missing.append(f"payoff_or_cost({alt},{state})")
    return missing


def _not_decision_theory(reason: str, missing: list[str]) -> dict[str, Any]:
    return {
        "problem_type": "Decision Theory",
        "subtype": "NOT_DECISION_THEORY",
        "confidence": 0.2,
        "evidence": [reason],
        "decision_maker": "",
        "alternatives": [],
        "states_of_nature": [],
        "probabilities": [],
        "payoffs_or_costs": [],
        "information_options": [],
        "test_accuracy_data": [],
        "objective": "",
        "missing_information": missing,
        "can_solve": False,
    }


def _recognition_markdown(recognition: dict[str, Any]) -> str:
    evidence = "\n".join(f"- {item}" for item in recognition["evidence"]) or "- Chưa có dấu hiệu đủ mạnh."
    missing = "\n".join(f"- {item}" for item in recognition["missing_information"]) or "- Không thiếu dữ liệu bắt buộc."
    return (
        "# Lời giải\n\n"
        "## 1. Nhận dạng dạng toán\n\n"
        f"- Dạng toán chính: {recognition['problem_type']}\n"
        f"- Dạng toán phụ: {recognition['subtype']}\n"
        f"- Decision maker: {recognition['decision_maker'] or 'chưa xác định'}\n"
        f"- Objective: {recognition['objective']}\n"
        f"- Mức tin cậy: {recognition['confidence']:.2f}\n"
        f"- Có thể giải ngay: {'Có' if recognition['can_solve'] else 'Không'}\n\n"
        "Vì sao dùng Decision Theory:\n"
        f"{evidence}\n\n"
        "Dữ liệu còn thiếu:\n"
        f"{missing}\n\n"
        "## 2. Dữ liệu đã trích xuất\n\n"
        f"- Alternatives: {', '.join(recognition['alternatives']) or 'chưa xác định'}\n"
        f"- States of nature: {', '.join(recognition['states_of_nature']) or 'chưa xác định'}\n"
        f"- Information options: {', '.join(recognition['information_options']) or 'không có'}\n\n"
    )


def _verification_markdown(verification: dict[str, Any]) -> str:
    checks = "\n".join(f"- {item}" for item in verification.get("checks", [])) or "- Không có check bổ sung."
    return (
        "\n## 6. Kiểm tra nghiệm\n\n"
        f"- Trạng thái kiểm tra: {'passed' if verification.get('passed') else 'failed'}\n"
        f"{checks}\n"
    )
