from __future__ import annotations

import random
from typing import Any


def normalize_states(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = sum(max(0.0, float(state.get("probability", 0))) for state in states)
    if total <= 0:
        probability = 1 / max(1, len(states))
        return [{**state, "probability": probability} for state in states]
    return [{**state, "probability": max(0.0, float(state.get("probability", 0))) / total} for state in states]


def payoff_lookup(payoff_matrix: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    return {
        (cell["alternative"], cell["state"]): float(cell.get("payoff", 0)) - float(cell.get("cost", 0))
        for cell in payoff_matrix
    }


def expected_payoff(problem: dict[str, Any]) -> dict[str, Any]:
    states = normalize_states(problem.get("states", []))
    lookup = payoff_lookup(problem.get("payoff_matrix", []))
    results = []
    for alt in problem.get("alternatives", []):
        name = alt["name"]
        value = sum(state["probability"] * lookup.get((name, state["name"]), 0.0) for state in states)
        worst = min((lookup.get((name, state["name"]), 0.0) for state in states), default=0.0)
        results.append({"alternative": name, "expected_value": value, "worst_case": worst})
    results.sort(key=lambda item: item["expected_value"], reverse=True)
    md = "### Báo cáo Expected Value\n\n"
    md += "#### 1. Công thức\n\n"
    md += "EV(a_i) = Σ_j P(s_j) × payoff(a_i, s_j)\n\n"
    md += "#### 2. Kết quả theo phương án\n\n"
    md += "| Phương án | Expected value | Worst case |\n|---|---:|---:|\n"
    for row in results:
        md += f"| {row['alternative']} | {row['expected_value']:.4f} | {row['worst_case']:.4f} |\n"
    if results:
        md += f"\n#### 3. Khuyến nghị\n\nChọn **{results[0]['alternative']}** vì có expected value lớn nhất. Đây là quyết định tốt theo mô hình kỳ vọng, không phải bảo đảm outcome thực tế chắc chắn tốt.\n"
    return {
        "status": "computed",
        "criterion": "expected_value",
        "results": results,
        "recommendation": results[0]["alternative"] if results else None,
        "markdown_report": md,
    }


def solve_bayes_problem(problem: dict[str, Any]) -> dict[str, Any]:
    spec = problem.get("bayes", {})
    prior = min(1.0, max(0.0, float(spec.get("prior", 0.5))))
    sensitivity = min(1.0, max(0.0, float(spec.get("sensitivity", 0.9))))
    false_positive_rate = min(1.0, max(0.0, float(spec.get("false_positive_rate", 0.1))))
    observed_positive = bool(spec.get("observed_positive", True))
    result = bayesian_update(prior, sensitivity, false_positive_rate, observed_positive)
    evidence_symbol = "E+" if observed_positive else "E-"
    md = "### Báo cáo Bayes\n\n"
    md += "#### 1. Mô hình\n\n"
    md += f"- Prior P(H): {prior:.2%}\n"
    md += f"- Sensitivity P(E+|H): {sensitivity:.2%}\n"
    md += f"- False positive P(E+|¬H): {false_positive_rate:.2%}\n"
    md += f"- Quan sát: {evidence_symbol}\n\n"
    md += "#### 2. Công thức\n\n"
    if observed_positive:
        md += "P(H|E+) = P(E+|H)P(H) / [P(E+|H)P(H) + P(E+|¬H)P(¬H)]\n\n"
    else:
        md += "P(H|E-) = P(E-|H)P(H) / [P(E-|H)P(H) + P(E-|¬H)P(¬H)]\n\n"
    md += "#### 3. Kết quả\n\n"
    md += f"- P(evidence): {result['evidence_probability']:.4f}\n"
    md += f"- Posterior: **{result['posterior']:.2%}**\n\n"
    md += "#### 4. Khuyến nghị\n\nPosterior là xác suất đã cập nhật sau bằng chứng. Nếu dữ liệu sensitivity hoặc false-positive chưa chắc, cần phân tích độ nhạy trước khi ra quyết định chắc chắn.\n"
    return {**result, "status": "computed", "solver": "bayes_rule", "markdown_report": md, "recommendation": "Dùng posterior probability sau khi cập nhật bằng chứng."}


def solve_independent_probability(problem: dict[str, Any]) -> dict[str, Any]:
    probs = [min(1.0, max(0.0, float(p))) for p in problem.get("independent_probabilities", [])]
    p_all = 1.0
    p_none = 1.0
    for p in probs:
        p_all *= p
        p_none *= 1 - p
    p_at_least_one = 1 - p_none
    md = "### Báo cáo biến cố độc lập\n\n"
    md += f"Xác suất từng biến cố: {', '.join(f'{p:.2%}' for p in probs)}\n\n"
    md += "| Đại lượng | Công thức | Kết quả |\n|---|---|---:|\n"
    md += f"| Tất cả xảy ra | Π p_i | {p_all:.4%} |\n"
    md += f"| Không biến cố nào xảy ra | Π (1-p_i) | {p_none:.4%} |\n"
    md += f"| Ít nhất một biến cố xảy ra | 1 - Π(1-p_i) | {p_at_least_one:.4%} |\n"
    return {
        "status": "computed",
        "solver": "independent_probability",
        "individual_probabilities": probs,
        "P_all_occur": p_all,
        "P_none_occur": p_none,
        "P_at_least_one": p_at_least_one,
        "markdown_report": md,
        "recommendation": "Dùng phép nhân xác suất độc lập; nếu biến cố phụ thuộc thì mô hình này không hợp lệ.",
    }


def decision_criteria(problem: dict[str, Any], alpha: float = 0.5) -> dict[str, Any]:
    """Apply multiple decision criteria for comparison.

    Criteria: maximin, maximax, minimax regret, Hurwicz, expected value.
    alpha = optimism coefficient for Hurwicz (0=pessimist, 1=optimist).
    """
    states = normalize_states(problem.get("states", []))
    lookup = payoff_lookup(problem.get("payoff_matrix", []))
    alternatives = [alt["name"] for alt in problem.get("alternatives", [])]
    state_names = [s["name"] for s in states]

    rows: list[dict[str, Any]] = []
    for alt in alternatives:
        payoffs = [lookup.get((alt, sn), 0.0) for sn in state_names]
        worst = min(payoffs) if payoffs else 0.0
        best = max(payoffs) if payoffs else 0.0
        ev = sum(states[j]["probability"] * payoffs[j] for j in range(len(states)))
        hurwicz = alpha * best + (1 - alpha) * worst
        rows.append({
            "alternative": alt, "payoffs": payoffs,
            "min": worst, "max": best, "ev": round(ev, 4), "hurwicz": round(hurwicz, 4),
        })

    # Minimax regret
    for j in range(len(state_names)):
        col_max = max(r["payoffs"][j] for r in rows)
        for r in rows:
            r.setdefault("regrets", []).append(round(col_max - r["payoffs"][j], 4))
    for r in rows:
        r["max_regret"] = max(r["regrets"])

    # Find winners
    maximin_winner = max(rows, key=lambda r: r["min"])["alternative"]
    maximax_winner = max(rows, key=lambda r: r["max"])["alternative"]
    regret_winner = min(rows, key=lambda r: r["max_regret"])["alternative"]
    hurwicz_winner = max(rows, key=lambda r: r["hurwicz"])["alternative"]
    ev_winner = max(rows, key=lambda r: r["ev"])["alternative"]

    criteria_results = {
        "maximin": {"winner": maximin_winner, "description": "Pessimistic: maximize the worst-case payoff"},
        "maximax": {"winner": maximax_winner, "description": "Optimistic: maximize the best-case payoff"},
        "minimax_regret": {"winner": regret_winner, "description": "Minimize maximum regret (opportunity cost)"},
        "hurwicz": {"winner": hurwicz_winner, "alpha": alpha, "description": f"Weighted optimism (α={alpha})"},
        "expected_value": {"winner": ev_winner, "description": "Maximize expected payoff (requires probabilities)"},
    }

    # Consensus check
    winners = [maximin_winner, maximax_winner, regret_winner, hurwicz_winner, ev_winner]
    from collections import Counter
    counts = Counter(winners)
    consensus = counts.most_common(1)[0]

    return {
        "status": "computed",
        "criteria": criteria_results,
        "detail_table": rows,
        "states": state_names,
        "consensus": {"alternative": consensus[0], "agreed_by": consensus[1], "out_of": 5},
        "recommendation": (
            f"Consensus: {consensus[0]} được chọn bởi {consensus[1]}/5 tiêu chí. "
            + ("Quyết định robust." if consensus[1] >= 4 else "Các tiêu chí không đồng thuận; xem xét thêm dữ liệu.")
        ),
    }


def value_of_information(problem: dict[str, Any]) -> dict[str, Any]:
    states = normalize_states(problem.get("states", []))
    lookup = payoff_lookup(problem.get("payoff_matrix", []))
    alternatives = [alt["name"] for alt in problem.get("alternatives", [])]
    current = expected_payoff(problem)
    current_best = current["results"][0]["expected_value"] if current["results"] else 0.0
    perfect = 0.0
    for state in states:
        perfect += state["probability"] * max((lookup.get((alt, state["name"]), 0.0) for alt in alternatives), default=0.0)
    evpi = perfect - current_best
    return {"EVwPI": perfect, "EVwoPI": current_best, "EVPI": max(0.0, evpi), "recommendation": "Thu thập thêm thông tin nếu chi phí thông tin nhỏ hơn EVPI."}


def rollback_decision_tree(problem: dict[str, Any]) -> dict[str, Any]:
    nodes = {node["id"]: dict(node) for node in problem.get("decision_tree", [])}
    memo: dict[str, float] = {}

    def value(node_id: str) -> float:
        if node_id in memo:
            return memo[node_id]
        node = nodes[node_id]
        if node["node_type"] == "outcome":
            memo[node_id] = float(node.get("value", 0) or 0)
        elif node["node_type"] == "chance":
            memo[node_id] = sum(float(nodes[child].get("probability", 0) or 0) * value(child) for child in node.get("children", []))
        else:
            memo[node_id] = max((value(child) for child in node.get("children", [])), default=0.0)
        return memo[node_id]

    roots = [node_id for node_id, node in nodes.items() if node["node_type"] == "decision"]
    if not roots:
        raise ValueError("Decision tree requires at least one decision node.")
    root = roots[0]
    child_values = [{"node": child, "label": nodes[child].get("label"), "value": value(child)} for child in nodes[root].get("children", [])]
    child_values.sort(key=lambda item: item["value"], reverse=True)
    return {"status": "computed", "root_value": value(root), "branches": child_values, "recommendation": child_values[0] if child_values else None}


def solve_binary_event_tree(problem: dict[str, Any]) -> dict[str, Any]:
    spec = problem.get("probability_tree", {})
    success_probability = min(1.0, max(0.0, float(spec.get("success_probability", 0))))
    trials = max(1, int(spec.get("trials", 1)))
    success_label = spec.get("success_label", "success")
    failure_label = spec.get("failure_label", "failure")
    failure_probability = 1 - success_probability

    outcomes: list[dict[str, Any]] = []
    for mask in range(2**trials):
        events = []
        probability = 1.0
        successes = 0
        calc_steps = []
        for trial in range(trials):
            is_success = bool(mask & (1 << (trials - trial - 1)))
            events.append(success_label if is_success else failure_label)
            if is_success:
                successes += 1
                probability *= success_probability
                calc_steps.append(f"{success_probability:.2f}")
            else:
                probability *= failure_probability
                calc_steps.append(f"{failure_probability:.2f}")
                
        calc_str = " \\times ".join(calc_steps) + f" = {probability:.4f}"
        outcomes.append(
            {
                "events": events,
                "label": " / ".join(events),
                "probability": probability,
                "success_count": successes,
                "calculation": calc_str,
            }
        )

    first_success_second_failure = None
    if trials >= 2:
        first_success_second_failure = success_probability * failure_probability

    at_least_one_success = 1 - failure_probability**trials
    tree_levels = [
        {
            "trial": trial + 1,
            "branches": [
                {"label": success_label, "probability": success_probability},
                {"label": failure_label, "probability": failure_probability},
            ],
        }
        for trial in range(trials)
    ]

    md = f"### Báo cáo kết quả cây xác suất (Probability Tree)\n\n"
    md += f"**Mô hình:** {trials} sự kiện độc lập liên tiếp.\n"
    md += f"- Xác suất `{success_label}`: {success_probability:.2%}\n"
    md += f"- Xác suất `{failure_label}`: {failure_probability:.2%}\n\n"

    md += "#### 1. Cấu trúc cây xác suất\n\n"
    md += "```mermaid\ngraph LR\n"
    md += "  Start((Bắt đầu))\n"
    for trial in range(trials):
        prev_nodes = [f"N{trial}_{i}" for i in range(2**trial)] if trial > 0 else ["Start"]
        for i, parent in enumerate(prev_nodes):
            md += f"  {parent} -->|\"{success_probability:.2f}\"| N{trial+1}_{i*2}[\"{success_label}\"]\n"
            md += f"  {parent} -->|\"{failure_probability:.2f}\"| N{trial+1}_{i*2+1}[\"{failure_label}\"]\n"
    md += "```\n\n"

    md += "#### 2. Không gian mẫu và Xác suất các biến cố đơn\n\n"
    md += "| Lộ trình các sự kiện | Tổng số lần đạt | Chi tiết phép tính | Xác suất |\n"
    md += "|:---|:---:|:---|---:|\n"
    for outcome in outcomes:
        md += f"| {outcome['label']} | {outcome['success_count']} | ${outcome['calculation']}$ | **{outcome['probability']:.2%}** |\n"
    md += "\n"

    if trials >= 2:
        md += "#### 3. Phân tích các câu hỏi liên quan\n\n"
        
        # Breakdown for at least one success
        md += f"- **Xác suất có ít nhất 1 lần `{success_label}`:** `{at_least_one_success:.2%}`\n"
        md += f"  - *Cách tính (Phần bù):* $1 - P(\\text{{0 lần đạt}}) = 1 - ({failure_probability}^{trials}) = 1 - {failure_probability**trials:.4f} = {at_least_one_success:.4f}$\n"
        
        # Breakdown for first success, second failure
        if first_success_second_failure is not None:
            md += f"- **Xác suất lần 1 `{success_label}`, lần 2 `{failure_label}`:** `{first_success_second_failure:.2%}`\n"
            md += f"  - *Cách tính (Nhân xác suất độc lập):* $P(\\text{{Lần 1}}) \\times P(\\text{{Lần 2}}) = {success_probability:.2f} \\times {failure_probability:.2f} = {first_success_second_failure:.4f}$\n"

    return {
        "status": "computed",
        "solver": "binary_probability_tree",
        "success_probability": success_probability,
        "failure_probability": failure_probability,
        "trials": trials,
        "tree_levels": tree_levels,
        "outcomes": outcomes,
        "queries": {
            "first_success_second_failure": first_success_second_failure,
            "at_least_one_success": at_least_one_success,
        },
        "markdown_report": md,
        "recommendation": "Dùng cây xác suất độc lập để liệt kê đầy đủ các biến cố đơn và cộng các biến cố thỏa điều kiện.",
    }


def bayesian_update(prior: float, sensitivity: float, false_positive_rate: float, observed_positive: bool = True) -> dict[str, float]:
    prior = min(1.0, max(0.0, prior))
    sensitivity = min(1.0, max(0.0, sensitivity))
    false_positive_rate = min(1.0, max(0.0, false_positive_rate))
    if observed_positive:
        evidence = sensitivity * prior + false_positive_rate * (1 - prior)
        posterior = (sensitivity * prior / evidence) if evidence else prior
    else:
        evidence = (1 - sensitivity) * prior + (1 - false_positive_rate) * (1 - prior)
        posterior = ((1 - sensitivity) * prior / evidence) if evidence else prior
    return {"prior": prior, "posterior": posterior, "evidence_probability": evidence}


def simulate_payoffs(problem: dict[str, Any], iterations: int = 2000, seed: int = 7) -> dict[str, Any]:
    rng = random.Random(seed)
    states = normalize_states(problem.get("states", []))
    lookup = payoff_lookup(problem.get("payoff_matrix", []))
    cumulative = []
    total = 0.0
    for state in states:
        total += state["probability"]
        cumulative.append((total, state["name"]))
    output = []
    for alt in problem.get("alternatives", []):
        values = []
        for _ in range(iterations):
            draw = rng.random()
            state_name = next(name for cutoff, name in cumulative if draw <= cutoff)
            values.append(lookup.get((alt["name"], state_name), 0.0))
        values.sort()
        mean = sum(values) / len(values)
        output.append({"alternative": alt["name"], "mean": mean, "p05": values[int(0.05 * (iterations - 1))], "p50": values[int(0.5 * (iterations - 1))], "p95": values[int(0.95 * (iterations - 1))]})
    output.sort(key=lambda item: item["mean"], reverse=True)
    return {"iterations": iterations, "results": output}
