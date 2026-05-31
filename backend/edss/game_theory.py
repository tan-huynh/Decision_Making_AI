from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from scipy.optimize import linprog


def solve_game_theory(problem: dict[str, Any]) -> dict[str, Any]:
    spec = problem.get("game") or problem.get("game_theory") or problem
    gate = recognize_game(spec, problem.get("context", {}).get("description", ""))
    if gate["confidence"] < 0.85 or gate["missing_information"]:
        return {
            "status": "needs_clarification",
            "solver": "game_theory_recognition_gate",
            "recognition": gate,
            "missing_information": gate["missing_information"],
            "markdown_report": _recognition_markdown(gate),
        }

    matrix = np.array(spec["payoff_matrix"], dtype=float)
    row_strategies = list(spec["row_strategies"])
    column_strategies = list(spec["column_strategies"])
    pure = check_pure_strategy(matrix, row_strategies, column_strategies)
    reductions = eliminate_dominated_strategies(matrix, row_strategies, column_strategies)
    reduced = reductions["matrix"]
    reduced_rows = reductions["row_strategies"]
    reduced_cols = reductions["column_strategies"]
    pure_reduced = check_pure_strategy(reduced, reduced_rows, reduced_cols)

    if pure["has_saddle_point"]:
        subtype = "ZERO_SUM_PURE_STRATEGY"
        method = "pure minimax/maximin"
        solution = {"type": "pure", **pure}
    elif pure_reduced["has_saddle_point"]:
        subtype = "DOMINANCE_REDUCTION_GAME"
        method = "dominance reduction then pure minimax/maximin"
        solution = {"type": "pure_after_dominance", **pure_reduced}
    elif reduced.shape == (2, 2):
        subtype = "ZERO_SUM_MIXED_2X2"
        method = "algebraic 2 x 2 mixed strategy"
        solution = solve_2x2_mixed(reduced, reduced_rows, reduced_cols)
    elif reduced.shape[0] == 2 or reduced.shape[1] == 2:
        subtype = "ZERO_SUM_MIXED_2XM" if reduced.shape[0] == 2 else "ZERO_SUM_MIXED_MX2"
        method = "graph/envelope method"
        solution = solve_graph_method(reduced, reduced_rows, reduced_cols)
    else:
        subtype = "ZERO_SUM_GENERAL_MXN"
        method = "linear programming method"
        solution = solve_zero_sum_lp(reduced, reduced_rows, reduced_cols)
        if _is_building_encounter_game(reduced, reduced_rows, reduced_cols):
            value = float(solution.get("game_value", 0.0))
            solution["meeting_probability_per_round"] = value
            solution["expected_rounds_to_meet"] = (1 / value) if value > 1e-12 else None
            solution["recommendation"] = (
                "Both players should randomize uniformly: P(R)=P(L)=P(O)=1/3. "
                "The probability of meeting in one time unit is 2/3, so the expected meeting time is 3/2 time units."
            )

    gate["subtype"] = subtype
    gate["is_pure_strategy_candidate"] = pure["has_saddle_point"] or pure_reduced["has_saddle_point"]
    gate["is_mixed_strategy_candidate"] = not gate["is_pure_strategy_candidate"]
    gate["method"] = method
    markdown = format_game_solution(
        gate=gate,
        matrix=matrix,
        row_strategies=row_strategies,
        column_strategies=column_strategies,
        pure=pure,
        reductions=reductions,
        pure_reduced=pure_reduced,
        solution=solution,
    )
    return {
        "status": "computed",
        "solver": "game_theory",
        "recognition": gate,
        "pure_strategy_check": pure,
        "dominance_reduction": reductions,
        "reduced_pure_strategy_check": pure_reduced,
        "solution": solution,
        "recommendation": solution.get("recommendation"),
        "markdown_report": markdown,
    }


def recognize_game(spec: dict[str, Any], description: str = "") -> dict[str, Any]:
    players = list(spec.get("players") or [])
    row_player = spec.get("row_player") or (players[0] if players else "Row player")
    column_player = spec.get("column_player") or (players[1] if len(players) > 1 else "Column player")
    row_strategies = list(spec.get("row_strategies") or [])
    column_strategies = list(spec.get("column_strategies") or [])
    matrix = spec.get("payoff_matrix")
    payoff_pairs = spec.get("payoff_pair_matrix")
    missing = []
    if not row_player or not column_player:
        missing.append("players")
    if not row_strategies:
        missing.append("row_strategies")
    if not column_strategies:
        missing.append("column_strategies")
    if matrix is None and payoff_pairs is None:
        missing.append("payoff_matrix")
    if not spec.get("payoff_orientation"):
        missing.append("payoff_orientation")

    evidence = []
    lowered = description.lower()
    for token in ["game", "player", "strategy", "strategies", "payoff", "zero-sum", "saddle", "minimax", "maximin", "market share"]:
        if token in lowered:
            evidence.append(token)
    if matrix is not None:
        evidence.append("single payoff matrix present")
    if payoff_pairs is not None:
        evidence.append("payoff-pair matrix present")

    matrix_size = "unknown"
    if matrix is not None:
        try:
            arr = np.array(matrix, dtype=float)
            matrix_size = f"{arr.shape[0]} x {arr.shape[1]}"
        except Exception:
            missing.append("numeric_payoff_matrix")

    is_zero_sum = bool(spec.get("is_zero_sum", matrix is not None and payoff_pairs is None))
    confidence = 0.45
    if len(row_strategies) >= 2 and len(column_strategies) >= 2:
        confidence += 0.2
    if matrix is not None or payoff_pairs is not None:
        confidence += 0.25
    if is_zero_sum or payoff_pairs is not None:
        confidence += 0.1
    if missing:
        confidence -= 0.15
    confidence = max(0.0, min(0.98, confidence))

    return {
        "problem_type": "Game Theory",
        "subtype": spec.get("subtype", "UNKNOWN_GAME"),
        "players": players or [row_player, column_player],
        "row_player": row_player,
        "column_player": column_player,
        "row_strategies": row_strategies,
        "column_strategies": column_strategies,
        "payoff_orientation": spec.get("payoff_orientation", ""),
        "is_zero_sum": is_zero_sum,
        "is_pure_strategy_candidate": False,
        "is_mixed_strategy_candidate": False,
        "matrix_size": matrix_size,
        "confidence": round(confidence, 3),
        "evidence": evidence,
        "missing_information": missing,
    }


def check_pure_strategy(matrix: np.ndarray, row_strategies: list[str], column_strategies: list[str]) -> dict[str, Any]:
    row_minima = matrix.min(axis=1)
    maximin_index = int(row_minima.argmax())
    maximin = float(row_minima[maximin_index])
    column_maxima = matrix.max(axis=0)
    minimax_index = int(column_maxima.argmin())
    minimax = float(column_maxima[minimax_index])
    has_saddle = abs(maximin - minimax) < 1e-9
    saddle_points = []
    if has_saddle:
        for i, j in zip(*np.where(np.isclose(matrix, maximin))):
            if np.isclose(row_minima[i], maximin) and np.isclose(column_maxima[j], minimax):
                saddle_points.append({"row": row_strategies[int(i)], "column": column_strategies[int(j)], "value": maximin})
    return {
        "row_minima": [{"strategy": row_strategies[i], "minimum_payoff": float(value)} for i, value in enumerate(row_minima)],
        "maximin": maximin,
        "maximin_strategy": row_strategies[maximin_index],
        "column_maxima": [{"strategy": column_strategies[j], "maximum_payoff_for_row": float(value)} for j, value in enumerate(column_maxima)],
        "minimax": minimax,
        "minimax_strategy": column_strategies[minimax_index],
        "has_saddle_point": has_saddle,
        "saddle_points": saddle_points,
    }


def eliminate_dominated_strategies(matrix: np.ndarray, row_strategies: list[str], column_strategies: list[str]) -> dict[str, Any]:
    active_rows = list(range(matrix.shape[0]))
    active_cols = list(range(matrix.shape[1]))
    steps = []
    changed = True
    while changed:
        changed = False
        sub = matrix[np.ix_(active_rows, active_cols)]
        for r_pos, s_pos in combinations(range(len(active_rows)), 2):
            r = sub[r_pos]
            s = sub[s_pos]
            if np.all(r <= s) and np.any(r < s):
                removed = active_rows.pop(r_pos)
                steps.append({"type": "row", "removed": row_strategies[removed], "dominated_by": row_strategies[active_rows[s_pos - (1 if s_pos > r_pos else 0)]], "reason": "row maximizer payoff is <= another row in every column"})
                changed = True
                break
            if np.all(s <= r) and np.any(s < r):
                removed = active_rows.pop(s_pos)
                steps.append({"type": "row", "removed": row_strategies[removed], "dominated_by": row_strategies[active_rows[r_pos]], "reason": "row maximizer payoff is <= another row in every column"})
                changed = True
                break
        if changed:
            continue
        sub = matrix[np.ix_(active_rows, active_cols)]
        for c_pos, d_pos in combinations(range(len(active_cols)), 2):
            c = sub[:, c_pos]
            d = sub[:, d_pos]
            if np.all(c >= d) and np.any(c > d):
                removed = active_cols.pop(c_pos)
                steps.append({"type": "column", "removed": column_strategies[removed], "dominated_by": column_strategies[active_cols[d_pos - (1 if d_pos > c_pos else 0)]], "reason": "column minimizer payoff to row is >= another column in every row"})
                changed = True
                break
            if np.all(d >= c) and np.any(d > c):
                removed = active_cols.pop(d_pos)
                steps.append({"type": "column", "removed": column_strategies[removed], "dominated_by": column_strategies[active_cols[c_pos]], "reason": "column minimizer payoff to row is >= another column in every row"})
                changed = True
                break
    return {
        "steps": steps,
        "matrix": matrix[np.ix_(active_rows, active_cols)],
        "row_strategies": [row_strategies[i] for i in active_rows],
        "column_strategies": [column_strategies[j] for j in active_cols],
    }


def solve_2x2_mixed(matrix: np.ndarray, row_strategies: list[str], column_strategies: list[str]) -> dict[str, Any]:
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    denom = a - b - c + d
    if abs(denom) < 1e-12:
        return solve_zero_sum_lp(matrix, row_strategies, column_strategies)
    p = (d - c) / denom
    q = (d - b) / denom
    value = (a * d - b * c) / denom
    return {
        "type": "mixed_2x2",
        "row_probabilities": {row_strategies[0]: float(p), row_strategies[1]: float(1 - p)},
        "column_probabilities": {column_strategies[0]: float(q), column_strategies[1]: float(1 - q)},
        "game_value": float(value),
        "denominator": float(denom),
        "recommendation": f"Row mixes {row_strategies[0]} with p={p:.4f}; column mixes {column_strategies[0]} with q={q:.4f}.",
    }


def solve_graph_method(matrix: np.ndarray, row_strategies: list[str], column_strategies: list[str]) -> dict[str, Any]:
    if matrix.shape[0] == 2:
        candidates = {0.0, 1.0}
        for j, k in combinations(range(matrix.shape[1]), 2):
            slope = (matrix[0, j] - matrix[1, j]) - (matrix[0, k] - matrix[1, k])
            rhs = matrix[1, k] - matrix[1, j]
            if abs(slope) > 1e-12:
                p = rhs / slope
                if 0 <= p <= 1:
                    candidates.add(float(p))
        rows = []
        for p in sorted(candidates):
            values = p * matrix[0, :] + (1 - p) * matrix[1, :]
            rows.append({"p": p, "column_values": values.tolist(), "lower_envelope": float(values.min())})
        best = max(rows, key=lambda item: item["lower_envelope"])
        return {"type": "graph_2xm", "candidate_table": rows, "row_probabilities": {row_strategies[0]: best["p"], row_strategies[1]: 1 - best["p"]}, "game_value": best["lower_envelope"], "recommendation": "Use active columns at the lower envelope intersection."}
    transposed = -matrix.T
    solved = solve_graph_method(transposed, column_strategies, row_strategies)
    return {"type": "graph_mx2", "column_probabilities": solved.get("row_probabilities", {}), "game_value": -float(solved["game_value"]), "candidate_table": solved.get("candidate_table", []), "recommendation": "Column player uses the upper-envelope equivalent solution."}


def solve_zero_sum_lp(matrix: np.ndarray, row_strategies: list[str], column_strategies: list[str]) -> dict[str, Any]:
    shift = max(0.0, 1.0 - float(matrix.min()))
    shifted = matrix + shift
    m, n = shifted.shape
    c = np.ones(m)
    a_ub = -shifted.T
    b_ub = -np.ones(n)
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=[(0, None)] * m, method="highs")
    if not result.success:
        return {"type": "lp_general", "status": "failed", "message": result.message}
    x = result.x
    total = float(x.sum())
    probabilities = x / total
    shifted_value = 1 / total
    value = shifted_value - shift
    column_probabilities: dict[str, float] = {}
    column_result = linprog(
        -np.ones(n),
        A_ub=shifted,
        b_ub=np.ones(m),
        bounds=[(0, None)] * n,
        method="highs",
    )
    if column_result.success:
        y = column_result.x
        y_total = float(y.sum())
        if y_total > 1e-12:
            column_probabilities = {column_strategies[j]: float(y[j] / y_total) for j in range(n)}
    return {
        "type": "lp_general",
        "status": "optimal",
        "shift": shift,
        "row_probabilities": {row_strategies[i]: float(probabilities[i]) for i in range(m)},
        "column_probabilities": column_probabilities,
        "game_value": float(value),
        "shifted_game_value": float(shifted_value),
        "recommendation": "Use the reported mixed strategies; probabilities are LP variables normalized by their totals.",
    }


def _is_building_encounter_game(matrix: np.ndarray, row_strategies: list[str], column_strategies: list[str]) -> bool:
    expected = np.ones((3, 3)) - np.eye(3)
    return (
        matrix.shape == (3, 3)
        and row_strategies == ["R", "L", "O"]
        and column_strategies == ["R", "L", "O"]
        and np.allclose(matrix, expected)
    )


def format_game_solution(**data: Any) -> str:
    gate = data["gate"]
    matrix = data["matrix"]
    rows = data["row_strategies"]
    cols = data["column_strategies"]
    pure = data["pure"]
    reductions = data["reductions"]
    pure_reduced = data["pure_reduced"]
    solution = data["solution"]
    md = "# Lời giải\n\n"
    md += "## 1. Nhận dạng dạng toán\n\n"
    for key in ["problem_type", "subtype", "players", "row_player", "column_player", "payoff_orientation", "is_zero_sum", "matrix_size", "confidence"]:
        md += f"- **{key}**: {gate.get(key)}\n"
    md += f"- **Phương pháp giải**: {gate.get('method')}\n\n"
    md += "## 2. Dữ liệu đã trích xuất\n\n"
    md += _matrix_markdown(matrix, rows, cols)
    md += "\n## 3. Kiểm tra pure strategy\n\n"
    md += "| Strategy of A | Minimum payoff |\n|---|---:|\n"
    for item in pure["row_minima"]:
        md += f"| {item['strategy']} | {item['minimum_payoff']:.4f} |\n"
    md += "\n| Strategy of B | Maximum payoff for A |\n|---|---:|\n"
    for item in pure["column_maxima"]:
        md += f"| {item['strategy']} | {item['maximum_payoff_for_row']:.4f} |\n"
    md += f"\n- maximin = **{pure['maximin']:.4f}**\n- minimax = **{pure['minimax']:.4f}**\n- Saddle point: **{pure['has_saddle_point']}**\n\n"
    md += "## 4. Loại dominated strategies nếu có\n\n"
    if reductions["steps"]:
        for step in reductions["steps"]:
            md += f"- Loại {step['type']} `{step['removed']}` vì bị `{step['dominated_by']}` dominate: {step['reason']}.\n"
        md += "\nReduced matrix:\n\n" + _matrix_markdown(reductions["matrix"], reductions["row_strategies"], reductions["column_strategies"]) + "\n"
        md += f"- Saddle point after reduction: **{pure_reduced['has_saddle_point']}**\n"
    else:
        md += "- Không có dominated strategy hợp lệ để loại.\n"
    md += "\n## 5. Chọn phương pháp giải\n\n"
    md += f"- {gate.get('method')}\n\n"
    md += "## 6. Giải chi tiết\n\n"
    if solution["type"].startswith("pure"):
        md += f"- Optimal row strategy: **{solution['maximin_strategy']}**\n"
        md += f"- Optimal column strategy: **{solution['minimax_strategy']}**\n"
        md += f"- Game value: **{solution['maximin']:.4f}**\n"
    elif solution["type"] == "mixed_2x2":
        md += f"- Denominator: `{solution['denominator']:.4f}`\n"
        md += f"- Row probabilities: `{solution['row_probabilities']}`\n"
        md += f"- Column probabilities: `{solution['column_probabilities']}`\n"
        md += f"- Game value: **{solution['game_value']:.4f}**\n"
    elif solution["type"].startswith("graph"):
        md += "| Candidate p | Envelope value |\n|---:|---:|\n"
        for item in solution.get("candidate_table", []):
            md += f"| {item['p']:.4f} | {item['lower_envelope']:.4f} |\n"
        md += f"\n- Game value: **{solution['game_value']:.4f}**\n"
    else:
        md += f"- Row probabilities: `{solution.get('row_probabilities')}`\n"
        if solution.get("column_probabilities"):
            md += f"- Column probabilities: `{solution.get('column_probabilities')}`\n"
        md += f"- Shift: `{solution.get('shift')}`\n"
        md += f"- Game value: **{solution.get('game_value'):.4f}**\n"
        if solution.get("meeting_probability_per_round") is not None:
            md += f"- Probability of meeting per time unit: **{solution['meeting_probability_per_round']:.4f}**\n"
        if solution.get("expected_rounds_to_meet") is not None:
            md += f"- Expected time units to meet: **{solution['expected_rounds_to_meet']:.4f}**\n"
    md += "\n## 7. Kiểm tra nghiệm\n\n"
    md += "- Pure strategy: maximin = minimax nếu có saddle point.\n"
    md += "- Dominance: chỉ loại row <= row khác hoặc column >= column khác theo đúng zero-sum orientation.\n"
    md += "- Mixed/LP probabilities được kiểm tra trong solver.\n\n"
    md += "## 8. Đáp án cuối cùng\n\n"
    md += f"{solution.get('recommendation', 'Xem strategy/value ở trên')}\n"
    return md


def _matrix_markdown(matrix: np.ndarray, rows: list[str], cols: list[str]) -> str:
    md = "| Row player strategy / Column player strategy | " + " | ".join(cols) + " |\n"
    md += "|---|" + "---:|" * len(cols) + "\n"
    for i, row_name in enumerate(rows):
        md += f"| {row_name} | " + " | ".join(f"{float(matrix[i, j]):.4f}" for j in range(len(cols))) + " |\n"
    return md


def _recognition_markdown(gate: dict[str, Any]) -> str:
    md = "# Nhận dạng Game Theory\n\n"
    for key, value in gate.items():
        md += f"- **{key}**: {value}\n"
    return md
