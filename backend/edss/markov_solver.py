from __future__ import annotations

from typing import Any

import numpy as np

from .markov import absorbing_chain, n_step_transition, steady_state


def recognize_markov_processes(problem: dict[str, Any]) -> dict[str, Any]:
    spec = _markov_spec(problem)
    description = problem.get("context", {}).get("description", "") or ""
    lowered = description.lower()
    states = _states(problem, spec)
    matrix = spec.get("transition_matrix") or problem.get("transition_matrix")
    initial = spec.get("initial_distribution") or problem.get("initial_distribution")
    outputs = _requested_outputs(problem, spec, lowered)
    absorbing_states = _absorbing_states(matrix, states, spec)
    state_costs = spec.get("state_costs") or problem.get("state_costs") or []
    target_state = spec.get("target_state_for_first_passage") or problem.get("target_state_for_first_passage")
    n_steps = spec.get("n_steps") or problem.get("n_steps")
    time_step = spec.get("time_step") or problem.get("time_step") or _infer_time_step(lowered)
    evidence: list[str] = []
    missing: list[str] = []

    if problem.get("problem_type") in {"markov_processes", "markov_chain"} or spec:
        evidence.append("Structured Markov chain data present.")
    if any(token in lowered for token in ["markov", "transition matrix", "stationary", "steady", "absorbing", "first passage", "long-run"]):
        evidence.append("Description contains Markov/transition/stationary keywords.")
    if states:
        evidence.append("States are available.")
    if matrix:
        evidence.append("Transition matrix is available.")

    if not states and matrix:
        states = [f"s{i + 1}" for i in range(len(matrix))]
    if not states:
        missing.append("states")
    if not matrix:
        missing.append("transition_matrix")
    if not time_step or time_step == "unknown":
        missing.append("time_step")
    if not outputs:
        missing.append("requested_outputs")
    if matrix:
        matrix_validation = _validate_matrix(matrix)
        missing.extend(matrix_validation["missing_information"])
    else:
        matrix_validation = {"valid": False, "row_sums": [], "orientation": "unknown", "missing_information": ["transition_matrix"]}

    if "n_step_probability" in outputs or "state_vector_evolution" in outputs:
        if n_steps is None:
            missing.append("n_steps")
        if initial is None and "initial_state" not in spec and "initial_state" not in problem:
            missing.append("initial_distribution_or_initial_state")
    if "first_passage_time" in outputs and target_state is None:
        missing.append("target_state_for_first_passage")
    if "long_run_cost" in outputs and not state_costs:
        missing.append("state_costs")

    subtype = _subtype(outputs, absorbing_states, n_steps, state_costs, target_state, lowered)
    confidence = 0.3 + 0.2 * bool(states) + 0.25 * bool(matrix) + 0.15 * bool(outputs) + 0.1 * matrix_validation["valid"]
    if problem.get("problem_type") in {"markov_processes", "markov_chain"}:
        confidence += 0.1
    return {
        "problem_type": "Markov Processes",
        "subtype": subtype,
        "confidence": round(min(confidence, 0.99), 2),
        "evidence": evidence,
        "states": states,
        "time_step": time_step or "unknown",
        "transition_matrix": matrix or [],
        "transition_matrix_orientation": matrix_validation["orientation"],
        "initial_distribution": initial or [],
        "requested_outputs": outputs,
        "absorbing_states": absorbing_states,
        "state_costs": state_costs,
        "target_state_for_first_passage": target_state,
        "n_steps": n_steps,
        "missing_information": _dedupe(missing),
        "can_solve": confidence >= 0.85 and not missing and matrix_validation["valid"],
    }


def solve_markov_processes_problem(problem: dict[str, Any]) -> dict[str, Any]:
    recognition = recognize_markov_processes(problem)
    gate_md = _recognition_markdown(recognition)
    if not recognition["can_solve"]:
        return {
            "status": "needs_clarification",
            "solver": "markov_processes_recognition_gate",
            "recognition": recognition,
            "missing_data": recognition["missing_information"],
            "markdown_report": gate_md + "\nChưa đủ dữ liệu để giải Markov chain và kết luận.\n",
        }

    matrix = recognition["transition_matrix"]
    outputs = recognition["requested_outputs"]
    if "absorbing_chain" in outputs:
        result = absorbing_chain(matrix, _state_indices(recognition["absorbing_states"], recognition["states"]))
    elif "first_passage_time" in outputs:
        result = _first_passage(matrix, recognition["states"], recognition["target_state_for_first_passage"])
    elif "long_run_cost" in outputs:
        result = _long_run_cost(matrix, recognition["states"], recognition["state_costs"])
    elif "n_step_probability" in outputs or "state_vector_evolution" in outputs:
        initial = _initial_distribution(problem, recognition)
        result = n_step_transition(matrix, int(recognition["n_steps"]), initial)
        result["state_distribution_after_n"] = result.get("initial_after_n")
    else:
        result = steady_state(matrix)
        result["stationary_distribution"] = result.get("steady_state")

    verification = verify_markov_solution(recognition, result)
    result["recognition"] = recognition
    result["verification"] = verification
    result["markdown_report"] = gate_md + "\n" + result.get("markdown_report", "") + _verification_markdown(verification)
    return result


def verify_markov_solution(recognition: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    checks: list[str] = []
    passed = result.get("status") in {"computed", "optimal"}
    p = np.asarray(recognition["transition_matrix"], dtype=float)
    if np.any(p < -1e-9) or np.any(p > 1 + 1e-9):
        passed = False
        checks.append("Transition probabilities must lie in [0,1].")
    row_sums = p.sum(axis=1)
    if np.max(np.abs(row_sums - 1)) > 1e-7:
        passed = False
        checks.append("One or more transition matrix rows do not sum to 1.")
    else:
        checks.append("Transition matrix rows sum to 1.")

    vector = result.get("state_distribution_after_n")
    if vector is not None:
        if abs(sum(vector) - 1) > 1e-6:
            passed = False
            checks.append("n-step state distribution does not sum to 1.")
        else:
            checks.append("n-step state distribution sums to 1.")
    stationary = result.get("stationary_distribution") or result.get("steady_state")
    if stationary is not None:
        pi = np.asarray(stationary, dtype=float)
        if abs(float(pi.sum()) - 1) > 1e-6 or np.max(np.abs(pi @ p - pi)) > 1e-6:
            passed = False
            checks.append("Stationary distribution fails πP = π or sum π = 1.")
        else:
            checks.append("Stationary distribution satisfies πP = π and sum π = 1.")
    if result.get("absorption_probabilities") is not None:
        b = np.asarray(result["absorption_probabilities"], dtype=float)
        if np.max(np.abs(b.sum(axis=1) - 1)) > 1e-6:
            passed = False
            checks.append("Absorption probability rows do not sum to 1.")
        else:
            checks.append("Absorption probability rows sum to 1.")
    if result.get("mean_first_passage_steps") is not None:
        if any(value < -1e-8 for value in result["mean_first_passage_steps"].values()):
            passed = False
            checks.append("Mean first passage times must be nonnegative.")
        else:
            checks.append("Mean first passage equations returned nonnegative values.")
    if result.get("long_run_mean_cost") is not None:
        checks.append("Long-run mean cost computed as Σ π_j C_j.")
    return {"passed": passed, "checks": checks}


def _markov_spec(problem: dict[str, Any]) -> dict[str, Any]:
    return dict(problem.get("markov") or problem.get("markov_chain") or {})


def _states(problem: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    raw = spec.get("states") or problem.get("markov_states") or []
    if raw:
        return [item.get("name", str(item)) if isinstance(item, dict) else str(item) for item in raw]
    if problem.get("states"):
        return [item.get("name", str(idx)) for idx, item in enumerate(problem.get("states", []))]
    return []


def _requested_outputs(problem: dict[str, Any], spec: dict[str, Any], lowered: str) -> list[str]:
    outputs = list(spec.get("requested_outputs") or problem.get("requested_outputs") or [])
    if outputs:
        return outputs
    if any(token in lowered for token in ["absorbing", "absorption"]):
        outputs.append("absorbing_chain")
    if "first passage" in lowered or "expected time" in lowered or "average time" in lowered:
        outputs.append("first_passage_time")
    if "cost" in lowered or "profit" in lowered:
        outputs.append("long_run_cost")
    if "after" in lowered or "n-step" in lowered:
        outputs.append("n_step_probability")
    if any(token in lowered for token in ["stationary", "steady", "long-run", "long run"]):
        outputs.append("stationary_distribution")
    return outputs


def _infer_time_step(lowered: str) -> str:
    for token in ["day", "week", "month", "year", "generation", "period"]:
        if token in lowered:
            return token
    return "unknown"


def _validate_matrix(matrix: list[list[float]]) -> dict[str, Any]:
    try:
        p = np.asarray(matrix, dtype=float)
    except Exception:
        return {"valid": False, "row_sums": [], "orientation": "unknown", "missing_information": ["transition_matrix_numeric_values"]}
    missing: list[str] = []
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        missing.append("square_transition_matrix")
        return {"valid": False, "row_sums": [], "orientation": "unknown", "missing_information": missing}
    if np.any(p < -1e-12) or np.any(p > 1 + 1e-12):
        missing.append("probabilities_in_0_1")
    row_sums = p.sum(axis=1)
    col_sums = p.sum(axis=0)
    orientation = "rows_from_current_state_to_next_state"
    if np.max(np.abs(row_sums - 1)) > 1e-7:
        if np.max(np.abs(col_sums - 1)) <= 1e-7:
            orientation = "columns_from_current_state_to_next_state"
            missing.append("confirm_transition_matrix_orientation")
        else:
            missing.append("transition_matrix_row_sums")
    return {"valid": not missing, "row_sums": row_sums.round(10).tolist(), "orientation": orientation, "missing_information": missing}


def _absorbing_states(matrix: list[list[float]] | None, states: list[str], spec: dict[str, Any]) -> list[Any]:
    explicit = spec.get("absorbing_states")
    if explicit is not None:
        return explicit
    if not matrix:
        return []
    p = np.asarray(matrix, dtype=float)
    absorbing: list[Any] = []
    for idx in range(p.shape[0]):
        row = p[idx]
        if abs(row[idx] - 1) <= 1e-9 and np.sum(np.abs(np.delete(row, idx))) <= 1e-9:
            absorbing.append(states[idx] if idx < len(states) else idx)
    return absorbing


def _subtype(outputs: list[str], absorbing_states: list[Any], n_steps: Any, state_costs: Any, target_state: Any, lowered: str) -> str:
    if "formulate" in lowered or "model as" in lowered:
        return "MC_FORMULATION_ONLY"
    if "absorbing_chain" in outputs or absorbing_states:
        return "MC_ABSORBING_CHAIN"
    if "first_passage_time" in outputs or target_state is not None:
        return "MC_FIRST_PASSAGE_TIME"
    if "long_run_cost" in outputs or state_costs:
        return "MC_LONG_RUN_COST"
    if "n_step_probability" in outputs or n_steps is not None:
        return "MC_N_STEP_PROBABILITY"
    if "state_vector_evolution" in outputs:
        return "MC_STATE_VECTOR_EVOLUTION"
    if "stationary_distribution" in outputs:
        return "MC_STATIONARY_DISTRIBUTION"
    return "MC_ONE_STEP_TRANSITION_MATRIX"


def _state_indices(states_or_indices: list[Any], states: list[str]) -> list[int]:
    result: list[int] = []
    for item in states_or_indices:
        if isinstance(item, int):
            result.append(item)
        elif str(item) in states:
            result.append(states.index(str(item)))
        else:
            result.append(int(item))
    return result


def _initial_distribution(problem: dict[str, Any], recognition: dict[str, Any]) -> list[float]:
    if recognition["initial_distribution"]:
        return recognition["initial_distribution"]
    initial_state = problem.get("initial_state") or _markov_spec(problem).get("initial_state")
    states = recognition["states"]
    vector = [0.0] * len(states)
    if isinstance(initial_state, int):
        vector[initial_state] = 1.0
    elif initial_state in states:
        vector[states.index(initial_state)] = 1.0
    else:
        vector[0] = 1.0
    return vector


def _first_passage(matrix: list[list[float]], states: list[str], target_state: Any) -> dict[str, Any]:
    p = np.asarray(matrix, dtype=float)
    target = _state_indices([target_state], states)[0] if not isinstance(target_state, int) else target_state
    unknown = [idx for idx in range(p.shape[0]) if idx != target]
    a = np.eye(len(unknown))
    b = np.ones(len(unknown))
    for row_idx, state_i in enumerate(unknown):
        for col_idx, state_k in enumerate(unknown):
            a[row_idx, col_idx] -= p[state_i, state_k]
    values = np.linalg.solve(a, b)
    passage = {states[target] if idx == target else states[idx]: 0.0 for idx in range(len(states))}
    for idx, value in zip(unknown, values):
        passage[states[idx]] = round(float(value), 10)
    return {
        "status": "computed",
        "model": "markov_mean_first_passage_time",
        "target_state": states[target],
        "mean_first_passage_steps": passage,
        "markdown_report": f"### Mean First Passage Time\n\nTarget state = {states[target]}. Equations `m_i = 1 + Σ p_ik m_k` solved.",
    }


def _long_run_cost(matrix: list[list[float]], states: list[str], state_costs: list[Any]) -> dict[str, Any]:
    steady = steady_state(matrix)
    pi = np.asarray(steady["steady_state"], dtype=float)
    costs = np.asarray([item.get("cost", item.get("value", item)) if isinstance(item, dict) else item for item in state_costs], dtype=float)
    mean_cost = float(pi @ costs)
    rows = [
        {"state": states[idx], "stationary_probability": round(float(pi[idx]), 10), "cost": round(float(costs[idx]), 10), "contribution": round(float(pi[idx] * costs[idx]), 10)}
        for idx in range(len(states))
    ]
    return {
        "status": "computed",
        "model": "markov_long_run_mean_cost",
        "stationary_distribution": pi.round(10).tolist(),
        "state_cost_table": rows,
        "long_run_mean_cost": round(mean_cost, 10),
        "markdown_report": f"### Long-run Mean Cost\n\nC_bar = Σ π_j C_j = {mean_cost:.6f}.",
    }


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _recognition_markdown(r: dict[str, Any]) -> str:
    evidence = "\n".join(f"- {item}" for item in r["evidence"]) or "- Chưa có dấu hiệu đủ mạnh."
    missing = "\n".join(f"- {item}" for item in r["missing_information"]) or "- Không thiếu dữ liệu bắt buộc."
    return (
        "# Lời giải\n\n"
        "## 1. Nhận dạng dạng toán\n\n"
        f"- Dạng toán chính: {r['problem_type']}\n"
        f"- Dạng toán phụ: {r['subtype']}\n"
        "- Vì sao là Markov: trạng thái kế tiếp phụ thuộc vào trạng thái hiện tại qua ma trận chuyển.\n"
        f"- Time step: {r['time_step']}\n"
        f"- States: {', '.join(r['states'])}\n"
        f"- Requested output: {', '.join(r['requested_outputs'])}\n"
        f"- Mức tin cậy: {r['confidence']:.2f}\n\n"
        "Evidence:\n"
        f"{evidence}\n\n"
        "Missing information:\n"
        f"{missing}\n\n"
        "## 2. Dữ liệu đã trích xuất\n\n"
        f"- Transition matrix orientation: `{r['transition_matrix_orientation']}`\n"
        f"- Absorbing states: `{r['absorbing_states']}`\n\n"
    )


def _verification_markdown(v: dict[str, Any]) -> str:
    checks = "\n".join(f"- {item}" for item in v.get("checks", []))
    return f"\n## 6. Kiểm tra nghiệm\n\n- Trạng thái kiểm tra: {'passed' if v.get('passed') else 'failed'}\n{checks}\n"
