from __future__ import annotations

from typing import Any

import numpy as np


def n_step_transition(matrix: list[list[float]], steps: int, initial: list[float] | None = None) -> dict[str, Any]:
    p = _validate_transition_matrix(matrix)
    if steps < 0:
        raise ValueError("steps không được âm.")
    pn = np.linalg.matrix_power(p, steps)
    result: dict[str, Any] = {
        "status": "computed",
        "model": "markov_n_step",
        "steps": steps,
        "transition_matrix_n": pn.round(10).tolist(),
    }
    if initial is not None:
        v = np.asarray(initial, dtype=float)
        if v.shape[0] != p.shape[0]:
            raise ValueError("initial vector phải cùng số trạng thái với matrix.")
        result["initial_after_n"] = (v @ pn).round(10).tolist()
    result["markdown_report"] = f"### Markov n-step\n\nP^{steps} đã được tính bằng matrix power."
    return result


def steady_state(matrix: list[list[float]], tolerance: float = 1e-10, max_iter: int = 10000) -> dict[str, Any]:
    p = _validate_transition_matrix(matrix)
    n = p.shape[0]
    pi = np.ones(n) / n
    iterations = 0
    for iterations in range(1, max_iter + 1):
        nxt = pi @ p
        if np.max(np.abs(nxt - pi)) <= tolerance:
            pi = nxt
            break
        pi = nxt
    return {
        "status": "computed",
        "model": "markov_steady_state",
        "steady_state": pi.round(10).tolist(),
        "iterations": iterations,
        "markdown_report": (
            "### Markov Steady State\n\n"
            f"πP = π, Σπ_i = 1.\n\nπ = {pi.round(6).tolist()}."
        ),
    }


def absorbing_chain(matrix: list[list[float]], absorbing_indices: list[int]) -> dict[str, Any]:
    p = _validate_transition_matrix(matrix)
    n = p.shape[0]
    absorbing = sorted(set(absorbing_indices))
    if any(index < 0 or index >= n for index in absorbing):
        raise ValueError("absorbing_indices ngoài phạm vi matrix.")
    transient = [i for i in range(n) if i not in absorbing]
    if not transient:
        raise ValueError("Cần ít nhất một trạng thái transient.")
    q = p[np.ix_(transient, transient)]
    r = p[np.ix_(transient, absorbing)]
    identity = np.eye(len(transient))
    fundamental = np.linalg.inv(identity - q)
    expected_steps = fundamental @ np.ones(len(transient))
    absorption_probabilities = fundamental @ r
    return {
        "status": "computed",
        "model": "absorbing_markov_chain",
        "transient_states": transient,
        "absorbing_states": absorbing,
        "fundamental_matrix": fundamental.round(10).tolist(),
        "expected_time_to_absorption": expected_steps.round(10).tolist(),
        "absorption_probabilities": absorption_probabilities.round(10).tolist(),
        "markdown_report": (
            "### Absorbing Markov Chain\n\n"
            "N = (I-Q)^-1.\n\n"
            f"Expected time to absorption = {expected_steps.round(6).tolist()}."
        ),
    }


def _validate_transition_matrix(matrix: list[list[float]]) -> np.ndarray:
    p = np.asarray(matrix, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("Transition matrix phải là ma trận vuông.")
    if np.any(p < -1e-12):
        raise ValueError("Transition probabilities không được âm.")
    row_sums = p.sum(axis=1)
    if np.max(np.abs(row_sums - 1)) > 1e-7:
        raise ValueError("Mỗi hàng của transition matrix phải có tổng bằng 1.")
    return p
