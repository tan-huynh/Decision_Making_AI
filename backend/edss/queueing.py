from __future__ import annotations

import math
from typing import Any


def solve_mm1(arrival_rate: float, service_rate: float) -> dict[str, Any]:
    if arrival_rate < 0 or service_rate <= 0:
        raise ValueError("arrival_rate không được âm và service_rate phải dương.")
    rho = arrival_rate / service_rate
    if rho >= 1:
        return {
            "status": "unstable",
            "model": "M/M/1",
            "rho": rho,
            "message": "Hệ thống không ổn định vì lambda >= mu.",
            "markdown_report": f"### M/M/1\n\nrho = λ/μ = {rho:.4f} >= 1 nên hàng đợi không ổn định.",
        }
    l = rho / (1 - rho)
    lq = rho**2 / (1 - rho)
    w = 1 / (service_rate - arrival_rate)
    wq = arrival_rate / (service_rate * (service_rate - arrival_rate))
    return _queue_result("M/M/1", rho, l, lq, w, wq)


def solve_mmc(arrival_rate: float, service_rate: float, servers: int) -> dict[str, Any]:
    if arrival_rate < 0 or service_rate <= 0 or servers <= 0:
        raise ValueError("arrival_rate không âm, service_rate dương, servers nguyên dương.")
    rho = arrival_rate / (servers * service_rate)
    if rho >= 1:
        return {
            "status": "unstable",
            "model": "M/M/c",
            "rho": rho,
            "message": "Hệ thống không ổn định vì lambda >= c*mu.",
            "markdown_report": f"### M/M/c\n\nrho = λ/(cμ) = {rho:.4f} >= 1 nên hàng đợi không ổn định.",
        }
    a = arrival_rate / service_rate
    sum_terms = sum((a**n) / math.factorial(n) for n in range(servers))
    last = (a**servers) / (math.factorial(servers) * (1 - rho))
    p0 = 1 / (sum_terms + last)
    lq = p0 * (a**servers) * rho / (math.factorial(servers) * (1 - rho) ** 2)
    l = lq + a
    wq = lq / arrival_rate if arrival_rate else 0
    w = wq + 1 / service_rate
    result = _queue_result("M/M/c", rho, l, lq, w, wq)
    result["servers"] = servers
    result["p0"] = p0
    return result


def solve_mm1k(arrival_rate: float, service_rate: float, capacity: int) -> dict[str, Any]:
    if arrival_rate < 0 or service_rate <= 0 or capacity <= 0:
        raise ValueError("arrival_rate không âm, service_rate dương, capacity nguyên dương.")
    rho = arrival_rate / service_rate
    if abs(rho - 1) < 1e-12:
        p0 = 1 / (capacity + 1)
        probs = [p0 for _ in range(capacity + 1)]
    else:
        p0 = (1 - rho) / (1 - rho ** (capacity + 1))
        probs = [p0 * rho**n for n in range(capacity + 1)]
    pk = probs[-1]
    effective_arrival = arrival_rate * (1 - pk)
    l = sum(n * p for n, p in enumerate(probs))
    w = l / effective_arrival if effective_arrival else math.inf
    lq = l - effective_arrival / service_rate
    wq = lq / effective_arrival if effective_arrival else math.inf
    result = _queue_result("M/M/1/K", rho, l, lq, w, wq)
    result.update({"capacity": capacity, "p0": p0, "blocking_probability": pk, "effective_arrival_rate": effective_arrival})
    return result


def solve_mg1(arrival_rate: float, mean_service_time: float, service_time_variance: float) -> dict[str, Any]:
    if arrival_rate < 0 or mean_service_time <= 0 or service_time_variance < 0:
        raise ValueError("arrival_rate không âm, mean_service_time dương, variance không âm.")
    rho = arrival_rate * mean_service_time
    if rho >= 1:
        return {
            "status": "unstable",
            "model": "M/G/1",
            "rho": rho,
            "message": "Hệ thống không ổn định vì rho = lambda * E[S] >= 1.",
            "markdown_report": f"### M/G/1\n\nrho = λE[S] = {rho:.4f} >= 1 nên hàng đợi không ổn định.",
        }
    second_moment = service_time_variance + mean_service_time**2
    wq = arrival_rate * second_moment / (2 * (1 - rho))
    w = wq + mean_service_time
    lq = arrival_rate * wq
    l = arrival_rate * w
    result = _queue_result("M/G/1", rho, l, lq, w, wq)
    result.update({"mean_service_time": mean_service_time, "service_time_variance": service_time_variance, "pollaczek_khinchine": True})
    result["markdown_report"] = (
        "### Báo cáo M/G/1\n\n"
        "Dùng công thức Pollaczek-Khinchine: `Wq = λE[S^2] / (2(1-rho))`.\n\n"
        f"rho = {rho:.4f}, E[S^2] = {second_moment:.4f}, L = {l:.4f}, Lq = {lq:.4f}, W = {w:.4f}, Wq = {wq:.4f}."
    )
    return result


def solve_open_queue_network(nodes: list[dict[str, float]], routing_matrix: list[list[float]], external_arrivals: list[float]) -> dict[str, Any]:
    """Basic Jackson network traffic equations and M/M/1 metrics per node."""
    import numpy as np

    if not nodes:
        raise ValueError("Queue network cần nodes.")
    n = len(nodes)
    p = np.asarray(routing_matrix, dtype=float)
    gamma = np.asarray(external_arrivals, dtype=float)
    if p.shape != (n, n) or gamma.shape[0] != n:
        raise ValueError("routing_matrix phải là n x n và external_arrivals dài n.")
    identity = np.eye(n)
    lambdas = np.linalg.solve(identity - p.T, gamma)
    node_results = []
    for idx, node in enumerate(nodes):
        service_rate = float(node["service_rate"])
        metrics = solve_mm1(float(lambdas[idx]), service_rate)
        node_results.append({"node": node.get("name", f"Q{idx+1}"), "arrival_rate": round(float(lambdas[idx]), 8), **metrics})
    total_l = sum(item.get("L", 0) for item in node_results if item.get("status") == "stable")
    total_arrival = float(sum(gamma))
    total_w = total_l / total_arrival if total_arrival > 0 else 0
    return {
        "status": "computed",
        "model": "open_jackson_network",
        "effective_arrival_rates": [round(float(v), 8) for v in lambdas],
        "nodes": node_results,
        "total_L": round(total_l, 8),
        "total_W": round(total_w, 8),
        "markdown_report": (
            "### Báo cáo Network of Queues\n\n"
            "Giải traffic equations `λ = γ + P^T λ`, sau đó áp dụng M/M/1 cho từng node.\n\n"
            f"Total L = {total_l:.4f}, Total W = {total_w:.4f}."
        ),
    }


def _queue_result(model: str, rho: float, l: float, lq: float, w: float, wq: float) -> dict[str, Any]:
    return {
        "status": "stable",
        "model": model,
        "rho": rho,
        "L": l,
        "Lq": lq,
        "W": w,
        "Wq": wq,
        "markdown_report": (
            f"### Báo cáo {model}\n\n"
            f"Utilization rho = {rho:.4f}.\n\n"
            f"L = {l:.4f}, Lq = {lq:.4f}, W = {w:.4f}, Wq = {wq:.4f}.\n\n"
            "Kết luận ổn định vì điều kiện tải được thỏa."
        ),
    }
