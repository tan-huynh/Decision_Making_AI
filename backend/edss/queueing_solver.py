from __future__ import annotations

import math
from typing import Any

from .queueing import solve_mm1, solve_mm1k, solve_mmc, solve_open_queue_network


def recognize_queueing_theory(problem: dict[str, Any]) -> dict[str, Any]:
    q = {**problem.get("queueing", {}), **problem}
    description = problem.get("context", {}).get("description", "") or ""
    lowered = description.lower()
    evidence: list[str] = []
    missing: list[str] = []

    arrival = _num(q.get("arrival_rate") or q.get("lambda"))
    service = _num(q.get("service_rate") or q.get("mu"))
    servers = int(q.get("servers") or q.get("number_of_servers_s") or 0)
    capacity = q.get("queue_capacity") or q.get("capacity") or q.get("K")
    population_size = q.get("population_size") or q.get("population_size_N") or q.get("N")
    optimize_servers = bool(q.get("optimize_servers") or q.get("choose_number_of_servers") or _has_cost_data(q))

    if arrival is not None:
        evidence.append("Có arrival rate λ.")
    if service is not None:
        evidence.append("Có service rate μ hoặc average service time đã được chuyển thành rate.")
    if servers:
        evidence.append("Có số server.")
    if any(word in lowered for word in ["queue", "queueing", "waiting line", "arrival", "service", "server", "cashier", "poisson"]):
        evidence.append("Có keyword hàng đợi/arrival/service/server.")

    population_type = "finite" if population_size else q.get("population_type", "infinite")
    network_structure = "single_server"
    subtype = "QT_MM1_INFINITE"
    if q.get("nodes") and q.get("routing_matrix"):
        network_structure = "network_with_feedback"
        subtype = "QT_OPEN_QUEUEING_NETWORK"
    elif capacity:
        subtype = "QT_FINITE_CAPACITY"
    elif optimize_servers:
        subtype = "QT_COST_OPTIMIZATION"
        network_structure = "parallel_servers"
    elif population_type == "finite":
        subtype = "QT_MMS_FINITE_SOURCE" if servers and servers > 1 else "QT_MM1_FINITE_SOURCE"
    elif servers and servers > 1:
        subtype = "QT_MMS_INFINITE"
        network_structure = "parallel_servers"

    if arrival is None and subtype != "QT_OPEN_QUEUEING_NETWORK":
        missing.append("arrival_rate_lambda")
    if service is None and subtype != "QT_OPEN_QUEUEING_NETWORK":
        missing.append("service_rate_mu")
    if not servers and not optimize_servers and subtype != "QT_OPEN_QUEUEING_NETWORK":
        missing.append("number_of_servers_s")
    if population_type == "unknown":
        missing.append("population_type")

    confidence = 0.35 + 0.2 * (arrival is not None) + 0.2 * (service is not None) + 0.15 * bool(servers or optimize_servers) + 0.15 * bool(evidence)
    if subtype == "QT_OPEN_QUEUEING_NETWORK" and q.get("nodes") and q.get("routing_matrix") and q.get("external_arrivals"):
        confidence = 0.93
        missing = []
    can_solve = confidence >= 0.85 and not missing
    return {
        "problem_type": "Queueing Theory",
        "subtype": subtype,
        "confidence": round(min(confidence, 0.99), 2),
        "evidence": evidence,
        "arrival_rate_lambda": arrival,
        "service_rate_mu": service,
        "number_of_servers_s": servers or None,
        "population_type": population_type if population_type in {"finite", "infinite"} else "unknown",
        "population_size_N": int(population_size) if population_size else None,
        "queue_capacity_K": int(capacity) if capacity else None,
        "arrival_distribution": q.get("arrival_distribution", "Poisson"),
        "service_distribution": q.get("service_distribution", "exponential"),
        "queue_discipline": q.get("queue_discipline", "FIFO"),
        "network_structure": network_structure,
        "objective": _objective(q, optimize_servers),
        "cost_parameters": _cost_parameters(q),
        "requested_outputs": q.get("requested_outputs", ["rho", "P0", "Lq", "L", "Wq", "W"]),
        "missing_information": missing,
        "can_solve": can_solve,
    }


def solve_queueing_problem(problem: dict[str, Any]) -> dict[str, Any]:
    recognition = recognize_queueing_theory(problem)
    gate_md = _recognition_markdown(recognition)
    if not recognition["can_solve"]:
        return {
            "status": "needs_clarification",
            "solver": "queueing_recognition_gate",
            "recognition": recognition,
            "missing_data": recognition["missing_information"],
            "markdown_report": gate_md + "\nChưa đủ dữ liệu để chọn công thức hàng đợi và tính steady-state metrics.\n",
        }

    subtype = recognition["subtype"]
    lam = float(recognition["arrival_rate_lambda"] or 0)
    mu = float(recognition["service_rate_mu"] or 0)
    s = int(recognition["number_of_servers_s"] or 1)
    if subtype == "QT_OPEN_QUEUEING_NETWORK":
        q = {**problem.get("queueing", {}), **problem}
        result = solve_open_queue_network(q["nodes"], q["routing_matrix"], q["external_arrivals"])
    elif subtype == "QT_COST_OPTIMIZATION":
        result = _optimize_servers(problem, lam, mu)
    elif subtype == "QT_FINITE_CAPACITY":
        result = solve_mm1k(lam, mu, int(recognition["queue_capacity_K"]))
    elif s > 1:
        result = solve_mmc(lam, mu, s)
    else:
        result = solve_mm1(lam, mu)

    result = _augment_metrics(result, lam, mu, s)
    verification = verify_queueing_solution(recognition, result)
    result["recognition"] = recognition
    result["verification"] = verification
    result["markdown_report"] = gate_md + "\n" + _solution_markdown(recognition, result) + _verification_markdown(verification)
    return result


def verify_queueing_solution(recognition: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    checks: list[str] = []
    passed = True
    if result.get("status") == "unstable":
        checks.append("Stability check failed; steady-state metrics are not valid.")
        return {"passed": False, "checks": checks}
    lam = float(recognition.get("arrival_rate_lambda") or 0)
    mu = float(recognition.get("service_rate_mu") or 0)
    rho = result.get("rho")
    if rho is not None and not (0 <= float(rho) < 1):
        passed = False
        checks.append(f"ρ = {rho:.6f} không nằm trong [0,1).")
    elif rho is not None:
        checks.append(f"ρ = {rho:.6f} < 1, hệ ổn định.")
    if lam and all(key in result for key in ["L", "W"]):
        if abs(float(result["L"]) - lam * float(result["W"])) <= 1e-6:
            checks.append("Little's Law L = λW passed.")
        else:
            passed = False
            checks.append("Little's Law L = λW failed.")
    if lam and all(key in result for key in ["Lq", "Wq"]):
        if abs(float(result["Lq"]) - lam * float(result["Wq"])) <= 1e-6:
            checks.append("Little's Law Lq = λWq passed.")
    if mu and all(key in result for key in ["W", "Wq"]):
        if abs(float(result["W"]) - (float(result["Wq"]) + 1 / mu)) <= 1e-6:
            checks.append("Time relation W = Wq + 1/μ passed.")
    return {"passed": passed, "checks": checks}


def _optimize_servers(problem: dict[str, Any], lam: float, mu: float) -> dict[str, Any]:
    q = {**problem.get("queueing", {}), **problem}
    waiting_cost = float(q.get("waiting_cost", q.get("waiting_cost_per_customer_time", 0.0)))
    service_cost = float(q.get("service_cost", q.get("service_cost_per_server_time", 0.0)))
    max_servers = int(q.get("max_servers") or max(math.floor(lam / mu) + 8, 8))
    s_min = max(1, math.floor(lam / mu) + 1)
    rows = []
    best = None
    for s in range(s_min, max_servers + 1):
        metrics = solve_mmc(lam, mu, s)
        if metrics.get("status") != "stable":
            continue
        waiting = waiting_cost * float(metrics["Lq"])
        service = service_cost * s
        total = waiting + service
        row = {**metrics, "servers": s, "waiting_cost": waiting, "service_cost": service, "total_cost": total}
        rows.append(row)
        if best is None or total < best["total_cost"]:
            best = row
        if best and service > best["total_cost"] and s > best["servers"] + 1:
            break
    return {
        **(best or {}),
        "status": "optimal" if best else "needs_clarification",
        "model": "M/M/s cost optimization",
        "options": rows,
        "best": best,
        "optimal_servers": best["servers"] if best else None,
        "objective_value": best["total_cost"] if best else None,
    }


def _augment_metrics(result: dict[str, Any], lam: float, mu: float, s: int) -> dict[str, Any]:
    if result.get("model") == "M/M/1" and result.get("status") == "stable":
        rho = lam / mu
        result["p0"] = 1 - rho
        result["probability_wait"] = rho
    if result.get("model") == "M/M/c" and result.get("status") == "stable":
        a = lam / mu
        rho = lam / (s * mu)
        p0 = result.get("p0", 0.0)
        result["probability_wait"] = ((a**s) / (math.factorial(s) * (1 - rho))) * p0
    return result


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_cost_data(q: dict[str, Any]) -> bool:
    return any(key in q for key in ["waiting_cost", "service_cost", "waiting_cost_per_customer_time", "service_cost_per_server_time"])


def _cost_parameters(q: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(q[key])
        for key in ["waiting_cost", "service_cost", "waiting_cost_per_customer_time", "service_cost_per_server_time"]
        if key in q and q[key] is not None
    }


def _objective(q: dict[str, Any], optimize_servers: bool) -> str:
    if optimize_servers:
        return "minimize_total_cost"
    if q.get("compare_alternatives"):
        return "compare_alternatives"
    return "compute_performance"


def _recognition_markdown(r: dict[str, Any]) -> str:
    evidence = "\n".join(f"- {item}" for item in r["evidence"]) or "- Chưa có dấu hiệu đủ mạnh."
    missing = "\n".join(f"- {item}" for item in r["missing_information"]) or "- Không thiếu dữ liệu bắt buộc."
    return (
        "# Lời giải\n\n"
        "## 1. Nhận dạng dạng toán\n\n"
        f"- Dạng toán chính: {r['problem_type']}\n"
        f"- Dạng toán phụ: {r['subtype']}\n"
        f"- Kendall notation: {_kendall(r)}\n"
        f"- Arrival process: {r['arrival_distribution']}\n"
        f"- Service process: {r['service_distribution']}\n"
        f"- Number of servers: {r['number_of_servers_s'] or 'candidate values'}\n"
        f"- Population type: {r['population_type']}\n"
        f"- Queue discipline: {r['queue_discipline']}\n"
        f"- Objective: {r['objective']}\n"
        f"- Mức tin cậy: {r['confidence']:.2f}\n\n"
        "Evidence:\n"
        f"{evidence}\n\n"
        "Missing information:\n"
        f"{missing}\n\n"
    )


def _solution_markdown(r: dict[str, Any], result: dict[str, Any]) -> str:
    md = "## 2. Dữ liệu đã trích xuất và chuẩn hóa đơn vị\n\n"
    md += "| Parameter | Meaning | Original value | Converted value |\n|---|---|---:|---:|\n"
    md += f"| λ | arrival rate | {r['arrival_rate_lambda']} | {r['arrival_rate_lambda']} |\n"
    md += f"| μ | service rate per server | {r['service_rate_mu']} | {r['service_rate_mu']} |\n"
    md += f"| s | servers | {r['number_of_servers_s'] or 'candidate'} | {r['number_of_servers_s'] or 'candidate'} |\n\n"
    md += "## 3. Kiểm tra điều kiện ổn định\n\n"
    if result.get("rho") is not None:
        md += f"- ρ = {result['rho']:.6f}\n"
        md += f"- Kết luận: {'ổn định' if result.get('status') != 'unstable' else 'không ổn định'}\n\n"
    md += "## 5. Tính toán chi tiết\n\n"
    if result.get("options"):
        md += "| s | ρ | P0 | Lq | Wq | L | W | Waiting cost | Service cost | Total cost |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        for row in result["options"]:
            md += f"| {row['servers']} | {row['rho']:.4f} | {row.get('p0', 0):.4f} | {row['Lq']:.4f} | {row['Wq']:.4f} | {row['L']:.4f} | {row['W']:.4f} | {row['waiting_cost']:.4f} | {row['service_cost']:.4f} | {row['total_cost']:.4f} |\n"
        md += f"\nOptimal servers = **{result.get('optimal_servers')}**, minimum total cost = **{result.get('objective_value'):.4f}**.\n"
    else:
        for key in ["p0", "probability_wait", "Lq", "L", "Wq", "W"]:
            if key in result:
                md += f"- {key} = **{float(result[key]):.6f}**\n"
    return md + "\n"


def _verification_markdown(v: dict[str, Any]) -> str:
    checks = "\n".join(f"- {item}" for item in v.get("checks", []))
    return f"\n## 7. Kiểm tra nghiệm\n\n- Trạng thái kiểm tra: {'passed' if v.get('passed') else 'failed'}\n{checks}\n"


def _kendall(r: dict[str, Any]) -> str:
    if r["subtype"] == "QT_MMS_INFINITE":
        return f"M/M/{r['number_of_servers_s']}"
    if r["subtype"] == "QT_FINITE_CAPACITY":
        return f"M/M/1/{r['queue_capacity_K']}"
    if r["subtype"] == "QT_OPEN_QUEUEING_NETWORK":
        return "Jackson open network"
    return "M/M/1"
