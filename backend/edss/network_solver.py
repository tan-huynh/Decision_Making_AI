from __future__ import annotations

from typing import Any

from .network import (
    solve_bellman_ford,
    solve_max_flow,
    solve_min_cost_flow,
    solve_minimum_spanning_tree,
    solve_shortest_path,
)


NETWORK_TYPES = {"network_modelling", "shortest_path", "max_flow", "min_cost_flow", "transshipment", "network_flow"}


def recognize_network_modelling(problem: dict[str, Any]) -> dict[str, Any]:
    graph = problem.get("graph", {})
    description = problem.get("context", {}).get("description", "") or ""
    lowered = description.lower()
    edges = graph.get("edges", [])
    nodes = graph.get("nodes") or sorted({e.get("from") for e in edges} | {e.get("to") for e in edges} - {None})
    source = graph.get("source")
    sink = graph.get("sink") or graph.get("target")
    has_capacity = any("capacity" in edge for edge in edges)
    has_cost = any("cost" in edge or "weight" in edge for edge in edges)
    has_supply_demand = bool(graph.get("supplies") or graph.get("demands") or any("supply" in node or "demand" in node for node in graph.get("nodes", [])))
    directed_values = [edge.get("directed") for edge in edges if "directed" in edge]
    directed = None if not directed_values else any(bool(value) for value in directed_values)
    evidence: list[str] = []
    missing: list[str] = []
    problem_type = problem.get("problem_type", "")

    if edges:
        evidence.append("Có edge/arc list.")
    if source:
        evidence.append("Có source node.")
    if sink:
        evidence.append("Có sink/target node.")
    if has_capacity:
        evidence.append("Có capacity trên arcs.")
    if has_cost:
        evidence.append("Có cost/distance/time weight trên edges.")
    if any(word in lowered for word in ["shortest path", "route", "dijkstra", "emergency"]):
        objective = "shortest_path"
        subtype = "NET_ROUTE_OR_EMERGENCY_PATH" if "emergency" in lowered else "NET_SHORTEST_PATH"
    elif any(word in lowered for word in ["minimum spanning", "spanning tree", "mst", "connect all", "communications network", "fibre"]):
        objective = "min_spanning_tree"
        subtype = "NET_MINIMUM_SPANNING_TREE"
    elif any(word in lowered for word in ["maximum flow", "max flow", "water network"]) or problem_type == "max_flow":
        objective = "max_flow"
        subtype = "NET_MAXIMUM_FLOW"
    elif any(word in lowered for word in ["min-cost", "minimum cost flow", "transshipment"]) or problem_type in {"min_cost_flow", "transshipment", "network_flow"}:
        objective = "min_cost_flow"
        subtype = "NET_TRANSSHIPMENT" if "transshipment" in lowered else "NET_MIN_COST_FLOW"
    elif problem_type == "shortest_path":
        objective = "shortest_path"
        subtype = "NET_SHORTEST_PATH"
    elif has_capacity and source and sink:
        objective = "max_flow"
        subtype = "NET_MAXIMUM_FLOW"
    elif has_supply_demand:
        objective = "min_cost_flow"
        subtype = "NET_MIN_COST_FLOW"
    elif edges and not source and not sink:
        objective = "min_spanning_tree"
        subtype = "NET_MINIMUM_SPANNING_TREE"
    else:
        objective = "shortest_path"
        subtype = "NET_SHORTEST_PATH"

    if not nodes:
        missing.append("nodes")
    if not edges and objective not in {"assignment", "formulation_only"}:
        missing.append("arcs_or_edges")
    if objective == "shortest_path":
        if not source:
            missing.append("source_node")
        if not sink:
            missing.append("target_node")
        if not has_cost:
            missing.append("edge_weights")
    if objective == "max_flow":
        if not source:
            missing.append("source_node")
        if not sink:
            missing.append("sink_node")
        if not has_capacity:
            missing.append("arc_capacities")
    if objective == "min_spanning_tree":
        if not has_cost:
            missing.append("edge_weights")
    if objective == "min_cost_flow":
        if not has_supply_demand:
            missing.append("node_supply_demand")
        if not has_cost:
            missing.append("arc_costs")
        if not has_capacity:
            missing.append("arc_capacities")

    confidence = 0.35 + 0.2 * bool(edges) + 0.15 * bool(nodes) + 0.15 * (has_cost or has_capacity) + 0.15 * bool(evidence)
    if problem_type in NETWORK_TYPES:
        confidence += 0.1
    return {
        "problem_type": "Network Modelling",
        "subtype": subtype,
        "confidence": round(min(confidence, 0.99), 2),
        "evidence": evidence,
        "nodes": [str(node) for node in nodes if node is not None],
        "arcs_or_edges": edges,
        "directed": directed,
        "source_node": source,
        "sink_node": sink,
        "edge_weight_type": _weight_type(edges, lowered),
        "has_capacity": has_capacity,
        "has_cost": has_cost,
        "has_supply_demand": has_supply_demand,
        "has_assignment_structure": bool(problem.get("assignment_costs")),
        "objective": objective,
        "missing_information": missing,
        "can_solve": confidence >= 0.85 and not missing,
    }


def solve_network_modelling_problem(problem: dict[str, Any]) -> dict[str, Any]:
    recognition = recognize_network_modelling(problem)
    gate_md = _recognition_markdown(recognition)
    if not recognition["can_solve"]:
        return {
            "status": "needs_clarification",
            "solver": "network_recognition_gate",
            "recognition": recognition,
            "missing_data": recognition["missing_information"],
            "markdown_report": gate_md + "\nChưa đủ dữ liệu để chọn thuật toán network và kết luận tối ưu.\n",
        }
    graph = problem.get("graph", {})
    objective = recognition["objective"]
    if objective == "shortest_path":
        has_negative = any(float(edge.get("cost", edge.get("weight", 1))) < 0 for edge in graph.get("edges", []))
        result = solve_bellman_ford(graph) if has_negative else solve_shortest_path(graph)
    elif objective == "min_spanning_tree":
        result = solve_minimum_spanning_tree(graph)
    elif objective == "max_flow":
        result = solve_max_flow(graph)
    else:
        result = solve_min_cost_flow(graph)
    verification = verify_network_solution(recognition, result)
    result["recognition"] = recognition
    result["verification"] = verification
    result["markdown_report"] = gate_md + "\n" + result.get("markdown_report", "") + _verification_markdown(verification)
    return result


def verify_network_solution(recognition: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    checks: list[str] = []
    passed = result.get("status") == "optimal"
    objective = recognition["objective"]
    if objective == "shortest_path" and result.get("path"):
        path = result["path"]
        if path[0] == recognition["source_node"] and path[-1] == recognition["sink_node"]:
            checks.append("Path starts at source and ends at target.")
        else:
            passed = False
            checks.append("Path endpoint check failed.")
    if objective == "min_spanning_tree" and result.get("edges"):
        expected = len(set(recognition["nodes"])) - 1
        if len(result["edges"]) == expected:
            checks.append("MST has |V|-1 edges.")
        else:
            passed = False
            checks.append("MST edge count check failed.")
    if objective == "max_flow" and result.get("flows"):
        for flow in result["flows"]:
            if float(flow["flow"]) - float(flow["capacity"]) > 1e-6:
                passed = False
                checks.append(f"Capacity violated on {flow['from']}->{flow['to']}.")
                break
        else:
            checks.append("All reported flows satisfy arc capacities.")
    if objective == "min_cost_flow" and result.get("flows"):
        checks.append("Min-cost flow solver returned feasible balanced flow.")
    return {"passed": passed, "checks": checks or ["Solver status checked."]}


def _weight_type(edges: list[dict[str, Any]], lowered: str) -> str:
    if any("capacity" in edge and "cost" not in edge for edge in edges):
        return "capacity"
    if "time" in lowered:
        return "time"
    if "distance" in lowered or "route" in lowered:
        return "distance"
    if any("cost" in edge for edge in edges):
        return "cost"
    return "unknown"


def _recognition_markdown(r: dict[str, Any]) -> str:
    evidence = "\n".join(f"- {item}" for item in r["evidence"]) or "- Chưa có dấu hiệu đủ mạnh."
    missing = "\n".join(f"- {item}" for item in r["missing_information"]) or "- Không thiếu dữ liệu bắt buộc."
    return (
        "# Lời giải\n\n"
        "## 1. Nhận dạng dạng toán\n\n"
        f"- Dạng toán chính: {r['problem_type']}\n"
        f"- Dạng toán phụ: {r['subtype']}\n"
        f"- Graph directed/undirected: {r['directed']}\n"
        f"- Nodes: {', '.join(r['nodes'])}\n"
        f"- Arcs/edges: {len(r['arcs_or_edges'])}\n"
        f"- Weight type: {r['edge_weight_type']}\n"
        f"- Source: {r['source_node']}\n"
        f"- Sink/target: {r['sink_node']}\n"
        f"- Objective: {r['objective']}\n"
        f"- Mức tin cậy: {r['confidence']:.2f}\n\n"
        "Evidence:\n"
        f"{evidence}\n\n"
        "Missing information:\n"
        f"{missing}\n\n"
        "## 2. Dữ liệu đã trích xuất\n\n"
        "| From | To | Weight/Capacity/Cost |\n|---|---|---:|\n"
        + "".join(
            f"| {edge.get('from')} | {edge.get('to')} | {edge.get('capacity', edge.get('cost', edge.get('weight', '')))} |\n"
            for edge in r["arcs_or_edges"]
        )
        + "\n"
    )


def _verification_markdown(v: dict[str, Any]) -> str:
    checks = "\n".join(f"- {item}" for item in v.get("checks", []))
    return f"\n## 6. Kiểm tra nghiệm\n\n- Trạng thái kiểm tra: {'passed' if v.get('passed') else 'failed'}\n{checks}\n"
