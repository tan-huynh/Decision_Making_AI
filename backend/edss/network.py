"""Network Solvers — Shortest path, transportation, max flow, min cost flow.

Uses NetworkX for graph algorithms when available. Falls back to
pure-Python Dijkstra and least-cost heuristic.
"""

from __future__ import annotations

import heapq
from typing import Any

try:
    import networkx as nx  # type: ignore[import-untyped]
    HAS_NX = True
except ImportError:
    HAS_NX = False


# ── Shortest Path (Dijkstra) ─────────────────────────────────────────────


def solve_shortest_path(graph: dict[str, Any]) -> dict[str, Any]:
    """Dijkstra shortest path. Works with or without NetworkX."""
    source = graph.get("source")
    target = graph.get("target")
    edges = graph.get("edges", [])
    if not source or not target or not edges:
        raise ValueError("Shortest path requires source, target, and edges.")

    if HAS_NX:
        return _shortest_path_nx(source, target, edges)
    return _shortest_path_pure(source, target, edges)


def _shortest_path_nx(source: str, target: str, edges: list[dict[str, Any]]) -> dict[str, Any]:
    G = nx.DiGraph()
    for e in edges:
        directed = e.get("directed", True)
        G.add_edge(e["from"], e["to"], weight=float(e.get("cost", 1)))
        if not directed:
            G.add_edge(e["to"], e["from"], weight=float(e.get("cost", 1)))
    try:
        path = nx.shortest_path(G, source, target, weight="weight")
        cost = nx.shortest_path_length(G, source, target, weight="weight")
        return {
            "status": "optimal",
            "solver": "networkx_dijkstra",
            "objective_value": round(cost, 6),
            "path": path,
            "recommendation": f"Đường đi ngắn nhất: {' → '.join(path)}, chi phí = {cost:.0f}.",
        }
    except nx.NetworkXNoPath:
        return {"status": "infeasible", "message": f"Không có đường đi từ {source} đến {target}."}


def _shortest_path_pure(source: str, target: str, edges: list[dict[str, Any]]) -> dict[str, Any]:
    adj: dict[str, list[tuple[str, float]]] = {}
    for edge in edges:
        u, v = edge["from"], edge["to"]
        cost = float(edge.get("cost", 1))
        adj.setdefault(u, []).append((v, cost))
        if not edge.get("directed", True):
            adj.setdefault(v, []).append((u, cost))
    pq = [(0.0, source, [source])]
    seen: set[str] = set()
    while pq:
        cost, node, path = heapq.heappop(pq)
        if node in seen:
            continue
        seen.add(node)
        if node == target:
            return {
                "status": "optimal", "solver": "dijkstra",
                "objective_value": cost, "path": path,
                "recommendation": f"Đường đi ngắn nhất: {' → '.join(path)}, chi phí = {cost:.0f}.",
            }
        for nxt, weight in adj.get(node, []):
            if nxt not in seen:
                heapq.heappush(pq, (cost + weight, nxt, path + [nxt]))
    return {"status": "infeasible", "message": "No path found."}


# ── Transportation ────────────────────────────────────────────────────────


def solve_transportation(problem: dict[str, Any]) -> dict[str, Any]:
    """Solve transportation problem with auto-balancing and MODI certificate."""
    supplies = dict(problem.get("graph", {}).get("supplies", {}))
    demands = dict(problem.get("graph", {}).get("demands", {}))
    costs = problem.get("graph", {}).get("costs", {})

    if not supplies or not demands or not costs:
        raise ValueError("Transportation cần supplies, demands, và costs.")

    # Auto-balance with dummy source/destination
    total_supply = sum(float(v) for v in supplies.values())
    total_demand = sum(float(v) for v in demands.values())
    dummy_info: dict[str, Any] = {}

    if total_supply > total_demand + 1e-8:
        dummy_name = "_dummy_demand"
        dummy_amount = total_supply - total_demand
        demands[dummy_name] = dummy_amount
        for src in supplies:
            costs.setdefault(src, {})[dummy_name] = 0.0
        dummy_info = {"type": "dummy_demand", "name": dummy_name, "amount": round(dummy_amount, 4)}

    elif total_demand > total_supply + 1e-8:
        dummy_name = "_dummy_supply"
        dummy_amount = total_demand - total_supply
        supplies[dummy_name] = dummy_amount
        costs[dummy_name] = {dst: 0.0 for dst in demands}
        dummy_info = {"type": "dummy_supply", "name": dummy_name, "amount": round(dummy_amount, 4)}

    # Least-cost heuristic
    allocations: list[dict[str, Any]] = []
    total_cost = 0.0
    remaining_supply = {k: float(v) for k, v in supplies.items()}
    remaining_demand = {k: float(v) for k, v in demands.items()}

    pairs = sorted(
        ((float(costs[src][dst]), src, dst)
         for src in costs for dst in costs[src]
         if src in remaining_supply and dst in remaining_demand),
        key=lambda item: item[0],
    )
    for cost, src, dst in pairs:
        amount = min(remaining_supply.get(src, 0), remaining_demand.get(dst, 0))
        if amount <= 1e-8:
            continue
        remaining_supply[src] -= amount
        remaining_demand[dst] -= amount
        total_cost += amount * cost
        allocations.append({"from": src, "to": dst, "amount": round(amount, 6), "unit_cost": cost})

    # MODI certificate
    certificate = _transportation_modi(costs, allocations)

    # Filter out dummy from display
    real_allocations = [a for a in allocations if not a["from"].startswith("_dummy") and not a["to"].startswith("_dummy")]
    unmet = {k: round(v, 4) for k, v in remaining_demand.items() if v > 1e-8 and not k.startswith("_dummy")}

    return {
        "status": "optimal" if certificate.get("is_optimal") else "feasible_heuristic",
        "solver": "least_cost_transportation" + ("_balanced" if dummy_info else ""),
        "objective_value": round(total_cost, 6),
        "allocations": real_allocations,
        "all_allocations": allocations,
        "unmet_demand": unmet,
        "dummy": dummy_info or None,
        "optimality_certificate": certificate,
        "recommendation": (
            f"Tổng chi phí vận chuyển tối thiểu = {total_cost:.2f}. "
            + ("MODI xác nhận tối ưu." if certificate.get("is_optimal") else "Cần kiểm tra MODI/LP để xác nhận tối ưu.")
            + (f" Đã thêm {dummy_info['type']} = {dummy_info['amount']} để cân bằng." if dummy_info else "")
        ),
    }


def _transportation_modi(costs: dict[str, dict[str, float]], allocations: list[dict[str, Any]]) -> dict[str, Any]:
    """MODI method for optimality certificate."""
    basics = [(a["from"], a["to"]) for a in allocations if a.get("amount", 0) > 1e-8]
    if not basics:
        return {"is_optimal": False, "reason": "No basic allocations."}

    u: dict[str, float] = {basics[0][0]: 0.0}
    v: dict[str, float] = {}

    changed = True
    iterations = 0
    while changed and iterations < 100:
        changed = False
        iterations += 1
        for src, dst in basics:
            cost = float(costs.get(src, {}).get(dst, 0))
            if src in u and dst not in v:
                v[dst] = cost - u[src]
                changed = True
            elif dst in v and src not in u:
                u[src] = cost - v[dst]
                changed = True

    reduced_costs: dict[str, float] = {}
    for src, row in costs.items():
        for dst, cost in row.items():
            if (src, dst) in basics:
                continue
            if src not in u or dst not in v:
                continue
            reduced = float(cost) - u[src] - v[dst]
            reduced_costs[f"{src}→{dst}"] = round(reduced, 6)

    is_optimal = bool(reduced_costs) and all(rc >= -1e-6 for rc in reduced_costs.values())
    return {
        "is_optimal": is_optimal,
        "u": {k: round(v, 4) for k, v in u.items()},
        "v": {k: round(v, 4) for k, v in v.items()},
        "reduced_costs": reduced_costs,
        "min_reduced_cost": round(min(reduced_costs.values()), 6) if reduced_costs else 0.0,
    }


# ── Max Flow ──────────────────────────────────────────────────────────────


def solve_max_flow(graph: dict[str, Any]) -> dict[str, Any]:
    """Solve maximum flow problem using NetworkX."""
    if not HAS_NX:
        raise ValueError("NetworkX required for max flow. pip install networkx")

    source = graph.get("source")
    sink = graph.get("sink", graph.get("target"))
    edges = graph.get("edges", [])
    if not source or not sink or not edges:
        raise ValueError("Max flow requires source, sink, and edges with capacity.")

    G = nx.DiGraph()
    for e in edges:
        cap = float(e.get("capacity", float("inf")))
        G.add_edge(e["from"], e["to"], capacity=cap)

    flow_value, flow_dict = nx.maximum_flow(G, source, sink)

    flows: list[dict[str, Any]] = []
    for u in flow_dict:
        for v, f in flow_dict[u].items():
            if f > 1e-8:
                cap = G[u][v].get("capacity", float("inf"))
                flows.append({
                    "from": u, "to": v,
                    "flow": round(f, 6),
                    "capacity": cap,
                    "saturated": abs(f - cap) < 1e-6,
                })

    # Find min cut
    try:
        cut_value, (reachable, non_reachable) = nx.minimum_cut(G, source, sink)
        bottleneck_edges = [
            {"from": u, "to": v, "capacity": G[u][v]["capacity"]}
            for u in reachable for v in G[u]
            if v in non_reachable
        ]
    except Exception:
        bottleneck_edges = []

    return {
        "status": "optimal",
        "solver": "networkx_max_flow",
        "objective_value": round(flow_value, 6),
        "flows": flows,
        "bottleneck_edges": bottleneck_edges,
        "recommendation": (
            f"Luồng cực đại = {flow_value:.2f}. "
            f"Bottleneck: {', '.join(f'{b['from']}→{b['to']}' for b in bottleneck_edges[:3])}."
            if bottleneck_edges else f"Luồng cực đại = {flow_value:.2f}."
        ),
    }


# ── Min Cost Flow ─────────────────────────────────────────────────────────


def solve_min_cost_flow(graph: dict[str, Any]) -> dict[str, Any]:
    """Solve minimum cost flow problem using NetworkX."""
    if not HAS_NX:
        raise ValueError("NetworkX required for min cost flow. pip install networkx")

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not edges:
        raise ValueError("Min cost flow requires edges with cost and capacity.")

    G = nx.DiGraph()

    # Add nodes with supply/demand
    for node in nodes:
        nid = node.get("id", node.get("name"))
        supply = float(node.get("supply", 0))
        demand = float(node.get("demand", 0))
        G.add_node(nid, demand=demand - supply)  # NetworkX uses negative demand for supply

    # Add edges
    for e in edges:
        cap = float(e.get("capacity", float("inf")))
        cost = float(e.get("cost", 0))
        G.add_edge(e["from"], e["to"], capacity=int(cap), weight=int(cost))

    try:
        flow_cost, flow_dict = nx.network_simplex(G)
    except nx.NetworkXUnfeasible:
        return {"status": "infeasible", "message": "Bài toán min cost flow không khả thi (supply ≠ demand hoặc thiếu capacity)."}

    flows: list[dict[str, Any]] = []
    for u in flow_dict:
        for v, f in flow_dict[u].items():
            if f > 0:
                flows.append({
                    "from": u, "to": v,
                    "flow": f,
                    "unit_cost": G[u][v]["weight"],
                    "total_cost": f * G[u][v]["weight"],
                })

    return {
        "status": "optimal",
        "solver": "networkx_min_cost_flow",
        "objective_value": flow_cost,
        "flows": flows,
        "recommendation": f"Min cost flow tối ưu có tổng chi phí = {flow_cost}.",
    }
