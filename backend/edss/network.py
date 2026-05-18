"""Network Solvers — Shortest path, transportation, max flow, min cost flow.

Uses NetworkX for graph algorithms when available. Falls back to
pure-Python Dijkstra and least-cost heuristic.
"""

from __future__ import annotations

import heapq
from itertools import islice
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
    negative_edges = [edge for edge in edges if float(edge.get("cost", 1)) < 0]
    if negative_edges:
        return {
            "status": "needs_clarification",
            "solver": "dijkstra_validation",
            "message": "Dijkstra không áp dụng cho cạnh có trọng số âm. Hãy dùng Bellman-Ford hoặc đổi mô hình.",
            "negative_edges": negative_edges,
            "markdown_report": _network_failure_markdown(
                "Shortest Path Problem",
                "Trọng số âm được phát hiện",
                "Dijkstra yêu cầu mọi edge cost không âm. Cần dùng Bellman-Ford/min-cost flow hoặc kiểm tra lại cost.",
            ),
        }

    if HAS_NX:
        return _shortest_path_nx(source, target, edges)
    return _shortest_path_pure(source, target, edges)


def solve_bellman_ford(graph: dict[str, Any]) -> dict[str, Any]:
    source = graph.get("source")
    target = graph.get("target")
    edges = graph.get("edges", [])
    if not source or not target or not edges:
        raise ValueError("Bellman-Ford requires source, target, and edges.")
    nodes = sorted({source, target} | {e["from"] for e in edges} | {e["to"] for e in edges})
    dist = {node: float("inf") for node in nodes}
    pred: dict[str, str | None] = {node: None for node in nodes}
    dist[source] = 0.0
    relaxations = []
    expanded_edges = []
    for edge in edges:
        expanded_edges.append((edge["from"], edge["to"], float(edge.get("cost", 1))))
        if not edge.get("directed", True):
            expanded_edges.append((edge["to"], edge["from"], float(edge.get("cost", 1))))
    for iteration in range(len(nodes) - 1):
        changed = False
        for u, v, cost in expanded_edges:
            if dist[u] + cost < dist[v]:
                dist[v] = dist[u] + cost
                pred[v] = u
                changed = True
        relaxations.append({"iteration": iteration + 1, "distances": {k: round(v, 6) if v < float("inf") else "inf" for k, v in dist.items()}})
        if not changed:
            break
    negative_cycle = any(dist[u] + cost < dist[v] for u, v, cost in expanded_edges)
    if negative_cycle:
        return {
            "status": "unbounded",
            "solver": "bellman_ford",
            "message": "Đồ thị có negative cycle reachable từ source; shortest path không xác định hữu hạn.",
            "relaxations": relaxations,
            "markdown_report": _network_failure_markdown("Bellman-Ford", "Negative cycle", "Có chu trình âm nên chi phí có thể giảm vô hạn."),
        }
    if dist[target] == float("inf"):
        return {
            "status": "infeasible",
            "solver": "bellman_ford",
            "message": f"Không có đường đi từ {source} đến {target}.",
            "relaxations": relaxations,
            "markdown_report": _network_failure_markdown("Bellman-Ford", "No path", f"Không tìm được đường đi từ `{source}` đến `{target}`."),
        }
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = pred[node]
    path.reverse()
    md = "### Báo cáo Bellman-Ford\n\n"
    md += f"Đường đi ngắn nhất từ `{source}` đến `{target}` là **{' → '.join(path)}** với cost **{dist[target]:.4f}**.\n\n"
    md += "| Iteration | Distances |\n|---:|---|\n"
    for row in relaxations:
        md += f"| {row['iteration']} | `{row['distances']}` |\n"
    return {
        "status": "optimal",
        "solver": "bellman_ford",
        "objective_value": round(dist[target], 8),
        "path": path,
        "relaxations": relaxations,
        "markdown_report": md,
    }


def solve_minimum_spanning_tree(graph: dict[str, Any], algorithm: str = "kruskal") -> dict[str, Any]:
    edges = graph.get("edges", [])
    if not edges:
        raise ValueError("MST requires edges.")
    if HAS_NX:
        G = nx.Graph()
        for edge in edges:
            G.add_edge(edge["from"], edge["to"], weight=float(edge.get("cost", edge.get("weight", 1))))
        tree = nx.minimum_spanning_tree(G, algorithm="prim" if algorithm == "prim" else "kruskal", weight="weight")
        selected = [{"from": u, "to": v, "cost": data["weight"]} for u, v, data in tree.edges(data=True)]
        total = sum(item["cost"] for item in selected)
    else:
        selected, total = _kruskal_pure(edges)
    md = "### Báo cáo Minimum Spanning Tree\n\n"
    md += f"Thuật toán: **{algorithm.title()}**. Tổng trọng số cây khung nhỏ nhất = **{total:.4f}**.\n\n"
    md += "| From | To | Cost |\n|---|---|---:|\n"
    for edge in selected:
        md += f"| {edge['from']} | {edge['to']} | {edge['cost']:.4f} |\n"
    return {"status": "optimal", "solver": f"mst_{algorithm}", "objective_value": round(total, 8), "edges": selected, "markdown_report": md}


def _kruskal_pure(edges: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    selected = []
    total = 0.0
    for edge in sorted(edges, key=lambda item: float(item.get("cost", item.get("weight", 1)))):
        if union(edge["from"], edge["to"]):
            cost = float(edge.get("cost", edge.get("weight", 1)))
            selected.append({"from": edge["from"], "to": edge["to"], "cost": cost})
            total += cost
    return selected, total


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
        
        try:
            k_paths = list(islice(nx.shortest_simple_paths(G, source, target, weight="weight"), 5))
            alternatives = []
            for p in k_paths:
                c = nx.path_weight(G, p, weight="weight")
                alternatives.append({"path": " → ".join(p), "cost": c})
        except Exception:
            alternatives = []

        md = f"### Báo cáo kết quả thuật toán Dijkstra\n\n"
        md += f"**Mục tiêu:** Tìm đường đi ngắn nhất từ đỉnh `{source}` đến đỉnh `{target}`.\n\n"
        
        if alternatives:
            md += "#### So sánh các lộ trình khả thi (Top 5)\n\n"
            md += "Để đảm bảo tính tối ưu, thuật toán đã duyệt qua các đường đi có thể và đưa ra bảng so sánh sau:\n\n"
            md += "| Xếp hạng | Đường đi | Tổng chi phí | Chênh lệch so với tối ưu |\n"
            md += "|:---:|---|:---:|:---:|\n"
            for i, alt in enumerate(alternatives):
                diff = alt['cost'] - cost
                diff_str = f"+{diff:.0f}" if diff > 0 else "Tối ưu nhất"
                md += f"| #{i+1} | {alt['path']} | {alt['cost']:.0f} | {diff_str} |\n"
            md += "\n> **Kết luận:** Thuật toán xác định lộ trình hạng #1 là đường đi ngắn nhất tuyệt đối, loại trừ hoàn toàn các rủi ro chọn nhầm đường vòng.\n"
        else:
            md += f"**Đường đi ngắn nhất:** {' → '.join(path)} (Chi phí: {cost})\n"
            md += "Không tìm thấy đường đi thay thế nào khác trong đồ thị.\n"

        return {
            "status": "optimal",
            "solver": "networkx_dijkstra",
            "objective_value": round(cost, 6),
            "path": path,
            "alternatives": alternatives,
            "markdown_report": md,
            "recommendation": f"Đường đi ngắn nhất: {' → '.join(path)}, chi phí = {cost:.0f}.",
        }
    except nx.NetworkXNoPath:
        return {
            "status": "infeasible",
            "message": f"Không có đường đi từ {source} đến {target}.",
            "markdown_report": _network_failure_markdown(
                "Shortest Path Problem",
                "Không có đường đi khả thi",
                f"Đồ thị không kết nối được từ `{source}` đến `{target}`. Cần thêm cạnh hoặc kiểm tra hướng cạnh.",
            ),
        }


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
            md = f"### Báo cáo kết quả thuật toán Dijkstra\n\n"
            md += f"**Mục tiêu:** Tìm đường đi ngắn nhất từ đỉnh `{source}` đến đỉnh `{target}`.\n\n"
            md += f"**Đường đi ngắn nhất:** {' → '.join(path)}\n\n"
            md += f"**Tổng chi phí:** {cost:.0f}\n\n"
            md += "> Thuật toán Pure Dijkstra đã duyệt và xác nhận đây là đường đi ngắn nhất."
            return {
                "status": "optimal", "solver": "dijkstra",
                "objective_value": cost, "path": path,
                "markdown_report": md,
                "recommendation": f"Đường đi ngắn nhất: {' → '.join(path)}, chi phí = {cost:.0f}.",
            }
        for nxt, weight in adj.get(node, []):
            if nxt not in seen:
                heapq.heappush(pq, (cost + weight, nxt, path + [nxt]))
    return {"status": "infeasible", "message": "No path found."}


def _network_failure_markdown(title: str, heading: str, body: str) -> str:
    return f"### {title}\n\n#### {heading}\n\n{body}\n"


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

    # Build detailed markdown report
    md = "### Báo cáo kết quả bài toán Phân bổ Mạng lưới (Transportation/Allocation)\n\n"
    md += "**Mục tiêu:** Tối ưu hóa việc phân bổ từ các Nguồn cung (Sources) đến các Điểm cầu (Destinations) để đạt tổng giá trị mục tiêu (chi phí/khoảng cách) nhỏ nhất.\n\n"
    
    md += "#### 1. Phân bổ tối ưu\n\n"
    md += "| Từ (Nguồn) | Đến (Đích) | Lượng phân bổ | Trọng số/Đơn giá | Giá trị |\n"
    md += "|:---|:---|---:|---:|---:|\n"
    for a in real_allocations:
        md += f"| {a['from']} | {a['to']} | {a['amount']:.2f} | {a['unit_cost']:.2f} | {(a['amount'] * a['unit_cost']):.2f} |\n"
    
    md += f"\n**Tổng giá trị mục tiêu tối ưu:** {total_cost:.2f}\n\n"
    
    if dummy_info:
        md += f"*(Ghi chú: Đã thêm {dummy_info['type']} = {dummy_info['amount']} để cân bằng cung cầu)*\n\n"
        
    md += "#### 2. Phân tích tối ưu (MODI Method)\n\n"
    if certificate.get("is_optimal"):
        md += "> **Kết luận:** Thuật toán MODI xác nhận phương án phân bổ trên là **tối ưu tuyệt đối**. Không thể tìm được phương án nào có tổng giá trị nhỏ hơn.\n\n"
        if certificate.get("reduced_costs"):
            md += "Dưới đây là chi phí cơ hội (Reduced Costs) của một số tuyến không được sử dụng. Vì tất cả đều ≥ 0, việc ép dùng các tuyến này sẽ làm tăng tổng giá trị mục tiêu:\n\n"
            md += "| Tuyến chưa dùng | Reduced Cost (Giá trị tăng thêm / đơn vị) |\n"
            md += "|:---|---:|\n"
            for k, v in list(certificate["reduced_costs"].items())[:5]:
                if not k.startswith("_dummy") and "_dummy" not in k:
                    md += f"| {k} | +{v:.2f} |\n"
    else:
        md += "> **Kết luận:** Đây là phương án khả thi (Feasible) dựa trên thuật toán Heuristic, chưa có chứng chỉ tối ưu tuyệt đối bằng phương pháp MODI.\n\n"

    return {
        "status": "optimal" if certificate.get("is_optimal") else "feasible_heuristic",
        "solver": "least_cost_transportation" + ("_balanced" if dummy_info else ""),
        "objective_value": round(total_cost, 6),
        "allocations": real_allocations,
        "all_allocations": allocations,
        "unmet_demand": unmet,
        "dummy": dummy_info or None,
        "optimality_certificate": certificate,
        "markdown_report": md,
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

    md = "### Báo cáo Max Flow\n\n"
    md += f"**Mục tiêu:** Tìm luồng cực đại từ `{source}` đến `{sink}`.\n\n"
    md += f"**Giá trị luồng cực đại:** {flow_value:.4f}\n\n"
    md += "#### 1. Flow trên các cạnh được sử dụng\n\n"
    md += "| Cạnh | Flow | Capacity | Trạng thái |\n|---|---:|---:|---|\n"
    for item in flows:
        status = "Saturated" if item["saturated"] else "Còn dư"
        md += f"| {item['from']} → {item['to']} | {item['flow']:.4f} | {item['capacity']:.4f} | {status} |\n"
    if bottleneck_edges:
        md += "\n#### 2. Min-cut / Bottleneck\n\n"
        md += "| Cạnh cắt | Capacity |\n|---|---:|\n"
        for edge in bottleneck_edges:
            md += f"| {edge['from']} → {edge['to']} | {edge['capacity']:.4f} |\n"
        md += "\nTheo max-flow min-cut theorem, giá trị max flow bằng capacity của một min cut.\n"

    return {
        "status": "optimal",
        "solver": "networkx_max_flow",
        "objective_value": round(flow_value, 6),
        "flows": flows,
        "bottleneck_edges": bottleneck_edges,
        "markdown_report": md,
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
        return {
            "status": "infeasible",
            "message": "Bài toán min cost flow không khả thi (supply ≠ demand hoặc thiếu capacity).",
            "markdown_report": _network_failure_markdown(
                "Min Cost Flow",
                "Không khả thi",
                "Tổng supply/demand có thể chưa cân bằng, hoặc capacity trên mạng không đủ để chuyển flow tới demand nodes.",
            ),
        }

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

    md = "### Báo cáo Min-Cost Flow\n\n"
    md += "**Mục tiêu:** Thỏa supply/demand với tổng chi phí nhỏ nhất.\n\n"
    md += f"**Tổng chi phí tối ưu:** {flow_cost}\n\n"
    md += "#### 1. Flow tối ưu\n\n"
    md += "| Từ | Đến | Flow | Unit cost | Total cost |\n|---|---|---:|---:|---:|\n"
    for item in flows:
        md += f"| {item['from']} | {item['to']} | {item['flow']} | {item['unit_cost']} | {item['total_cost']} |\n"
    md += "\n#### 2. Kiểm tra cân bằng node\n\n"
    md += "NetworkX dùng quy ước node demand âm cho supply và dương cho demand. Solver `network_simplex` chỉ trả optimal khi toàn bộ cân bằng flow được thỏa.\n"

    return {
        "status": "optimal",
        "solver": "networkx_min_cost_flow",
        "objective_value": flow_cost,
        "flows": flows,
        "markdown_report": md,
        "recommendation": f"Min cost flow tối ưu có tổng chi phí = {flow_cost}.",
    }
