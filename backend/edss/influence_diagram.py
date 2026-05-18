from __future__ import annotations

from typing import Any

from .teaching_report import wrap_teaching_report


def analyze_influence_diagram(diagram: dict[str, Any]) -> dict[str, Any]:
    nodes = diagram.get("nodes", [])
    arcs = diagram.get("arcs", [])
    if not nodes:
        raise ValueError("Influence diagram cần nodes.")
    by_id = {node["id"]: node for node in nodes}
    decision_nodes = [node for node in nodes if node.get("type") == "decision"]
    chance_nodes = [node for node in nodes if node.get("type") == "chance"]
    value_nodes = [node for node in nodes if node.get("type") == "value"]
    missing = [arc for arc in arcs if arc.get("from") not in by_id or arc.get("to") not in by_id]
    topo = _topological_order(nodes, arcs) if not missing else []
    mermaid = ["```mermaid", "graph LR"]
    shape = {"decision": ("[", "]"), "chance": ("((", "))"), "value": ("{{", "}}")}
    for node in nodes:
        left, right = shape.get(node.get("type"), ("[", "]"))
        mermaid.append(f'  {node["id"]}{left}"{node.get("label", node["id"])}"{right}')
    for arc in arcs:
        label = arc.get("label", "")
        mermaid.append(f'  {arc["from"]} -->|"{label}"| {arc["to"]}' if label else f'  {arc["from"]} --> {arc["to"]}')
    mermaid.append("```")
    sections = [
        ("Cấu trúc node", f"- Decision nodes: {[n['id'] for n in decision_nodes]}\n- Chance nodes: {[n['id'] for n in chance_nodes]}\n- Value nodes: {[n['id'] for n in value_nodes]}"),
        ("Dependency arcs", "\n".join(f"- {a['from']} -> {a['to']}" for a in arcs) or "_Không có arcs._"),
        ("Sơ đồ", "\n".join(mermaid)),
        ("Thứ tự phân tích", " -> ".join(topo) if topo else "Không tạo được topological order; kiểm tra cycle hoặc arc lỗi."),
    ]
    return {
        "status": "computed" if not missing else "invalid",
        "decision_nodes": decision_nodes,
        "chance_nodes": chance_nodes,
        "value_nodes": value_nodes,
        "topological_order": topo,
        "invalid_arcs": missing,
        "mermaid": "\n".join(mermaid),
        "markdown_report": wrap_teaching_report(
            "Influence Diagram",
            "Decision Analysis Influence Diagram",
            sections,
            "Dùng sơ đồ này để xác định thông tin nào ảnh hưởng tới decision, uncertainty nào cần xác suất, và value node nào là hàm mục tiêu.",
        ),
    }


def influence_to_decision_tree(diagram: dict[str, Any]) -> dict[str, Any]:
    analysis = analyze_influence_diagram(diagram)
    if analysis["status"] != "computed":
        return {"status": "invalid", "analysis": analysis}
    decisions = analysis["decision_nodes"]
    chances = analysis["chance_nodes"]
    values = analysis["value_nodes"]
    skeleton = {
        "root": decisions[0]["id"] if decisions else None,
        "decision_nodes": [node["id"] for node in decisions],
        "chance_nodes": [node["id"] for node in chances],
        "terminal_value_nodes": [node["id"] for node in values],
        "note": "Skeleton conversion only. Add alternatives, chance probabilities, and terminal payoffs to run rollback EV.",
    }
    return {
        "status": "computed",
        "analysis": analysis,
        "decision_tree_skeleton": skeleton,
        "markdown_report": (
            "### Influence Diagram → Decision Tree Skeleton\n\n"
            f"Root decision: `{skeleton['root']}`.\n\n"
            f"Chance nodes cần xác suất: {skeleton['chance_nodes']}.\n\n"
            f"Value nodes cần payoff/utility: {skeleton['terminal_value_nodes']}."
        ),
    }


def _topological_order(nodes: list[dict[str, Any]], arcs: list[dict[str, Any]]) -> list[str]:
    ids = [node["id"] for node in nodes]
    incoming = {node_id: 0 for node_id in ids}
    outgoing = {node_id: [] for node_id in ids}
    for arc in arcs:
        outgoing[arc["from"]].append(arc["to"])
        incoming[arc["to"]] += 1
    queue = [node_id for node_id in ids if incoming[node_id] == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for nxt in outgoing[node]:
            incoming[nxt] -= 1
            if incoming[nxt] == 0:
                queue.append(nxt)
    return order if len(order) == len(ids) else []
