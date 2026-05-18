from __future__ import annotations

from typing import Any


def mermaid_dag(
    nodes: list[str],
    arcs: list[dict[str, Any]],
    highlighted: set[tuple[str, str]] | None = None,
    direction: str = "LR",
) -> str:
    highlighted = highlighted or set()
    lines = ["```mermaid", f"graph {direction}"]
    for node in nodes:
        safe = _node_id(node)
        lines.append(f'  {safe}(("{node}"))')
    link_index = 0
    highlight_indexes = []
    for arc in arcs:
        src = str(arc["from"])
        dst = str(arc["to"])
        label = str(arc.get("label", arc.get("cost", "")))
        lines.append(f'  {_node_id(src)} -->|"{_escape(label)}"| {_node_id(dst)}')
        if (src, dst) in highlighted:
            highlight_indexes.append(link_index)
        link_index += 1
    lines.append("  classDef selected stroke:#34d399,stroke-width:4px,color:#f8fafc;")
    lines.append("  classDef normal stroke:#93c5fd,stroke-width:1.5px;")
    for idx in highlight_indexes:
        lines.append(f"  linkStyle {idx} stroke:#34d399,stroke-width:4px;")
    lines.append("```")
    return "\n".join(lines)


def _node_id(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in label)
    return f"N_{cleaned}"


def _escape(label: str) -> str:
    return label.replace('"', "'")
