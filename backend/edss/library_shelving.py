from __future__ import annotations

import math
from typing import Any

from .graph_report import mermaid_dag


def solve_library_shelving_shortest_path(
    heights: list[float],
    counts: list[int],
    thickness: float,
    fixed_cost: float,
    area_cost: float,
) -> dict[str, Any]:
    if not heights or len(heights) != len(counts):
        raise ValueError("Cần heights và counts cùng độ dài.")
    items = sorted(zip(heights, counts), key=lambda item: item[0])
    heights = [float(h) for h, _ in items]
    counts = [int(c) for _, c in items]
    n = len(heights)

    arcs: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(i, n):
            shelf_height = heights[j]
            books = sum(counts[i : j + 1])
            area = shelf_height * thickness * books
            cost = fixed_cost + area_cost * area
            arcs.append(
                {
                    "from": i,
                    "to": j + 1,
                    "cover_heights": heights[i : j + 1],
                    "shelf_height": shelf_height,
                    "books": books,
                    "area": area,
                    "cost": cost,
                }
            )

    dp = [math.inf] * (n + 1)
    prev: list[dict[str, Any] | None] = [None] * (n + 1)
    label_rows: list[dict[str, Any]] = [{"node": "0", "candidates": "X_0 = (0, 0)", "best": "0", "predecessor": "0"}]
    dp[0] = 0.0
    for to_node in range(1, n + 1):
        candidates = []
        for arc in [a for a in arcs if a["to"] == to_node]:
            if not math.isfinite(dp[arc["from"]]):
                continue
            candidate = dp[arc["from"]] + arc["cost"]
            candidates.append({"from": arc["from"], "arc_cost": arc["cost"], "total": candidate})
            if candidate < dp[to_node]:
                dp[to_node] = candidate
                prev[to_node] = arc
        label_rows.append(
            {
                "node": _node_label(to_node, heights),
                "candidates": "; ".join(
                    f"{cand['arc_cost']:,.0f} + {dp[cand['from']]:,.0f} from {_node_label(cand['from'], heights)} = {cand['total']:,.0f}"
                    for cand in candidates
                ),
                "best": f"{dp[to_node]:,.0f}",
                "predecessor": _node_label(prev[to_node]["from"], heights) if prev[to_node] else "",
            }
        )

    path = []
    node = n
    while node > 0 and prev[node] is not None:
        arc = prev[node]
        path.append(arc)
        node = arc["from"]
    path.reverse()

    markdown = _library_shelving_report(heights, counts, thickness, fixed_cost, area_cost, arcs, path, dp[n], label_rows)
    return {
        "status": "optimal",
        "solver": "dag_shortest_path_library_shelving",
        "objective_value": round(dp[n], 6),
        "nodes": list(range(n + 1)),
        "arcs": arcs,
        "label_rows": label_rows,
        "path": path,
        "shelves": [
            {
                "shelf_height": arc["shelf_height"],
                "stores_book_heights": arc["cover_heights"],
                "books": arc["books"],
                "area": round(arc["area"], 6),
                "cost": round(arc["cost"], 6),
            }
            for arc in path
        ],
        "markdown_report": markdown,
    }


def _library_shelving_report(
    heights: list[float],
    counts: list[int],
    thickness: float,
    fixed_cost: float,
    area_cost: float,
    arcs: list[dict[str, Any]],
    path: list[dict[str, Any]],
    total_cost: float,
    label_rows: list[dict[str, Any]],
) -> str:
    node_labels = ["0"] + [f"{height:g}" for height in heights]
    graph_arcs = []
    for arc in arcs:
        src = node_labels[arc["from"]]
        dst = node_labels[arc["to"]]
        graph_arcs.append({"from": src, "to": dst, "label": f"${arc['cost']:,.0f}"})
    highlighted = {(node_labels[arc["from"]], node_labels[arc["to"]]) for arc in path}
    graph = mermaid_dag(node_labels, graph_arcs, highlighted=highlighted)

    lines = [
        "# Library Shelving Problem - Shortest Path",
        "",
        "**Dạng bài:** Shortest path trên DAG để chia nhóm book heights liên tiếp thành các shelving units.",
        "",
        "## 1. Mô hình shortest path",
        "",
        "Sắp xếp book heights tăng dần. Node `i` biểu diễn đã lưu xong các nhóm height đầu tiên đến trước vị trí `i`.",
        "",
        "Arc `(i,j)` nghĩa là xây một shelving unit có height bằng book height lớn nhất trong nhóm `i..j-1`, dùng để chứa tất cả books trong nhóm đó.",
        "",
        "Chi phí arc:",
        "",
        "`fixed_cost + area_cost * shelf_height * thickness * number_of_books_in_group`",
        "",
        f"Ở đây fixed cost = {fixed_cost:g}, area cost = {area_cost:g}, thickness = {thickness:g} cm.",
        "",
        "## 2. Construction graph",
        "",
        graph,
        "",
        "Các cạnh màu xanh là shortest path được chọn. Mỗi arc label là tổng chi phí xây một shelving unit cho nhóm book heights tương ứng.",
        "",
        "## 3. Dữ liệu",
        "",
        "| Book height | Number of books |",
        "|---:|---:|",
    ]
    for h, c in zip(heights, counts):
        lines.append(f"| {h:g} | {c} |")

    lines.extend(["", "## 4. Arc cost table", "", "| Arc | Cost formula | Covers heights | Shelf height | Books | Area | Cost |", "|---|---|---|---:|---:|---:|---:|"])
    for arc in arcs:
        label = f"C{_node_label(arc['from'], heights)}-{_node_label(arc['to'], heights)}"
        covers = ", ".join(f"{h:g}" for h in arc["cover_heights"])
        formula = f"({arc['books']} books × {arc['shelf_height']:g} cm × {thickness:g} cm × ${area_cost:g}/cm²) + ${fixed_cost:,.0f}"
        lines.append(f"| {label} | {formula} | {covers} | {arc['shelf_height']:g} | {arc['books']} | {arc['area']:.2f} | ${arc['cost']:,.2f} |")

    lines.extend(["", "## 5. Ford shortest-path label calculation", "", "| Node | Candidate calculations | Best label X_j | Predecessor |", "|---|---|---:|---|"])
    for row in label_rows:
        lines.append(f"| {row['node']} | {row['candidates']} | {row['best']} | {row['predecessor']} |")

    lines.extend(["", "## 6. Feasible path table", "", "| Feasible path | Shelving plan | Total cost |", "|---|---|---:|"])
    for candidate_path in _enumerate_feasible_paths(arcs, len(heights)):
        plan = "; ".join(f"{arc['shelf_height']:g} cm shelf for {', '.join(f'{h:g}' for h in arc['cover_heights'])}" for arc in candidate_path)
        cost = sum(float(arc["cost"]) for arc in candidate_path)
        path_label_candidate = " → ".join([_node_label(candidate_path[0]["from"], heights)] + [_node_label(arc["to"], heights) for arc in candidate_path])
        lines.append(f"| {path_label_candidate} | {plan} | ${cost:,.2f} |")

    lines.extend(["", "## 7. Shortest path solution", "", "| Step | Shelf height | Stores book heights | Books | Cost |", "|---:|---:|---|---:|---:|"])
    for idx, arc in enumerate(path, start=1):
        covers = ", ".join(f"{h:g}" for h in arc["cover_heights"])
        lines.append(f"| {idx} | {arc['shelf_height']:g} | {covers} | {arc['books']} | {arc['cost']:.2f} |")

    path_label = " → ".join([str(path[0]["from"])] + [str(arc["to"]) for arc in path]) if path else ""
    lines.extend(
        [
            "",
            f"**Shortest path:** `{path_label}`",
            "",
            f"**Minimum total cost:** `${total_cost:,.2f}`",
            "",
            "## 8. Khuyến nghị",
            "",
            "Xây các shelving units theo từng arc trên shortest path. Đây là quyết định tốt theo mô hình vì mọi cách chia nhóm liên tiếp đã được biểu diễn bằng arc và thuật toán DAG shortest path đã chọn tổng chi phí nhỏ nhất.",
        ]
    )
    return "\n".join(lines)


def _node_label(index: int, heights: list[float]) -> str:
    if index == 0:
        return "0"
    return f"{heights[index - 1]:g}"


def _enumerate_feasible_paths(arcs: list[dict[str, Any]], n: int) -> list[list[dict[str, Any]]]:
    outgoing: dict[int, list[dict[str, Any]]] = {}
    for arc in arcs:
        outgoing.setdefault(int(arc["from"]), []).append(arc)
    paths: list[list[dict[str, Any]]] = []

    def walk(node: int, current: list[dict[str, Any]]) -> None:
        if node == n:
            paths.append(list(current))
            return
        for arc in outgoing.get(node, []):
            current.append(arc)
            walk(int(arc["to"]), current)
            current.pop()

    walk(0, [])
    return paths
