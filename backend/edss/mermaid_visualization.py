from __future__ import annotations

from typing import Any


DRAW_KEYWORDS = [
    "vẽ",
    "draw",
    "diagram",
    "sơ đồ",
    "cây",
    "tree",
    "graph",
    "network",
    "mô hình",
    "minh họa",
    "decision tree",
    "đường đi",
    "hàng đợi",
    "markov chain",
    "dp stages",
    "inventory model",
    "branch-and-bound",
]


def mermaid_visualization_gate(problem: dict[str, Any], problem_type: str | None = None) -> dict[str, Any]:
    kind = problem_type or problem.get("problem_type") or ""
    description = problem.get("context", {}).get("description", "") or ""
    lowered = description.lower()
    explicit = any(keyword in lowered for keyword in DRAW_KEYWORDS)
    diagram_type = _diagram_type(kind, problem, lowered)
    needs = explicit or _always_visual(kind, problem)
    required, available, missing = _diagram_slots(kind, problem, diagram_type)
    return {
        "needs_mermaid": needs,
        "diagram_type": diagram_type,
        "chapter": _chapter(kind),
        "diagram_purpose": _diagram_purpose(kind, diagram_type),
        "required_data": required,
        "available_data": available,
        "missing_data": missing,
        "can_draw_now": not missing,
    }


def attach_mermaid_if_needed(problem: dict[str, Any], result: dict[str, Any], problem_type: str | None = None) -> dict[str, Any]:
    gate = mermaid_visualization_gate(problem, problem_type)
    if not gate["needs_mermaid"]:
        return result
    report = result.get("markdown_report", "")
    if "```mermaid" in report:
        result["mermaid_gate"] = gate
        return result
    diagram = build_mermaid_diagram(problem, problem_type or problem.get("problem_type") or "", gate)
    result["mermaid_gate"] = gate
    result["markdown_report"] = diagram + "\n\n" + report
    return result


def build_mermaid_diagram(problem: dict[str, Any], problem_type: str, gate: dict[str, Any] | None = None) -> str:
    gate = gate or mermaid_visualization_gate(problem, problem_type)
    chart = _chart(problem_type, problem, gate["diagram_type"])
    missing = gate.get("missing_data", [])
    note = "- Thiếu dữ liệu: " + ", ".join(missing) if missing else "- Diagram dùng dữ liệu đã trích xuất từ mô hình."
    return (
        "## 1. Nhận dạng biểu đồ cần vẽ\n\n"
        f"- Dạng toán: {problem_type or problem.get('problem_type', 'unknown')}\n"
        f"- Loại biểu đồ Mermaid: {gate['diagram_type']}\n"
        f"- Lý do chọn loại biểu đồ: {gate['diagram_purpose']}\n\n"
        "## 2. Mermaid diagram\n\n"
        f"{chart}\n\n"
        "## 3. Ghi chú ngắn\n\n"
        f"{note}\n"
    )


def _diagram_type(kind: str, problem: dict[str, Any], lowered: str) -> str:
    if kind in {"decision_tree", "decision_theory"}:
        return "decision_tree"
    if kind in {"shortest_path", "max_flow", "min_cost_flow", "network_flow", "network_modelling", "transshipment"}:
        return "network_graph"
    if kind == "queueing_theory":
        return "queue_structure"
    if kind == "markov_processes":
        return "markov_state_transition"
    if kind == "dynamic_programming":
        return "dp_stage_state"
    if kind in {"inventory", "inventory_theory"}:
        return "inventory_workflow"
    if kind == "integer_programming":
        return "branch_and_bound" if "branch" in lowered else "integer_programming_workflow"
    if kind == "nonlinear_programming":
        return "kkt_workflow"
    if kind == "game_theory":
        return "game_theory_workflow"
    if kind == "linear_programming":
        return "lp_graphical_workflow" if len(problem.get("variables", []) or problem.get("variable_names", []) or problem.get("c", [])) == 2 else "simplex_workflow"
    return "model_workflow"


def _always_visual(kind: str, problem: dict[str, Any]) -> bool:
    return kind in {"network_modelling", "shortest_path", "max_flow", "min_cost_flow", "network_flow", "queueing_theory", "markov_processes"} or bool(problem.get("decision_tree"))


def _diagram_slots(kind: str, problem: dict[str, Any], diagram_type: str) -> tuple[list[str], list[str], list[str]]:
    if diagram_type == "network_graph":
        graph = problem.get("graph", {})
        required = ["nodes_or_edges"]
        available = ["edges"] if graph.get("edges") else []
        return required, available, [] if available else ["graph.edges"]
    if diagram_type == "markov_state_transition":
        spec = problem.get("markov") or problem.get("markov_chain") or {}
        matrix = spec.get("transition_matrix") or problem.get("transition_matrix")
        states = spec.get("states") or problem.get("markov_states") or problem.get("states")
        required = ["states", "transition_matrix"]
        available = []
        if states:
            available.append("states")
        if matrix:
            available.append("transition_matrix")
        missing = [item for item in required if item not in available]
        return required, available, missing
    if diagram_type == "decision_tree":
        required = ["decision_tree_or_alternatives"]
        available = ["decision_tree"] if problem.get("decision_tree") else []
        if problem.get("alternatives"):
            available.append("alternatives")
        return required, available, [] if available else ["decision alternatives/tree"]
    return [], [], []


def _chart(kind: str, problem: dict[str, Any], diagram_type: str) -> str:
    if diagram_type == "network_graph":
        return _network_chart(problem)
    if diagram_type == "markov_state_transition":
        return _markov_chart(problem)
    if diagram_type == "queue_structure":
        return _queue_chart(problem)
    if diagram_type == "decision_tree":
        return _decision_tree_chart(problem)
    if diagram_type == "dp_stage_state":
        return _dp_chart(problem)
    if diagram_type == "inventory_workflow":
        return _inventory_chart()
    if diagram_type == "integer_programming_workflow":
        return _integer_chart()
    if diagram_type == "branch_and_bound":
        return _branch_and_bound_chart()
    if diagram_type == "kkt_workflow":
        return _kkt_chart()
    if diagram_type == "game_theory_workflow":
        return _game_chart()
    if diagram_type == "simplex_workflow":
        return _simplex_chart()
    if diagram_type == "lp_graphical_workflow":
        return _lp_graphical_chart()
    return _model_chart()


def _network_chart(problem: dict[str, Any]) -> str:
    graph = problem.get("graph", {})
    edges = graph.get("edges") or []
    if not edges:
        return "```mermaid\ngraph LR\n    A[\"[?]\"] --> B[\"[?]\"]\n```"
    lines = ["```mermaid", "graph LR"]
    for edge in edges:
        src = _node_id(edge.get("from", "?"))
        dst = _node_id(edge.get("to", "?"))
        label = edge.get("capacity", edge.get("cost", edge.get("weight", edge.get("distance", ""))))
        arrow = "-->" if edge.get("directed", True) else "---"
        if label != "":
            lines.append(f'    {src}["{_label(edge.get("from", "?"))}"] {arrow}|"{_label(label)}"| {dst}["{_label(edge.get("to", "?"))}"]')
        else:
            lines.append(f'    {src}["{_label(edge.get("from", "?"))}"] {arrow} {dst}["{_label(edge.get("to", "?"))}"]')
    lines.append("```")
    return "\n".join(lines)


def _markov_chart(problem: dict[str, Any]) -> str:
    spec = problem.get("markov") or problem.get("markov_chain") or {}
    matrix = spec.get("transition_matrix") or problem.get("transition_matrix") or []
    raw_states = spec.get("states") or problem.get("markov_states") or problem.get("states") or []
    states = [item.get("name", str(item)) if isinstance(item, dict) else str(item) for item in raw_states]
    if not matrix:
        return "```mermaid\nstateDiagram-v2\n    S1 --> S2: [?]\n```"
    if not states:
        states = [f"s{i + 1}" for i in range(len(matrix))]
    lines = ["```mermaid", "stateDiagram-v2"]
    for i, row in enumerate(matrix):
        for j, prob in enumerate(row):
            if float(prob) > 0:
                lines.append(f"    {_node_id(states[i])} --> {_node_id(states[j])}: {_label(prob)}")
    lines.append("```")
    return "\n".join(lines)


def _queue_chart(problem: dict[str, Any]) -> str:
    servers = problem.get("servers") or (problem.get("queueing") or {}).get("servers") or 1
    if int(servers or 1) <= 1 and not problem.get("optimize_servers"):
        return '```mermaid\ngraph LR\n    A["Arrivals<br/>rate lambda"] --> Q["Queue"]\n    Q --> S["Single server<br/>rate mu"]\n    S --> D["Departures"]\n```'
    return '```mermaid\ngraph LR\n    A["Arrivals<br/>rate lambda"] --> Q["Single queue"]\n    Q --> S1["Server 1<br/>rate mu"]\n    Q --> S2["Server 2<br/>rate mu"]\n    Q --> Ss["Server s<br/>rate mu"]\n    S1 --> D["Departures"]\n    S2 --> D\n    Ss --> D\n```'


def _decision_tree_chart(problem: dict[str, Any]) -> str:
    if problem.get("decision_tree"):
        lines = ["```mermaid", "flowchart LR"]
        for node in problem["decision_tree"]:
            node_id = _node_id(node.get("id", "?"))
            label = _label(node.get("label", node.get("id", "?")))
            shape_open, shape_close = ("[", "]") if node.get("node_type") == "decision" else ("((", "))") if node.get("node_type") == "chance" else ("[", "]")
            lines.append(f'    {node_id}{shape_open}"{label}"{shape_close}')
        for node in problem["decision_tree"]:
            for child in node.get("children", []):
                lines.append(f'    {_node_id(node.get("id", "?"))} --> {_node_id(child)}')
        lines.append("```")
        return "\n".join(lines)
    return '```mermaid\nflowchart LR\n    D0["Decision: choose alternative"] --> C1(("Chance: state of nature"))\n    C1 --> T1["Terminal payoff [?]"]\n```'


def _dp_chart() -> str:
    return '```mermaid\ngraph LR\n    S1["Stage n<br/>State s_n"] -->|"Decision x_n"| S2["Stage n+1<br/>State s_{n+1}"]\n    S2 -->|"Decision x_{n+1}"| S3["Stage n+2<br/>State s_{n+2}"]\n```'


def _inventory_chart() -> str:
    return '```mermaid\nflowchart LR\n    A["Inventory at Q"] --> B["Demand reduces inventory"]\n    B --> C["Inventory reaches reorder point ROP"]\n    C --> D["Place order"]\n    D --> E["Lead time L"]\n    E --> F["Replenishment arrives"]\n    F --> A\n```'


def _integer_chart() -> str:
    return '```mermaid\nflowchart TD\n    A["IP model"] --> B["Identify integer and binary variables"]\n    B --> C["Build linear objective and constraints"]\n    C --> D["Add logical/linking constraints"]\n    D --> E["Solve MILP / branch-and-bound"]\n    E --> F["Verify integrality and feasibility"]\n```'


def _branch_and_bound_chart() -> str:
    return '```mermaid\nflowchart TD\n    A["LP relaxation"] --> B{"Integer solution?"}\n    B -->|"Yes"| C["Optimal IP solution"]\n    B -->|"No"| D["Choose fractional variable x_k"]\n    D --> E["Branch: x_k <= floor(x_k*)"]\n    D --> F["Branch: x_k >= ceil(x_k*)"]\n    E --> G["Solve LP relaxation"]\n    F --> H["Solve LP relaxation"]\n```'


def _kkt_chart() -> str:
    return '```mermaid\nflowchart TD\n    A["NLP problem"] --> B["Identify variables, objective, constraints"]\n    B --> C["Normalize constraints g_i(x) <= 0"]\n    C --> D["Build Lagrangian"]\n    D --> E["Write KKT conditions"]\n    E --> F["Solve active-set cases"]\n    F --> G["Verify local/global optimum"]\n```'


def _game_chart() -> str:
    return '```mermaid\nflowchart TD\n    A["Payoff matrix"] --> B["Identify players and strategies"]\n    B --> C["Check zero-sum or non-zero-sum"]\n    C --> D["Compute maximin and minimax"]\n    D --> E{"Saddle point?"}\n    E -->|"Yes"| F["Pure strategy solution"]\n    E -->|"No"| G["Dominance reduction and mixed strategy"]\n```'


def _simplex_chart() -> str:
    return '```mermaid\nflowchart TD\n    A["LP model"] --> B["Convert to standard form"]\n    B --> C["Add slack/surplus/artificial variables"]\n    C --> D["Build initial tableau"]\n    D --> E{"Optimality condition met?"}\n    E -->|"No"| F["Choose entering variable"]\n    F --> G["Ratio test: choose leaving variable"]\n    G --> H["Pivot"]\n    H --> E\n    E -->|"Yes"| I["Read optimal solution"]\n```'


def _lp_graphical_chart() -> str:
    return '```mermaid\nflowchart TD\n    A["Extract variables x, y"] --> B["Build objective function"]\n    B --> C["Build linear constraints"]\n    C --> D["Find boundary lines"]\n    D --> E["Find feasible region"]\n    E --> F["Find corner points"]\n    F --> G["Evaluate objective at each corner"]\n    G --> H["Select optimum"]\n```'


def _model_chart() -> str:
    return '```mermaid\nflowchart TD\n    A["Extract data"] --> B["Build model"]\n    B --> C["Choose solver"]\n    C --> D["Verify result"]\n```'


def _chapter(kind: str) -> str:
    return {
        "linear_programming": "1",
        "integer_programming": "2",
        "nonlinear_programming": "3",
        "network_modelling": "4",
        "shortest_path": "4",
        "max_flow": "4",
        "min_cost_flow": "4",
        "inventory_theory": "5",
        "inventory": "5",
        "queueing_theory": "6",
        "decision_tree": "7",
        "game_theory": "8",
        "dynamic_programming": "9",
        "markov_processes": "10",
    }.get(kind, "?")


def _diagram_purpose(kind: str, diagram_type: str) -> str:
    if "tree" in diagram_type:
        return "tree"
    if "network" in diagram_type:
        return "network"
    if "markov" in diagram_type:
        return "state_transition"
    if diagram_type in {"simplex_workflow", "kkt_workflow", "game_theory_workflow", "lp_graphical_workflow"}:
        return "algorithm"
    return "model"


def _node_id(value: Any) -> str:
    text = str(value)
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    if not cleaned or cleaned[0].isdigit():
        cleaned = "N_" + cleaned
    return cleaned


def _label(value: Any) -> str:
    return str(value).replace('"', "'").replace("$", "USD")
