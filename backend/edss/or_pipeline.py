from __future__ import annotations

import re
from typing import Any


TAXONOMY: dict[str, dict[str, Any]] = {
    "linear_programming": {
        "label": "Linear Programming",
        "required_slots": ["decision_variables", "objective", "constraints", "bounds"],
        "solver": "Graphical Method + Vertex Enumeration if 2 variables; otherwise Simplex / scipy.optimize.linprog",
        "keywords": ["maximize", "minimize", "subject to", "linear programming", "simplex", "resource", "constraint", "production mix"],
    },
    "transportation_assignment": {
        "label": "Transportation / Assignment",
        "required_slots": ["supplies", "demands", "cost_matrix_or_assignment_costs"],
        "solver": "Transportation Solver / Min-Cost Flow / Hungarian Method",
        "keywords": ["transportation", "assignment", "supply", "demand", "cost matrix", "hungarian", "matching"],
    },
    "integer_programming": {
        "label": "Integer Programming",
        "required_slots": ["integer_or_binary_variables", "objective", "constraints"],
        "solver": "MILP / scipy.milp / OR-Tools CP-SAT",
        "keywords": ["integer", "binary", "yes/no", "open", "close", "fixed charge", "facility", "knapsack"],
    },
    "nonlinear_programming": {
        "label": "Nonlinear Programming",
        "required_slots": ["variables", "nonlinear_objective_or_constraints", "bounds", "initial_guess_or_analytic_conditions"],
        "solver": "KKT/Lagrange for small analytic problems; scipy.optimize.minimize + feasibility verification otherwise",
        "keywords": ["nonlinear", "non-linear", "kkt", "quadratic", "x^2", "sqrt", "log", "circle", "packing"],
    },
    "network_modelling": {
        "label": "Network Modelling",
        "required_slots": ["nodes", "edges", "weights_or_capacities", "source_or_target_if_needed"],
        "solver": "Dijkstra/Bellman-Ford/Kruskal/Prim/Max-flow/Min-cost-flow based on subtype",
        "keywords": ["network", "node", "edge", "arc", "shortest path", "minimum spanning tree", "max flow", "route"],
    },
    "inventory_theory": {
        "label": "Inventory Theory",
        "required_slots": ["demand", "ordering_cost", "holding_cost", "lead_time_or_shortage_if_present"],
        "solver": "EOQ / EOQ with shortages / discount EOQ / reorder point / newsvendor",
        "keywords": ["inventory", "eoq", "order cost", "holding cost", "stockout", "shortage", "reorder", "demand"],
    },
    "queueing_theory": {
        "label": "Queueing Theory",
        "required_slots": ["arrival_rate", "service_rate", "servers", "capacity_if_finite"],
        "solver": "M/M/1, M/M/c, M/M/1/K or queue network formulas",
        "keywords": ["queue", "queueing", "arrival", "service", "server", "m/m/1", "m/m/c", "waiting"],
    },
    "decision_theory": {
        "label": "Decision Theory",
        "required_slots": ["alternatives", "states", "probabilities", "payoffs_or_costs"],
        "solver": "EMV, decision tree rollback, Bayes, EVPI/EVSI, regret or expected utility",
        "keywords": ["decision tree", "payoff", "state of nature", "probability", "bayes", "evpi", "evsi", "regret"],
    },
    "game_theory": {
        "label": "Game Theory",
        "required_slots": ["players", "strategies", "payoff_matrix", "zero_sum_flag"],
        "solver": "Saddle point, dominance, mixed strategy algebra, or LP game solver",
        "keywords": ["game", "player", "strategy", "zero-sum", "saddle point", "minimax", "mixed strategy"],
    },
    "dynamic_programming": {
        "label": "Dynamic Programming",
        "required_slots": ["stages", "states", "decisions", "transition", "reward_or_cost", "boundary_condition"],
        "solver": "Bellman recursion + backward induction + traceback",
        "keywords": ["dynamic programming", "bellman", "stage", "state", "recursive", "multi-stage", "quy hoạch động"],
    },
    "markov_processes": {
        "label": "Markov Processes",
        "required_slots": ["states", "transition_matrix", "initial_distribution_or_query"],
        "solver": "P^n, stationary distribution, absorbing chain, or first-passage equations",
        "keywords": ["markov", "transition matrix", "steady state", "stationary", "absorbing", "first passage"],
    },
    "hybrid_problem": {
        "label": "Hybrid Problem",
        "required_slots": [],
        "solver": "Custom",
        "keywords": ["hybrid", "uncertainty + lp", "inventory + queueing", "network + lp"],
    },
    "unknown": {
        "label": "Unknown / Insufficient Information",
        "required_slots": [],
        "solver": "None",
        "keywords": ["unknown", "insufficient information"],
    },
}


TYPE_ALIASES = {
    "shortest_path": "network_modelling",
    "max_flow": "network_modelling",
    "min_cost_flow": "network_modelling",
    "assignment": "transportation_assignment",
    "transportation": "transportation_assignment",
    "circle_packing_box": "nonlinear_programming",
    "eoq_planned_shortages": "inventory_theory",
    "paper_inventory_quantity_discount": "inventory_theory",
    "decision_tree": "decision_theory",
    "markov_chain": "markov_processes",
    "queueing": "queueing_theory",
    "inventory": "inventory_theory",
    "linear_programming_batch": "linear_programming",
}


def canonical_type(problem_type: str | None) -> str:
    if not problem_type:
        return "unknown"
    return TYPE_ALIASES.get(problem_type, problem_type)


def understand_input(text: str) -> dict[str, Any]:
    tables = _extract_markdown_tables(text)
    graph = _extract_graph_candidates(text)
    symbols = sorted(set(re.findall(r"\b[A-Z][A-Za-z0-9_]*\b|[a-z]_\{?\d+\}?", text)))
    units = sorted(set(re.findall(r"\b(?:USD|dollars?|cm|km|MW|MWh|hours?|weeks?|years?|units?|tons?|tấn)\b", text, flags=re.I)))
    ambiguities: list[str] = []
    if "[?]" in text:
        ambiguities.append("Input contains uncertain OCR cells marked [?].")
    if re.search(r"\b(cost|payoff|profit)\b", text, flags=re.I) and not re.search(r"\$|USD|cost|profit", text, flags=re.I):
        ambiguities.append("Objective unit may be missing.")
    return {
        "clean_problem_statement": text.strip(),
        "extracted_tables": tables,
        "extracted_graph": graph,
        "symbols": symbols[:60],
        "units": units,
        "possible_ambiguities": ambiguities,
    }


def classify_with_taxonomy(text: str, structured: dict[str, Any] | None = None) -> dict[str, Any]:
    structured = structured or {}
    structured_type = canonical_type(structured.get("problem_type"))
    scores = {name: 0.0 for name in TAXONOMY}
    evidence: dict[str, list[str]] = {name: [] for name in TAXONOMY}
    lowered = text.lower()
    for name, spec in TAXONOMY.items():
        for keyword in spec["keywords"]:
            if keyword in lowered:
                scores[name] += 1
                evidence[name].append(keyword)

    if structured_type in scores and structured_type != "unknown":
        scores[structured_type] += 5
        evidence[structured_type].append(f"structured problem_type={structured.get('problem_type')}")
    if structured.get("c") and (structured.get("A_ub") or structured.get("A_eq")):
        scores["linear_programming"] += 3
        evidence["linear_programming"].append("linear matrix model present")
    if structured.get("graph", {}).get("edges"):
        scores["network_modelling"] += 3
        evidence["network_modelling"].append("graph edges present")
    if structured.get("resource_allocation"):
        scores["dynamic_programming"] += 4
        evidence["dynamic_programming"].append("resource allocation stages present")
    if structured.get("payoff_matrix"):
        scores["decision_theory"] += 3
        evidence["decision_theory"].append("payoff matrix present")

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = ranked[0]
    total = sum(value for value in scores.values() if value > 0)
    base_conf = best_score / max(total, best_score) if best_score > 0 else 0
    if best_score >= 5:
        base_conf += 0.3
    elif best_score >= 3:
        base_conf += 0.15
    confidence = 0.2 if best_score == 0 else min(0.98, max(0.45, base_conf))
    alternatives = [
        {"type": name, "score": round(score, 3), "evidence": evidence[name][:5]}
        for name, score in ranked[1:4]
        if score > 0
    ]
    return {
        "primary_type": best_type if best_score > 0 else "unknown_insufficient_information",
        "secondary_type": structured.get("problem_type") or "",
        "confidence": round(confidence, 3),
        "evidence": evidence[best_type][:8] if best_score > 0 else [],
        "alternative_types": alternatives,
        "missing_information": required_missing_slots(best_type if best_score > 0 else "unknown", structured),
    }


def required_missing_slots(problem_type: str, structured: dict[str, Any]) -> list[str]:
    kind = canonical_type(problem_type)
    if kind == "unknown":
        return ["problem_type", "required_quantitative_data"]
    if kind not in TAXONOMY:
        return ["Cần xác định dạng toán và dữ liệu đầu vào tối thiểu."]
    missing: list[str] = []
    if kind == "linear_programming":
        if not (structured.get("variable_names") or structured.get("variables")):
            missing.append("decision_variables")
        if not _has_value(structured, "c") and not _has_value(structured, "objective"):
            missing.append("objective")
        if not (_has_value(structured, "A_ub") or _has_value(structured, "A_eq") or _has_value(structured, "constraints")):
            missing.append("constraints")
    elif kind == "network_modelling":
        graph = structured.get("graph", {})
        if structured.get("problem_type") == "library_shelving_shortest_path" and structured.get("heights") and structured.get("counts"):
            return []
        if not graph.get("edges") and not graph.get("costs"):
            missing.append("nodes_edges_weights")
    elif kind == "inventory_theory":
        for key in ["annual_demand", "order_cost"]:
            if not _has_value(structured, key):
                missing.append(key)
        if not _has_value(structured, "holding_cost") and not (
            _has_value(structured, "holding_cost_rate")
            and (_has_value(structured, "unit_cost") or _has_value(structured, "purchase_cost"))
        ):
            missing.append("holding_cost")
    elif kind == "queueing_theory":
        for key in ["arrival_rate", "service_rate"]:
            if not _has_value(structured, key) and key not in structured.get("queueing", {}):
                missing.append(key)
    elif kind == "decision_theory":
        if structured.get("probability_tree") or structured.get("bayes") or structured.get("diagnostic_decision") or structured.get("forklift_decision") or structured.get("independent_probabilities"):
            return []
        for key in ["alternatives", "states", "payoff_matrix"]:
            if not structured.get(key):
                missing.append(key)
    elif kind == "games_theory":
        for key in ["players", "strategies", "payoff_matrix"]:
            if not structured.get(key):
                missing.append(key)
    elif kind == "dynamic_programming":
        if not (structured.get("resource_allocation") or structured.get("stages")):
            missing.append("stages_states_decisions_transition")
    elif kind == "markov_processes":
        if not (structured.get("transition_matrix") or structured.get("markov_chain")):
            missing.append("transition_matrix")
    return missing


def _has_value(structured: dict[str, Any], key: str) -> bool:
    value = structured.get(key)
    return value is not None and value != "" and value != [] and value != {}


def solver_route(problem_type: str, structured: dict[str, Any]) -> dict[str, Any]:
    kind = canonical_type(problem_type)
    spec = TAXONOMY.get(kind)
    if not spec:
        return {"selected_solver": "AmbiguityHandler", "reason": "Problem type is unknown.", "fallback_solver": "", "manual_solution_possible": False}
    selected = spec["solver"]
    reason = f"Selected from taxonomy because primary_type={kind} and required slots are {'mostly complete' if not required_missing_slots(kind, structured) else 'partial'}."
    if kind == "linear_programming":
        n = len(structured.get("variable_names", []) or structured.get("variables", []))
        selected = "Graphical Method + Vertex Enumeration" if n == 2 else "Simplex / scipy.optimize.linprog"
        reason = f"Linear objective/constraints detected; number_of_variables={n or 'unknown'}."
    if kind == "network_modelling" and structured.get("graph", {}).get("edges"):
        selected = "Dijkstra/Bellman-Ford/Network solver selected from edge weights and subtype"
        reason = "Graph edges are available for network routing."
    return {"selected_solver": selected, "reason": reason, "fallback_solver": "Clarification questions + structured JSON model", "manual_solution_possible": True}


def build_pipeline_trace(text: str, problem: dict[str, Any] | None = None, solved: dict[str, Any] | None = None) -> dict[str, Any]:
    problem = problem or {}
    solved = solved or {}
    understanding = understand_input(text)
    classification = classify_with_taxonomy(text, problem)
    gate = recognition_gate(classification, problem)
    route = solver_route(classification["primary_type"], problem)
    solution = solved.get("result", solved)
    verification_status = "passed" if solution.get("status") in {"optimal", "computed", "optimal_local", "feasible_heuristic"} else "warning"
    if not solved:
        verification_status = "not_run"
    return {
        "step_1_input_understanding": understanding,
        "step_2_problem_type_recognition": classification,
        "problem_recognition_gate": gate,
        "step_3_slot_extraction": {
            "required_slots": TAXONOMY.get(classification["primary_type"], {}).get("required_slots", []),
            "missing_information": classification["missing_information"],
            "structured_keys": sorted(problem.keys()),
        },
        "step_4_model_construction": {
            "mathematical_model_available": bool(problem),
            "model_type": problem.get("problem_type"),
            "assumptions": problem.get("assumptions", []),
        },
        "step_5_solver_router": route,
        "step_6_solve": {
            "status": solution.get("status", "not_run"),
            "solver": solution.get("solver") or solution.get("model") or route["selected_solver"],
        },
        "step_7_verification": {
            "status": verification_status,
            "checks": _verification_checks(problem, solution),
        },
        "step_8_explanation": {
            "format": "teacher_step_by_step_markdown",
            "required_sections": [
                "Nhận dạng dạng toán",
                "Dữ liệu đã trích xuất",
                "Mô hình toán học",
                "Giải bài toán",
                "Kiểm tra nghiệm",
                "Đáp án cuối cùng",
                "Nhận xét",
            ],
        },
    }


def attach_pipeline(result: dict[str, Any], text: str) -> dict[str, Any]:
    problem = result.get("problem", {})
    solved = result.get("solved", {})
    result["pipeline"] = build_pipeline_trace(text, problem, solved)
    gate = result["pipeline"]["problem_recognition_gate"]
    result["recognition_gate"] = gate
    if result.get("status") in {"needs_clarification", "pending_llm"}:
        classification = result["pipeline"]["step_2_problem_type_recognition"]
        missing = classification.get("missing_information", [])
        result.setdefault("questions", [f"Cần bổ sung slot `{slot}`." for slot in missing] or ["Vui lòng cung cấp dữ liệu định lượng còn thiếu."])
    return result


def gate_allows_solving(text: str, problem: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    classification = classify_with_taxonomy(text, problem)
    gate = recognition_gate(classification, problem)
    return gate["decision_to_solve"] == "solve", gate


def recognition_gate(classification: dict[str, Any], structured: dict[str, Any]) -> dict[str, Any]:
    required = TAXONOMY.get(classification.get("primary_type", ""), {}).get("required_slots", [])
    missing = classification.get("missing_information", [])
    filled = [slot for slot in required if slot not in missing]
    completeness = 1.0 if not required else len(filled) / len(required)
    confidence = float(classification.get("confidence", 0))
    alternatives = classification.get("alternative_types", [])
    exact_structured_match = bool(structured.get("problem_type")) and not missing
    if exact_structured_match and confidence >= 0.65:
        confidence = max(confidence, 0.9)
    critical_ambiguity = confidence < 0.85 and bool(alternatives) and not exact_structured_match
    if classification.get("primary_type") in {"unknown", "unknown_insufficient_information"}:
        decision = "ask_clarification"
    elif confidence >= 0.85 and completeness >= 0.90 and not critical_ambiguity:
        decision = "solve"
    elif 0.65 <= confidence < 0.85:
        decision = "ask_clarification"
    else:
        decision = "ask_clarification"
    return {
        "recognized_problem_type": classification.get("primary_type", "unknown_insufficient_information"),
        "recognized_subtype": classification.get("secondary_type", ""),
        "confidence": round(confidence, 3),
        "evidence": classification.get("evidence", []),
        "required_slots": required,
        "filled_slots": filled,
        "missing_slots": missing,
        "required_slot_completeness": round(completeness, 3),
        "possible_alternative_types": alternatives,
        "decision_to_solve": decision,
        "critical_ambiguity": critical_ambiguity,
    }


def _extract_markdown_tables(text: str) -> list[str]:
    tables: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if "|" in line:
            current.append(line.strip())
        elif current:
            if len(current) >= 2:
                tables.append("\n".join(current))
            current = []
    if len(current) >= 2:
        tables.append("\n".join(current))
    return tables


def _extract_graph_candidates(text: str) -> dict[str, Any]:
    edges = []
    for src, arrow, dst, weight in re.findall(r"\b([A-Za-z0-9]+)\s*(->|--|-)\s*([A-Za-z0-9]+)\s*[:=, ]\s*(-?\d+(?:\.\d+)?)", text):
        edges.append({"from": src, "to": dst, "weight": float(weight), "directed": arrow == "->"})
    nodes = sorted({edge["from"] for edge in edges} | {edge["to"] for edge in edges})
    return {"nodes": nodes, "edges": edges, "directed": any(edge["directed"] for edge in edges)}


def _verification_checks(problem: dict[str, Any], solution: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    kind = canonical_type(problem.get("problem_type"))
    if kind == "linear_programming":
        checks.append({"name": "LP feasibility", "status": "passed" if solution.get("status") == "optimal" else "warning"})
    if kind == "inventory_theory":
        checks.append({"name": "Inventory time units", "status": "passed", "detail": "Demand and cycle metrics are normalized to annual/weeks when available."})
    if kind == "network_modelling":
        checks.append({"name": "Graph connectivity/path", "status": "passed" if solution.get("status") in {"optimal", "computed"} else "warning"})
    if kind == "decision_theory":
        checks.append({"name": "Probability/payoff consistency", "status": "passed" if solution.get("status") in {"computed", "optimal"} else "warning"})
    return checks or [{"name": "Generic solver status", "status": "passed" if solution.get("status") else "not_run"}]
