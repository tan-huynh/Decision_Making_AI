from __future__ import annotations

from typing import Any


KEYWORDS = {
    "linear_programming": ["linear programming", "lp", "simplex", "resource", "capacity", "production mix", "allocate", "constraint", "profit", "cost minimization", "phân bổ", "công suất", "sản xuất", "ràng buộc", "maximize", "minimize", "subject to"],
    "transportation_assignment": ["transport", "shipment", "warehouse", "supply", "demand", "logistics", "transshipment", "vận tải", "kho", "cung", "cầu", "cij", "truyền tải", "assign", "assignment", "hungarian", "matching", "worker", "machine", "job", "crew", "phân công", "ghép", "người-việc"],
    "integer_programming": ["integer programming", "mixed integer", "mip", "binary variable", "fixed charge", "facility location", "capital budgeting", "nguyên", "nhị phân", "yes/no", "open", "close", "knapsack", "set covering", "scheduling"],
    "nonlinear_programming": ["nonlinear", "kkt", "convex", "quadratic", "phi tuyến", "lồi", "xy", "x^2", "sqrt", "log", "exponential", "utility maximization", "geometric packing"],
    "network_modelling": ["shortest path", "dijkstra", "bellman-ford", "minimum spanning tree", "maximum flow", "minimum-cost flow", "route", "routing", "distance", "đường đi", "đường đi ngắn", "tuyến", "luồng cực đại", "nodes", "arcs", "edges", "kruskal", "prim", "cpm", "pert", "project network"],
    "inventory_theory": ["eoq", "economic order quantity", "quantity discount", "reorder point", "safety stock", "newsvendor", "lot size", "mức đặt hàng", "điểm đặt hàng", "holding cost", "order cost", "shortage", "backorder", "lead time"],
    "queueing_theory": ["m/m/1", "m/m/c", "queue", "queueing", "waiting line", "arrival rate", "service rate", "hàng đợi", "tốc độ đến", "tốc độ phục vụ", "servers", "cashiers", "stations", "poisson", "exponential service", "utilization"],
    "decision_theory": ["chance", "state of nature", "payoff", "expected value", "kỳ vọng", "regret", "uncertainty", "bất định", "xác suất", "probability", "bayes", "posterior", "hậu nghiệm", "hình cây", "biến cố", "emV", "evpi", "evsi", "minimax", "maximin", "maximax", "expected utility"],
    "game_theory": ["game theory", "players", "strategies", "payoff matrix", "zero-sum", "saddle point", "mixed strategy", "dominance", "2x2 game", "2xm game", "minimax"],
    "dynamic_programming": ["dynamic programming", "quy hoạch động", "bellman", "stage", "sequential", "policy", "multi-stage", "giai đoạn", "states", "recursive", "multiperiod", "resource allocation", "replacement problem", "backward induction"],
    "markov_processes": ["markov", "transition matrix", "steady-state", "absorbing", "n-step", "ma trận chuyển", "trạng thái ổn định", "first passage time", "stationary distribution", "long-run cost", "pi_n", "pi_0"],
    "hybrid_problem": ["hybrid", "uncertainty + lp", "inventory + queueing", "network + lp", "decision tree + bayes", "markov + cost"],
    "unknown": ["unknown", "insufficient information"],
}


def classify_problem(description: str, structured: dict[str, Any] | None = None) -> dict[str, Any]:
    text = description.lower()
    scores = {kind: 0 for kind in KEYWORDS}
    for kind, words in KEYWORDS.items():
        scores[kind] += sum(1 for word in words if word in text)
    structured = structured or {}
    
    # Structural Evidence
    if structured.get("assignment_costs") or (structured.get("supplies") and structured.get("demands")):
        scores["transportation_assignment"] += 5
    if structured.get("graph", {}).get("edges"):
        scores["network_modelling"] += 4
    if structured.get("variables") and structured.get("constraints") and structured.get("objective"):
        scores["linear_programming"] += 5
    if structured.get("ip") or structured.get("integrality"):
        scores["integer_programming"] += 8
    if any(item.get("variable_type") in {"integer", "binary", "zero-one"} for item in structured.get("variables", [])):
        scores["integer_programming"] += 6
    if structured.get("nlp") or structured.get("radii"):
        scores["nonlinear_programming"] += 8
    if structured.get("decision_tree") or structured.get("payoff_matrix"):
        scores["decision_theory"] += 5
    if (
        structured.get("probability_tree")
        or structured.get("bayes")
        or structured.get("diagnostic_decision")
        or structured.get("imperfect_information_decision")
        or structured.get("forklift_decision")
        or structured.get("independent_probabilities")
    ):
        scores["decision_theory"] += 7
    if structured.get("alternatives") and structured.get("states") and structured.get("payoff_matrix"):
        scores["decision_theory"] += 6
    if structured.get("game"):
        scores["game_theory"] += 10
    if structured.get("players") and structured.get("strategies") and structured.get("payoff_matrix"):
        scores["game_theory"] += 7
    if structured.get("inventory"):
        scores["inventory_theory"] += 6
    if (
        structured.get("annual_demand") or structured.get("demand")
    ) and (structured.get("order_cost") or structured.get("ordering_cost")) and (
        structured.get("holding_cost") or structured.get("holding_cost_rate")
    ):
        scores["inventory_theory"] += 5
    if structured.get("queueing"):
        scores["queueing_theory"] += 6
    if structured.get("arrival_rate") and structured.get("service_rate"):
        scores["queueing_theory"] += 4
    if structured.get("transition_matrix") or structured.get("markov_chain") or structured.get("markov"):
        scores["markov_processes"] += 6
    if structured.get("stages") and structured.get("states") and structured.get("decisions"):
        scores["dynamic_programming"] += 6
        
    best = max(scores.items(), key=lambda item: item[1])
    if best[1] == 0:
        return {
            "problem_type": "unknown",
            "confidence": 0.25,
            "scores": scores,
            "reason": "Không tìm thấy đủ tín hiệu Keyword và Structural.",
        }
    total = sum(scores.values())
    confidence = round(best[1] / max(1, total), 2)
    # Boost confidence if structural signals are very strong
    if best[1] > 5:
        confidence = min(0.95, confidence + 0.3)
        
    return {
        "problem_type": best[0],
        "confidence": confidence,
        "scores": scores,
        "reason": f"Matched {best[1]} indicators for {best[0]}.",
    }


def missing_data_questions(problem: dict[str, Any]) -> list[str]:
    questions: list[str] = []
    kind = problem.get("problem_type") or classify_problem(problem.get("context", {}).get("description", ""), problem).get("problem_type")
    
    if kind == "linear_programming" or kind == "nonlinear_programming":
        if not problem.get("variables"):
            questions.append("Các biến quyết định là gì và giới hạn dưới/trên của từng biến?")
        if not problem.get("objective"):
            questions.append("Hàm mục tiêu cần maximize/minimize là gì và hệ số của từng biến?")
        if not problem.get("constraints"):
            questions.append("Các ràng buộc tài nguyên/công suất/ngân sách/thời gian là gì?")
    if kind == "integer_programming":
        ip = problem.get("ip", {})
        if not (problem.get("variables") or problem.get("variable_names") or ip.get("variable_names")):
            questions.append("Các biến quyết định là gì và giới hạn dưới/trên của từng biến?")
        if not (problem.get("objective") or problem.get("c") or ip.get("c")):
            questions.append("Hàm mục tiêu cần maximize/minimize là gì và hệ số của từng biến?")
        if not (problem.get("constraints") or problem.get("A_ub") or problem.get("A_eq") or ip.get("A_ub") or ip.get("A_eq")):
            questions.append("Các ràng buộc tài nguyên/công suất/ngân sách/thời gian là gì?")
        if not (
            problem.get("integrality")
            or ip.get("integrality")
            or any(item.get("variable_type") in {"integer", "binary", "zero-one"} for item in problem.get("variables", []))
        ):
            questions.append("Biến nào là integer và biến nào là binary/zero-one?")
            
    if kind == "decision_theory":
        if (
            problem.get("probability_tree")
            or problem.get("bayes")
            or problem.get("diagnostic_decision")
            or problem.get("imperfect_information_decision")
            or problem.get("forklift_decision")
            or problem.get("independent_probabilities")
        ):
            return questions or ["Dữ liệu đủ cho mô hình xác suất ban đầu; vui lòng xác nhận giả định độc lập/phụ thuộc."]
        if not problem.get("states"):
            questions.append("Các trạng thái bất định và xác suất tương ứng là gì?")
        if not problem.get("payoff_matrix"):
            questions.append("Payoff/cost/loss của từng phương án dưới từng trạng thái là gì?")
            
    if kind == "transportation_assignment":
        if not problem.get("assignment_costs") and not (problem.get("supplies") and problem.get("demands")):
            questions.append("Cần ma trận chi phí vận chuyển/phân công và thông tin cung cầu (nếu có).")
            
    if kind == "network_modelling":
        if not problem.get("graph", {}).get("edges"):
            questions.append("Danh sách node, cạnh và cost/distance/time/capacity của từng cạnh là gì?")
            
    if kind == "inventory_theory":
        if "demand" not in problem and "annual_demand" not in problem:
            questions.append("Demand (nhu cầu) là bao nhiêu và đơn vị thời gian là gì?")
        if "order_cost" not in problem and "ordering_cost" not in problem:
            questions.append("Chi phí đặt hàng (Ordering cost / Setup cost) là bao nhiêu?")
        if "holding_cost" not in problem:
            questions.append("Chi phí lưu kho (Holding cost) là bao nhiêu?")
            
    if kind == "queueing_theory":
        queueing = problem.get("queueing", {})
        if "arrival_rate" not in problem and "arrival_rate" not in queueing:
            questions.append("Tốc độ đến (Arrival rate - λ) là bao nhiêu và đơn vị là gì?")
        if "service_rate" not in problem and "service_rate" not in queueing:
            questions.append("Tốc độ phục vụ (Service rate - μ) là bao nhiêu và đơn vị là gì?")
        if "servers" not in problem and "servers" not in queueing and not problem.get("optimize_servers") and not queueing.get("optimize_servers"):
            questions.append("Số server s là bao nhiêu, hoặc phạm vi số server cần thử để tối ưu là gì?")
            
    if kind == "markov_processes":
        markov = problem.get("markov") or problem.get("markov_chain") or {}
        if not (problem.get("transition_matrix") or markov.get("transition_matrix")):
            questions.append("Ma trận chuyển trạng thái (Transition matrix) P là gì?")
        if not (problem.get("markov_states") or problem.get("states") or markov.get("states")):
            questions.append("Các trạng thái của Markov chain là gì?")
        if not (problem.get("time_step") or markov.get("time_step")):
            questions.append("Một bước chuyển Markov tương ứng với đơn vị thời gian nào?")
        if not (problem.get("requested_outputs") or markov.get("requested_outputs")):
            questions.append("Cần tính output nào: n-step, stationary, first passage, absorbing hay long-run cost?")
            
    if kind == "game_theory":
        game = problem.get("game", {})
        if game and game.get("payoff_matrix") and game.get("row_strategies") and game.get("column_strategies"):
            return questions or ["Dữ liệu đủ cho Game Theory recognition gate và solver."]
        if not problem.get("payoff_matrix"):
            questions.append("Ma trận lợi ích (Payoff matrix) của các người chơi là gì?")
            
    if kind == "dynamic_programming":
        if problem.get("resource_allocation"):
            spec = problem.get("resource_allocation", {})
            if spec.get("total_resource") is None:
                questions.append("Tổng tài nguyên ban đầu của bài DP là bao nhiêu?")
            if not spec.get("stage_returns"):
                questions.append("Bảng lợi ích/chi phí theo lượng phân bổ cho từng stage là gì?")
        elif problem.get("stages"):
            incomplete = any(
                not stage.get("states") or not stage.get("actions") or not stage.get("transitions") or not stage.get("rewards")
                for stage in problem.get("stages", [])
            )
            if incomplete:
                questions.append("Mỗi stage cần có states, actions, transition function và reward/cost function.")
        elif not (problem.get("demands") and ("order_cost" in problem or "holding_cost" in problem)):
            questions.append("Thông tin về stages, states, decisions, transition function, reward/cost function và boundary condition là gì?")

    if kind == "unknown":
        questions.append("Không thể nhận dạng bài toán. Vui lòng cung cấp thêm thông tin rõ ràng hơn.")
        
    return questions or ["Dữ liệu có vẻ đã đầy đủ, hệ thống đang kiểm tra logic trước khi chạy solver."]
