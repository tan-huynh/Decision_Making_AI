from __future__ import annotations

from typing import Any


def solve_resource_allocation_dp(
    total_resource: int,
    stage_returns: list[list[float]],
    resource_name: str = "resource",
    sense: str = "maximize",
) -> dict[str, Any]:
    if total_resource < 0 or not stage_returns:
        raise ValueError("Resource allocation DP requires total resource and stage return tables.")
    stages = len(stage_returns)
    maximize = sense != "minimize"
    impossible = float("-inf") if maximize else float("inf")
    dp = [[impossible] * (total_resource + 1) for _ in range(stages + 1)]
    choice = [[0] * (total_resource + 1) for _ in range(stages + 1)]
    calculation = [[""] * (total_resource + 1) for _ in range(stages + 1)]
    dp[0][0] = 0.0
    for stage in range(1, stages + 1):
        returns = stage_returns[stage - 1]
        for used in range(total_resource + 1):
            best_value = float("-inf") if maximize else float("inf")
            best_amount = 0
            options = []
            for amount, profit in enumerate(returns):
                prev_val = dp[stage - 1][used - amount] if amount <= used else impossible
                if amount <= used and prev_val != impossible:
                    prev_val = dp[stage - 1][used - amount]
                    value = prev_val + float(profit)
                    if stage == 1:
                        options.append(f"\\underset{{x={amount}}}{{{profit}}}")
                    else:
                        options.append(f"\\underset{{x={amount}}}{{{profit} + {prev_val}}}")
                        
                    if (maximize and value > best_value) or ((not maximize) and value < best_value):
                        best_value = value
                        best_amount = amount
            dp[stage][used] = best_value
            choice[stage][used] = best_amount
            
            if options:
                if stage == 1:
                    calculation[stage][used] = f"R_1({best_amount}) = {best_value}"
                else:
                    calculation[stage][used] = f"\\max\\{{ {', '.join(options)} \\}} = {best_value}"

    allocation = []
    remaining = total_resource
    for stage in range(stages, 0, -1):
        amount = choice[stage][remaining]
        allocation.append(
            {
                "stage": stage,
                resource_name: amount,
                "profit": stage_returns[stage - 1][amount],
            }
        )
        remaining -= amount
    allocation.reverse()
    tables = [
        {
            "stage": stage,
            "values": {resource: dp[stage][resource] for resource in range(total_resource + 1) if dp[stage][resource] != impossible},
            "choice": {resource: choice[stage][resource] for resource in range(total_resource + 1) if dp[stage][resource] != impossible},
            "calc": {resource: calculation[stage][resource] for resource in range(total_resource + 1) if dp[stage][resource] != impossible},
        }
        for stage in range(1, stages + 1)
    ]
    md = f"### Báo cáo kết quả bài toán Quy hoạch động (Dynamic Programming)\n\n"
    objective_word = "tối đa hóa" if maximize else "tối thiểu hóa"
    operator = "\\max" if maximize else "\\min"
    md += f"**Mục tiêu:** Phân bổ tối ưu {total_resource} {resource_name} qua {stages} giai đoạn để {objective_word} tổng giá trị mục tiêu.\n\n"

    md += "#### 1. Mô hình toán học\n\n"
    md += f"- **Trạng thái $s$**: Số lượng {resource_name} đã được phân bổ.\n"
    md += f"- **Quyết định $x$**: Số lượng {resource_name} được phân bổ cho giai đoạn hiện tại.\n"
    md += f"- **Phương trình Bellman**: $F_k(s) = {operator}_{{x \\le s}} \\{{ \\text{{value}}_k(x) + F_{{k-1}}(s-x) \\}}$\n\n"

    md += "#### 2. Sơ đồ lộ trình phân bổ tối ưu\n\n"
    md += "```mermaid\ngraph LR\n"
    md += '  Start(("Bắt đầu (0)"))\n'
    current_node = "Start"
    cum_resource = 0
    for idx, item in enumerate(allocation):
        cum_resource += int(item[resource_name])
        next_node = f'N{idx}("GĐ {item["stage"]} ({cum_resource})")'
        md += f'  {current_node} -->|"+{item[resource_name]} {resource_name} (Giá trị: {item["profit"]})"| {next_node}\n'
        current_node = next_node
    md += f'  {current_node} --> End(("Tổng giá trị: {dp[stages][total_resource]}"))\n'
    md += "```\n\n"

    md += "#### 3. Bảng tổng hợp phương án tối ưu\n\n"
    md += f"| Giai đoạn | Lượng {resource_name} phân bổ | Giá trị đạt được |\n"
    md += "|:---:|---:|---:|\n"
    for item in allocation:
        md += f"| Giai đoạn {item['stage']} | **{item[resource_name]}** | {item['profit']} |\n"
    md += f"\n**Tổng giá trị tối ưu:** {dp[stages][total_resource]}\n\n"

    md += "#### 4. Bảng phân tích chi tiết quy hoạch động (DP Tables)\n\n"
    for table in tables:
        md += f"**Giai đoạn {table['stage']}**\n\n"
        md += f"| Lượng tài nguyên đã dùng ($s$) | Chi tiết phép tính ${operator} \\{{ \\text{{value}}_k(x) + F_{{k-1}}(s-x) \\}}$ | Giá trị tối ưu $F(s)$ | Lượng chọn tối ưu ($x$) |\n"
        md += "|:---:|:---|---:|---:|\n"
        for s in sorted(table['values'].keys()):
            md += f"| {s} | ${table['calc'][s]}$ | **{table['values'][s]}** | **{table['choice'][s]}** |\n"
        md += "\n"

    return {
        "status": "optimal",
        "solver": "resource_allocation_dynamic_programming",
        "sense": sense,
        "objective_value": dp[stages][total_resource],
        "allocation": allocation,
        "dp_tables": tables,
        "markdown_report": md,
        "recommendation": f"Allocate {resource_name} by stage as {[item[resource_name] for item in allocation]} to {sense} total value {dp[stages][total_resource]:.3f}.",
    }


def solve_finite_horizon_dp(problem: dict[str, Any]) -> dict[str, Any]:
    stages = problem.get("stages", [])
    if not stages:
        raise ValueError("DP requires stages with states/actions/transitions.")
    value_next: dict[str, float] = {}
    policy: list[dict[str, Any]] = []
    for stage_index in range(len(stages) - 1, -1, -1):
        stage = stages[stage_index]
        value_current: dict[str, float] = {}
        stage_policy: dict[str, str] = {}
        for state in stage.get("states", []):
            best_value = float("-inf")
            best_action = None
            for action in stage.get("actions", []):
                transitions = stage.get("transitions", {}).get(state, {}).get(action, [])
                expected = float(stage.get("rewards", {}).get(state, {}).get(action, 0))
                for transition in transitions:
                    expected += float(transition.get("probability", 1)) * value_next.get(transition.get("to"), 0)
                if expected > best_value:
                    best_value = expected
                    best_action = action
            value_current[state] = best_value if best_action is not None else 0.0
            stage_policy[state] = best_action or "none"
        value_next = value_current
        policy.append({"stage": stage_index, "value": value_current, "policy": stage_policy})
    policy.reverse()
    return {"status": "computed", "solver": "finite_horizon_backward_induction", "policy": policy, "initial_value": policy[0]["value"] if policy else {}}


def solve_inventory_dp(problem: dict[str, Any]) -> dict[str, Any]:
    """Finite-horizon deterministic inventory DP.

    Input keys:
      periods, demands, max_inventory, initial_inventory, order_cost, holding_cost,
      shortage_cost, fixed_order_cost optional.
    State is beginning inventory. Decision is order quantity.
    """
    demands = [int(v) for v in problem.get("demands", [])]
    if not demands:
        raise ValueError("Inventory DP cần demands theo từng kỳ.")
    periods = int(problem.get("periods", len(demands)))
    max_inventory = int(problem.get("max_inventory", max(max(demands), problem.get("initial_inventory", 0)) + sum(demands)))
    initial = int(problem.get("initial_inventory", 0))
    unit_order = float(problem.get("order_cost", 0))
    holding = float(problem.get("holding_cost", 0))
    shortage = float(problem.get("shortage_cost", 1e6))
    fixed = float(problem.get("fixed_order_cost", 0))
    max_order = int(problem.get("max_order", max_inventory + max(demands)))
    states = range(0, max_inventory + 1)
    next_value = {s: 0.0 for s in states}
    policy = []
    for t in range(periods - 1, -1, -1):
        demand = demands[t]
        current_value = {}
        current_policy = {}
        rows = []
        for inv in states:
            best_cost = float("inf")
            best_q = 0
            best_end = 0
            for q in range(max_order + 1):
                available = inv + q
                end_inv = max(0, available - demand)
                lost = max(0, demand - available)
                if end_inv > max_inventory:
                    continue
                cost = unit_order * q + (fixed if q > 0 else 0) + holding * end_inv + shortage * lost + next_value[end_inv]
                if cost < best_cost:
                    best_cost = cost
                    best_q = q
                    best_end = end_inv
            current_value[inv] = best_cost
            current_policy[inv] = best_q
            rows.append({"state_inventory": inv, "order_q": best_q, "ending_inventory": best_end, "value": round(best_cost, 6)})
        policy.append({"period": t + 1, "demand": demand, "rows": rows, "value": current_value, "policy": current_policy})
        next_value = current_value
    policy.reverse()
    rollout = []
    inv = initial
    total_cost = 0.0
    for stage in policy:
        q = int(stage["policy"][inv])
        end = max(0, inv + q - stage["demand"])
        lost = max(0, stage["demand"] - inv - q)
        period_cost = unit_order * q + (fixed if q > 0 else 0) + holding * end + shortage * lost
        total_cost += period_cost
        rollout.append({"period": stage["period"], "begin_inventory": inv, "order_q": q, "demand": stage["demand"], "ending_inventory": end, "shortage": lost, "period_cost": round(period_cost, 6)})
        inv = end
    md = "### Báo cáo Inventory Dynamic Programming\n\n"
    md += "State `s_t` là tồn kho đầu kỳ; decision `q_t` là lượng đặt hàng. Bellman: `F_t(s)=min_q cost_t(s,q)+F_{t+1}(s')`.\n\n"
    md += "| Period | Begin Inv | Order | Demand | End Inv | Shortage | Cost |\n|---:|---:|---:|---:|---:|---:|---:|\n"
    for row in rollout:
        md += f"| {row['period']} | {row['begin_inventory']} | {row['order_q']} | {row['demand']} | {row['ending_inventory']} | {row['shortage']} | {row['period_cost']:.4f} |\n"
    md += f"\n**Minimum total cost:** {total_cost:.4f}\n"
    return {
        "status": "optimal",
        "solver": "deterministic_inventory_dynamic_programming",
        "objective_value": round(total_cost, 6),
        "rollout": rollout,
        "dp_tables": policy,
        "markdown_report": md,
    }
