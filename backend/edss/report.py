"""Report Generator — 15-section decision report template.

Generates structured Markdown reports from solver results with
full audit trail and decision quality assessment.
"""

from __future__ import annotations

import time
from typing import Any


def render_markdown_report(problem: dict[str, Any], solved: dict[str, Any]) -> str:
    """Generate a comprehensive 15-section decision report."""
    ctx = problem.get("context", {})
    result = solved.get("result", {})
    model = solved.get("model", {})
    validation = solved.get("validation", {})
    sensitivity = solved.get("sensitivity", {})
    risk = solved.get("risk", {})

    sections = [
        _section_header(ctx),
        _section_problem_statement(ctx),
        _section_decision_context(ctx),
        _section_data_inputs(problem),
        _section_assumptions(problem, model),
        _section_classification(solved),
        _section_mathematical_model(model),
        _section_solver_selection(result),
        _section_solver_result(result),
        _section_sensitivity(sensitivity, result),
        _section_risk_analysis(risk, result),
        _section_voi(result),
        _section_recommendation(solved),
        _section_limitations(validation, problem),
        _section_audit(solved),
    ]
    return "\n\n".join(s for s in sections if s)


def _section_header(ctx: dict[str, Any]) -> str:
    title = ctx.get("title", "Engineering Decision Report")
    return f"# {title}\n\n*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*"


def _section_problem_statement(ctx: dict[str, Any]) -> str:
    desc = ctx.get("description", "Chưa có mô tả.")
    return f"## 1. Problem Statement\n\n{desc}"


def _section_decision_context(ctx: dict[str, Any]) -> str:
    lines = [
        "## 2. Decision Context\n",
        f"- **Domain**: {ctx.get('domain', 'N/A')}",
        f"- **Decision maker**: {ctx.get('decision_maker', 'N/A')}",
        f"- **Objective**: {ctx.get('objective_direction', 'N/A')}",
        f"- **Unit**: {ctx.get('unit', 'N/A')}",
        f"- **Time horizon**: {ctx.get('time_horizon', 'N/A')}",
    ]
    return "\n".join(lines)


def _section_data_inputs(problem: dict[str, Any]) -> str:
    lines = ["## 3. Data Inputs\n"]

    variables = problem.get("variables", [])
    if variables:
        lines.append("### Decision Variables")
        lines.append("| Name | Type | Lower | Upper | Unit |")
        lines.append("|---|---|---|---|---|")
        for v in variables:
            lines.append(
                f"| {v.get('name', '')} | {v.get('variable_type', 'continuous')} "
                f"| {v.get('lower_bound', 0)} | {v.get('upper_bound', '∞')} "
                f"| {v.get('unit', '')} |"
            )

    constraints = problem.get("constraints", [])
    if constraints:
        lines.append("\n### Constraints")
        lines.append("| Name | Expression | RHS | Resource |")
        lines.append("|---|---|---|---|")
        for c in constraints:
            coeffs = c.get("coefficients", {})
            expr = " + ".join(f"{v}×{k}" for k, v in coeffs.items())
            lines.append(
                f"| {c.get('name', '')} | {expr} {c.get('operator', '≤')} {c.get('rhs', '')} "
                f"| {c.get('rhs', '')} | {c.get('resource', '')} |"
            )

    states = problem.get("states", [])
    if states:
        lines.append("\n### States of Nature")
        lines.append("| State | Probability |")
        lines.append("|---|---|")
        for s in states:
            lines.append(f"| {s.get('name', '')} | {s.get('probability', '')} |")

    return "\n".join(lines)


def _section_assumptions(problem: dict[str, Any], model: dict[str, Any]) -> str:
    assumptions = problem.get("assumptions", []) + model.get("assumptions", [])
    if not assumptions:
        return "## 4. Assumptions\n\n- Chưa liệt kê giả định cụ thể."
    items = "\n".join(f"- {a}" for a in dict.fromkeys(assumptions))
    return f"## 4. Assumptions\n\n{items}"


def _section_classification(solved: dict[str, Any]) -> str:
    kind = solved.get("problem_type", "N/A")
    labels = {
        "linear_programming": "Linear Programming (Quy hoạch tuyến tính)",
        "transportation": "Transportation Problem (Bài toán vận tải)",
        "assignment": "Assignment Problem (Bài toán phân công)",
        "shortest_path": "Shortest Path (Đường đi ngắn nhất)",
        "dynamic_programming": "Dynamic Programming (Quy hoạch động)",
        "decision_tree": "Decision Under Uncertainty (Quyết định dưới bất định)",
        "simulation_risk": "Simulation / Monte Carlo",
        "multi_objective": "Multi-Objective Decision (Nhiều mục tiêu)",
    }
    return f"## 5. Problem Classification\n\n**Type**: {labels.get(kind, kind)}"


def _section_mathematical_model(model: dict[str, Any]) -> str:
    formulation = model.get("formulation", "N/A")
    return f"## 6. Mathematical Model\n\n```\n{formulation}\n```"


def _section_solver_selection(result: dict[str, Any]) -> str:
    solver = result.get("solver", "N/A")
    status = result.get("status", "N/A")
    return f"## 7. Solver Selection\n\n- **Solver**: {solver}\n- **Status**: {status}"


def _section_solver_result(result: dict[str, Any]) -> str:
    lines = ["## 8. Solver Result\n"]
    status = result.get("status", "N/A")

    if status == "optimal":
        lines.append(f"**Objective value**: {result.get('objective_value', 'N/A')}")
        solution = result.get("solution", {})
        if solution:
            lines.append("\n### Optimal Solution")
            lines.append("| Variable | Value |")
            lines.append("|---|---|")
            for k, v in solution.items():
                lines.append(f"| {k} | {v} |")
        binding = result.get("binding_constraints", [])
        if binding:
            lines.append(f"\n**Binding constraints**: {', '.join(binding)}")
    elif status == "computed":
        results = result.get("results", [])
        if results:
            lines.append("### Expected Values")
            lines.append("| Alternative | Expected Value | Worst Case |")
            lines.append("|---|---|---|")
            for r in results:
                lines.append(f"| {r.get('alternative', '')} | {r.get('expected_value', ''):.4f} | {r.get('worst_case', ''):.4f} |")
        recommendation = result.get("recommendation")
        if recommendation:
            lines.append(f"\n**Recommendation**: {recommendation}")
    else:
        lines.append(f"Status: {status}")
        if result.get("message"):
            lines.append(f"Message: {result['message']}")

    return "\n".join(lines)


def _section_sensitivity(sensitivity: dict[str, Any], result: dict[str, Any]) -> str:
    lines = ["## 9. Sensitivity Analysis\n"]

    shadow = result.get("shadow_prices", sensitivity.get("shadow_prices", {}))
    if shadow:
        lines.append("### Shadow Prices")
        lines.append("| Constraint | Shadow Price |")
        lines.append("|---|---|")
        for k, v in shadow.items():
            lines.append(f"| {k} | {v} |")

    slacks = result.get("slacks", {})
    if slacks:
        lines.append("\n### Slacks")
        lines.append("| Constraint | Slack |")
        lines.append("|---|---|")
        for k, v in slacks.items():
            lines.append(f"| {k} | {v} |")

    tornado = sensitivity.get("tornado", [])
    if tornado:
        lines.append("\n### Tornado Chart Data (top 5)")
        lines.append("| Parameter | Low | High | Impact |")
        lines.append("|---|---|---|---|")
        for t in tornado[:5]:
            lines.append(f"| {t['parameter']} | {t['low']:.2f} | {t['high']:.2f} | {t['impact']:.2f} |")

    if len(lines) == 1:
        lines.append("Chưa có dữ liệu sensitivity analysis.")
    return "\n".join(lines)


def _section_risk_analysis(risk: dict[str, Any], result: dict[str, Any]) -> str:
    lines = ["## 10. Risk Analysis\n"]
    risks = risk.get("risks", [])
    if risks:
        lines.append("| Alternative | P5 (Downside) | P50 (Median) | P95 (Upside) |")
        lines.append("|---|---|---|---|")
        for r in risks:
            lines.append(
                f"| {r.get('alternative', '')} | {r.get('downside_proxy', 0):.2f} "
                f"| {r.get('median', 0):.2f} | {r.get('upside_proxy', 0):.2f} |"
            )
    else:
        lines.append("Chưa có dữ liệu risk analysis. Cần chạy simulation để tạo risk metrics.")
    return "\n".join(lines)


def _section_voi(result: dict[str, Any]) -> str:
    voi = result.get("voi", {})
    lines = ["## 11. Value of Information\n"]
    if voi:
        lines.append(f"- **EVwoPI**: {voi.get('EVwoPI', 'N/A')}")
        lines.append(f"- **EVwPI**: {voi.get('EVwPI', 'N/A')}")
        lines.append(f"- **EVPI**: {voi.get('EVPI', 'N/A')}")
        rec = voi.get("recommendation", "")
        if rec:
            lines.append(f"\n{rec}")
    else:
        lines.append("Không áp dụng cho loại bài toán này hoặc thiếu payoff matrix.")
    return "\n".join(lines)


def _section_recommendation(solved: dict[str, Any]) -> str:
    explanation = solved.get("recommendation_explanation", "")
    lines = [
        "## 12. Recommendation\n",
        explanation,
        "",
        "> **Lưu ý**: Quyết định tốt ≠ Outcome tốt. Kết quả may mắn không chứng minh quyết định đúng. "
        "Hệ thống đánh giá quy trình ra quyết định (model đúng, dữ liệu đủ, giả định rõ), "
        "không phải kết quả thực tế.",
    ]
    return "\n".join(lines)


def _section_limitations(validation: dict[str, Any], problem: dict[str, Any]) -> str:
    lines = ["## 13. Limitations & Required Additional Data\n"]

    warnings = validation.get("warnings", [])
    if warnings:
        lines.append("### Warnings")
        for w in warnings:
            lines.append(f"- ⚠️ {w}")

    info = validation.get("info", [])
    if info:
        lines.append("\n### Information")
        for i in info:
            lines.append(f"- ℹ️ {i}")

    missing = validation.get("errors", [])
    if missing:
        lines.append("\n### Missing/Invalid Data")
        for m in missing:
            lines.append(f"- ❌ {m}")

    if len(lines) == 1:
        lines.append("Không có cảnh báo từ model validator.")
    return "\n".join(lines)


def _section_audit(solved: dict[str, Any]) -> str:
    steps = solved.get("audit_steps", 0)
    return (
        "## 14. Audit Trail\n\n"
        f"- Total analysis steps logged: {steps}\n"
        "- Solver results are deterministic and reproducible with the same inputs.\n"
        "- Monte Carlo simulations use fixed seeds for reproducibility."
    )
